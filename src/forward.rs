//! Transparent forwarding to the Node backend.

use std::sync::Arc;

use axum::body::Body;
use axum::http::uri::Uri;
use axum::http::Request;
use axum::response::Response;
use hyper_util::client::legacy::connect::HttpConnector;

use crate::AppState;

/// HTTP client to the Node backend (loopback). Connections are pooled by
/// hyper's legacy client.
#[derive(Clone)]
pub struct NodeClient {
    http: hyper_util::client::legacy::Client<HttpConnector, http_body_util::Full<bytes::Bytes>>,
}

impl NodeClient {
    pub fn new() -> Self {
        let http =
            hyper_util::client::legacy::Client::builder(hyper_util::rt::TokioExecutor::new())
                .build_http::<http_body_util::Full<bytes::Bytes>>();
        Self { http }
    }
}

impl Default for NodeClient {
    fn default() -> Self {
        Self::new()
    }
}

/// Proxy a request to Node as-is. The request body is buffered (requests to
/// ST are JSON/multipart and modest in size; the OOM problem lives in the
/// character-list read path which is served locally).
pub async fn proxy(state: Arc<AppState>, req: Request<Body>) -> Response {
    let (mut parts, body) = req.into_parts();

    // Rewrite URI authority to the Node backend, preserving path+query.
    let path_and_query = parts
        .uri
        .path_and_query()
        .map(|pq| pq.as_str().to_string())
        .unwrap_or_else(|| parts.uri.path().to_string());
    let uri = Uri::builder()
        .scheme("http")
        .authority(state.cfg.node_addr().as_str())
        .path_and_query(path_and_query)
        .build()
        .expect("static scheme/authority; path preserved");

    // Buffer request body.
    let body_bytes = match axum::body::to_bytes(body, 512 * 1024 * 1024).await {
        Ok(b) => b,
        Err(e) => {
            tracing::warn!("failed to read request body for forwarding: {e}");
            return axum::http::StatusCode::BAD_REQUEST.into_response();
        }
    };

    parts.uri = uri;
    parts.headers.remove("host");
    if let Ok(host_hdr) =
        format!("host-header:{}", state.cfg.node_addr()).parse::<axum::http::HeaderValue>()
    {
        parts.headers.insert("x-forwarded-host", host_hdr);
    }

    let outgoing = Request::from_parts(parts, http_body_util::Full::new(body_bytes));

    match state.node.http.request(outgoing).await {
        Ok(resp) => {
            let (parts, body) = resp.into_parts();
            let body = axum::body::Body::new(body);
            Response::from_parts(parts, body)
        }
        Err(e) => {
            tracing::error!("forward to node {} failed: {e}", state.cfg.node_addr());
            axum::http::StatusCode::BAD_GATEWAY.into_response()
        }
    }
}

/// Helper to build responses from errors.
pub trait IntoResponse {
    fn into_response(self) -> Response;
}

impl IntoResponse for axum::http::StatusCode {
    fn into_response(self) -> Response {
        Response::builder()
            .status(self)
            .body(Body::empty())
            .unwrap()
    }
}
