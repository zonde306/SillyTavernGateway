//! Transparent forwarding to the Node backend, plus the upstream client used
//! by the local chat-completions endpoint.

use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::http::uri::Uri;
use axum::http::Request;
use axum::response::IntoResponse;
use axum::response::Response;
use hyper_rustls::HttpsConnector;
use hyper_util::client::legacy::connect::HttpConnector;

use crate::AppState;

/// Upstream providers sit behind public DNS: a stalled TCP connect must not
/// hold a client for the OS default (~2 minutes on some targets).
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

type HttpClient = hyper_util::client::legacy::Client<
    HttpsConnector<HttpConnector>,
    http_body_util::Full<bytes::Bytes>,
>;

/// HTTP client to the Node backend (loopback). Connections are pooled by
/// hyper's legacy client.
#[derive(Clone)]
pub struct NodeClient {
    pub http: HttpClient,
}

impl NodeClient {
    pub fn new() -> Self {
        Self {
            http: build_client(),
        }
    }
}

impl Default for NodeClient {
    fn default() -> Self {
        Self::new()
    }
}

/// A client that speaks both `http://` and `https://`.
///
/// Node uses `fetch`, which handles TLS transparently, so a provider URL is
/// simply whatever the user typed. Hyper needs the connector to be told about
/// TLS explicitly — with a plain `HttpConnector`, every `https://` provider
/// fails at connect time with `client error (Connect)`, which looks exactly
/// like an unreachable host.
fn build_client() -> HttpClient {
    let mut http = HttpConnector::new();
    http.enforce_http(false); // let the TLS layer see https:// URIs
    http.set_connect_timeout(Some(CONNECT_TIMEOUT));

    let https = hyper_rustls::HttpsConnectorBuilder::new()
        .with_webpki_roots()
        .https_or_http()
        .enable_http1()
        .wrap_connector(http);

    hyper_util::client::legacy::Client::builder(hyper_util::rt::TokioExecutor::new()).build(https)
}

/// Proxy a request to Node as-is. The request body is buffered (requests to
/// ST are JSON/multipart and modest in size; the OOM problem lives in the
/// character-list read path which is served locally).
pub async fn proxy(state: Arc<AppState>, req: Request<Body>) -> Response {
    let (parts, body) = req.into_parts();

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

    let mut builder = Request::builder().method(parts.method.clone()).uri(uri);

    if let Some(headers) = builder.headers_mut() {
        *headers = parts.headers.clone();
        // The upstream authority changed; a stale Host header would confuse
        // Node's vhost checks and the CSRF origin validation.
        headers.remove(axum::http::header::HOST);
        if let Ok(value) = axum::http::HeaderValue::from_str(&state.cfg.node_addr()) {
            headers.insert(axum::http::header::HOST, value);
        }
        if let Some(original) = parts.headers.get(axum::http::header::HOST) {
            headers.insert(
                axum::http::HeaderName::from_static("x-forwarded-host"),
                original.clone(),
            );
        }
    }

    let forwarded = match builder.body(http_body_util::Full::new(bytes::Bytes::from(
        body_bytes.to_vec(),
    ))) {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!("failed to build forwarded request: {e}");
            return axum::http::StatusCode::BAD_REQUEST.into_response();
        }
    };

    match state.node.http.request(forwarded).await {
        Ok(resp) => {
            let (mut parts, body) = resp.into_parts();
            parts.headers.insert(
                axum::http::header::HeaderName::from_static("via"),
                axum::http::HeaderValue::from_static("st-gateway"),
            );
            Response::from_parts(parts, Body::new(body))
        }
        Err(e) => {
            tracing::warn!("Node backend unreachable: {e}");
            axum::http::StatusCode::BAD_GATEWAY.into_response()
        }
    }
}
