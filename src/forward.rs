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
use http_body_util::BodyExt as _;

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

/// Whether the forwarded request is expected to produce a streamed body.
///
/// Only the generation endpoints stream, and only when the body asks for it;
/// keeping every other request's `Accept-Encoding` intact preserves the
/// bandwidth win for character lists, i18n bundles and static assets.
fn wants_streaming_response(parts: &axum::http::request::Parts) -> bool {
    let path = parts.uri.path();
    matches!(
        path,
        "/api/backends/chat-completions/generate"
            | "/api/backends/text-completions/generate"
            | "/api/backends/kobold/generate"
    )
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
        // Event streams must reach the client chunk-by-chunk, so the Node
        // backend must not gzip them on our behalf.
        //
        // `server-main.js` mounts express `compression()`, and `text/event-stream`
        // is compressible, so Node happily wraps a streamed chat completion in
        // gzip. zlib then buffers whatever it cannot flush as a complete block
        // and Node never calls `flush()`, so a thinking model's keep-alive
        // comments (`: PING` from Google, and any byte we inject) pile up in
        // the compressor until the final payload arrives — the client sees a
        // stalled connection followed by everything at once. Verified against
        // this repo's own node_modules/compression: chunks that arrived 1 s
        // apart were delivered together after 3 s.
        //
        // Dropping `Accept-Encoding` makes the default filter decline
        // (`identity`), which is exactly what a byte-streaming proxy wants.
        // Node's own route still sees the original request; only the hop we
        // own is uncompressed. Everything else keeps its compression.
        if wants_streaming_response(&parts) {
            headers.remove(axum::http::header::ACCEPT_ENCODING);
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
            let streamed = is_event_stream(&parts);
            parts.headers.insert(
                axum::http::header::HeaderName::from_static("via"),
                axum::http::HeaderValue::from_static("st-gateway"),
            );
            if !streamed {
                return Response::from_parts(parts, Body::new(body));
            }
            // A forwarded chat completion is exactly as likely to stall as a
            // local one: Node sends nothing while the provider thinks, and
            // makersuite/vertex/claude all live on this path. Wrapping it here
            // keeps one heartbeat policy for every provider instead of
            // "works for openai, silent for gemini".
            streamed_response(state.as_ref(), parts, body)
        }
        Err(e) => {
            tracing::warn!("Node backend unreachable: {e}");
            axum::http::StatusCode::BAD_GATEWAY.into_response()
        }
    }
}

/// Whether the response is a byte stream that must reach the client live.
fn is_event_stream(parts: &axum::http::response::Parts) -> bool {
    matches!(
        parts
            .headers
            .get(axum::http::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .map(|v| v
                .split(';')
                .next()
                .unwrap_or("")
                .trim()
                .to_ascii_lowercase())
            .as_deref(),
        Some("text/event-stream")
    )
}

/// Wrap a forwarded event stream with the same priming token and wall-clock
/// heartbeat the local path uses.
fn streamed_response(
    state: &AppState,
    mut parts: axum::http::response::Parts,
    body: hyper::body::Incoming,
) -> Response {
    parts.headers.remove(axum::http::header::CONTENT_LENGTH);
    parts.headers.insert(
        axum::http::header::CACHE_CONTROL,
        axum::http::HeaderValue::from_static("no-cache"),
    );
    // Tell nginx/CF not to sit on the body. Node sets neither header, and a
    // buffering proxy in front of the gateway reintroduces the stall the
    // heartbeat exists to prevent.
    parts.headers.insert(
        axum::http::HeaderName::from_static("x-accel-buffering"),
        axum::http::HeaderValue::from_static("no"),
    );
    let interval = crate::heartbeat::interval_from_config(state.cfg.gateway.heartbeat_secs);
    let wrapped = crate::heartbeat::HeartbeatBody::new(
        body.map_err(|_| -> core::convert::Infallible {
            unreachable!("hyper incoming bodies never yield an error here")
        }),
        crate::chat_completions::Protocol::Sse,
        interval,
    );
    Response::from_parts(parts, Body::new(wrapped))
}

/// POST an `application/x-www-form-urlencoded` body and return the response
/// text. Used by the Vertex AI service-account token exchange, which is the
/// only provider setup that needs the network before the main request.
pub async fn post_form(url: &str, form: &str) -> Result<String, String> {
    let uri: Uri = url.parse().map_err(|e| format!("bad url {url}: {e}"))?;
    let authority = uri
        .authority()
        .ok_or_else(|| format!("url has no authority: {url}"))?
        .clone();
    let path_and_query = uri
        .path_and_query()
        .map(|p| p.as_str().to_string())
        .unwrap_or_else(|| "/".to_string());

    let request = Request::builder()
        .method("POST")
        .uri(path_and_query)
        .header(
            axum::http::header::CONTENT_TYPE,
            "application/x-www-form-urlencoded",
        )
        .header(axum::http::header::HOST, authority.as_str())
        .body(http_body_util::Full::new(bytes::Bytes::from(
            form.to_string().into_bytes(),
        )))
        .map_err(|e| format!("failed to build token request: {e}"))?;

    let response = NodeClient::new()
        .http
        .request(request)
        .await
        .map_err(|e| format!("token request failed: {e}"))?;
    let bytes = http_body_util::BodyExt::collect(response.into_body())
        .await
        .map_err(|e| format!("could not read token response: {e}"))?
        .to_bytes();
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}
