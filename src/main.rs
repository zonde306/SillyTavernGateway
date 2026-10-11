//! st-gateway: Rust gateway for SillyTavern.
//!
//! Zero-modification deployment: the Rust gateway is the single public entry.
//! - Migrated routes are served locally (memory-efficient, no pixel decoding).
//! - All other routes are transparently forwarded to the original Node backend.
//!
//! Configuration: reads `./config.yaml` by default (the start.sh working
//! directory, i.e. the SillyTavern repo root), override with
//! `--config <path>` or `ST_GATEWAY_CONFIG`. Gateway settings live under the
//! top-level `gateway:` key; `port` / `dataRoot` are shared with Node.
//!
//! Auth is replicated from upstream source (verified against node_modules):
//! - cookie-session: `session` cookie = base64(JSON), `.sig` = keygrip HMAC-SHA1/base64url
//! - csrf-sync: plain equality between `x-csrf-token` header and `session.csrfToken`

// The world-book json! literal in card.rs nests deeply.
#![recursion_limit = "512"]

use std::sync::Arc;

use axum::body::Body;
use axum::http::Request;

use st_gateway::{config, forward, routes, AppState};

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "st_gateway=info".into()),
        )
        .init();

    // Config resolution: --config arg > ST_GATEWAY_CONFIG env > ./config.yaml
    let config_path = parse_config_arg().or_else(|| std::env::var("ST_GATEWAY_CONFIG").ok());
    let cfg = Arc::new(match config_path {
        Some(path) => config::ResolvedConfig::load_explicit(&path),
        None => config::ResolvedConfig::load_default(),
    });

    tracing::info!(
        listen = %cfg.listen_addr(),
        node = %cfg.node_addr(),
        data_root = %cfg.data_root.display(),
        "st-gateway starting (migrated routes: {:?})",
        cfg.route_mode_summary()
    );

    let listen: std::net::SocketAddr = cfg
        .listen_addr()
        .parse()
        .unwrap_or_else(|e| panic!("invalid listen_addr {}: {e}", cfg.listen_addr()));

    let state = Arc::new(AppState {
        disk_cache: st_gateway::disk_cache::DiskCache::new(&cfg.data_root),
        cfg,
        node: forward::NodeClient::new(),
    });

    let app = routes::build_router(state);

    let listener = tokio::net::TcpListener::bind(listen)
        .await
        .unwrap_or_else(|e| panic!("failed to bind {listen}: {e}"));
    tracing::info!("listening on {listen}");
    axum::serve(listener, app.into_make_service())
        .await
        .unwrap();
}

/// Extract `--config <path>` (or `--config=<path>`) from argv.
fn parse_config_arg() -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    let mut iter = args.iter().peekable();
    while let Some(arg) = iter.next() {
        if arg == "--config" {
            return iter.next().cloned();
        }
        if let Some(value) = arg.strip_prefix("--config=") {
            return Some(value.to_string());
        }
    }
    None
}

/// Forward a request to the Node backend untouched (used by fallback).
pub async fn forward_node(state: Arc<AppState>, req: Request<Body>) -> axum::response::Response {
    forward::proxy(state, req).await
}
