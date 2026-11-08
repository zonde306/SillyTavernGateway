//! Library facade so integration tests (tests/) can exercise the gateway's
//! internals against golden vectors generated from the real Node modules.

// The world-book json! literal in card.rs nests deeply.
#![recursion_limit = "512"]

pub mod atomic;
pub mod auth;
pub mod avatar;
pub mod card;
pub mod chat_completions;
pub mod config;
pub mod disk_cache;
pub mod forward;
pub mod heartbeat;
pub mod png;
pub mod prompt;
pub mod routes;
pub mod secrets;
pub mod thumbnails;

use std::sync::Arc;

/// Shared application state (mirrors main.rs).
pub struct AppState {
    pub cfg: Arc<config::ResolvedConfig>,
    pub node: forward::NodeClient,
    /// Shared node-persist-compatible disk cache (data/_cache/characters).
    pub disk_cache: disk_cache::DiskCache,
    /// Thumbnail decode limiter (permits derived from batch_bytes budget).
    pub thumb_limiter: std::sync::Arc<thumbnails::ThumbLimiter>,
    /// Concurrency gate for `/api/backends/chat-completions/generate`
    /// (permits derived from gateway.batch_bytes; image-heavy payloads cost
    /// more permits).
    pub generate_limiter: std::sync::Arc<chat_completions::GenerateLimiter>,
}
