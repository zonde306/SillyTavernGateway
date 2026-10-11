//! Library facade so integration tests (tests/) can exercise the gateway's
//! internals against golden vectors generated from the real Node modules.

// The world-book json! literal in card.rs nests deeply.
#![recursion_limit = "512"]

pub mod auth;
pub mod card;
pub mod config;
pub mod forward;
pub mod png;
pub mod routes;

use std::sync::Arc;

/// Shared application state (mirrors main.rs).
pub struct AppState {
    pub cfg: Arc<config::ResolvedConfig>,
    pub node: forward::NodeClient,
}
