//! Gateway configuration.
//!
//! Loading order (matches the SillyTavern convention of running from the
//! repository root where `start.sh` lives):
//!
//! 1. CLI arg `--config <path>` (see main.rs) — explicit override
//! 2. Env `ST_GATEWAY_CONFIG`
//! 3. Default: `./config.yaml` — resolved against the *current working
//!    directory*, i.e. the start.sh working directory (ST repo root)
//!
//! The gateway config lives inside ST's own `config.yaml` under the top-level
//! `gateway:` key, so operators manage a single file. All other ST settings
//! (`port`, `dataRoot`, `enableUserAccounts`, ...) are read from the same
//! file so the gateway stays consistent with the Node backend without
//! duplicate configuration.

use std::collections::HashMap;
use std::path::PathBuf;

use serde::Deserialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RouteMode {
    Local,
    Forward,
}

/// Full parsed ST config.yaml. Only the keys we consume are declared; the
/// rest is ignored by serde. Field names intentionally mirror ST's
/// config.yaml keys (camelCase).
#[allow(non_snake_case)]
#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct StConfig {
    /// Node listen port (default 8000 like upstream).
    pub port: Option<u16>,
    /// Data root, e.g. `./data`.
    pub dataRoot: Option<String>,
    /// Whether user accounts are enabled (affects default handle resolution).
    pub enableUserAccounts: Option<bool>,
    /// CSRF protection toggle mirror of `disableCsrfProtection`.
    pub disableCsrfProtection: Option<bool>,
    /// Session timeout in seconds (not yet used locally; kept for parity).
    pub sessionTimeout: Option<i64>,
    /// `performance:` section — only lazyLoadCharacters is consumed.
    pub performance: Option<PerformanceSection>,
    /// `thumbnails:` section (format, quality, dimensions, enabled).
    pub thumbnails: Option<ThumbnailsSection>,
    /// Gateway-specific section.
    pub gateway: Option<GatewaySection>,
}

/// Mirror of ST's `thumbnails:` config section.
#[allow(non_snake_case)]
#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct ThumbnailsSection {
    pub enabled: Option<bool>,
    /// Output format: "jpg" (default) or "png".
    pub format: Option<String>,
    /// JPEG quality, 1-100 (default 95).
    pub quality: Option<u8>,
    /// Per-type output dimensions [width, height].
    pub dimensions: Option<ThumbnailDimensions>,
}

/// Per-type thumbnail dimensions, mirroring `thumbnailDimensions` in
/// image-metadata.js.
#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct ThumbnailDimensions {
    pub bg: Option<Vec<u32>>,
    pub avatar: Option<Vec<u32>>,
    pub persona: Option<Vec<u32>>,
}

impl ThumbnailsSection {
    pub fn enabled(&self) -> bool {
        self.enabled.unwrap_or(true)
    }

    pub fn format_is_png(&self) -> bool {
        self.format
            .as_deref()
            .map(|f| f.to_ascii_lowercase().trim() == "png")
            .unwrap_or(false)
    }

    pub fn quality(&self) -> u8 {
        self.quality.unwrap_or(95).clamp(1, 100)
    }

    /// Dimensions for a type ("bg" | "avatar" | "persona"), defaults per
    /// upstream: bg [160,90], avatar/persona [96,144].
    pub fn dims(&self, type_name: &str) -> (u32, u32) {
        let defaults: &[(&str, u32, u32)] =
            &[("bg", 160, 90), ("avatar", 96, 144), ("persona", 96, 144)];
        let d = self.dimensions.as_ref();
        for (name, w, h) in defaults {
            if *name == type_name {
                let configured = match d {
                    Some(dim) => match *name {
                        "bg" => dim.bg.as_deref(),
                        "avatar" => dim.avatar.as_deref(),
                        "persona" => dim.persona.as_deref(),
                        _ => None,
                    },
                    None => None,
                };
                if let Some(v) = configured {
                    if v.len() >= 2 {
                        return (v[0], v[1]);
                    }
                }
                return (*w, *h);
            }
        }
        (96, 144)
    }
}

/// Mirror of ST's `performance:` config section.
#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct PerformanceSection {
    /// Shallow character list (upstream `performance.lazyLoadCharacters`).
    #[allow(non_snake_case)]
    pub lazyLoadCharacters: Option<bool>,
}

/// Gateway-specific configuration under the `gateway:` key.
#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct GatewaySection {
    /// Public listen address of the gateway. Defaults to
    /// `auto` = bind 0.0.0.0:<port> reusing the ST port.
    pub listen_addr: Option<String>,
    /// Address of the original Node SillyTavern. Defaults to
    /// `127.0.0.1:<port + 1>`.
    pub node_addr: Option<String>,
    /// Per-route mode table. Keys are path prefixes without `/api`, e.g.
    /// `characters/all`. Unlisted routes are forwarded.
    pub routes: Option<HashMap<String, RouteMode>>,
    /// Shadow mode: when true, local implementations run but forward-mode
    /// responses are still produced for comparison (testing aid).
    pub shadow_compare: Option<bool>,
    /// Batch limit for `/api/characters/all` by total file size in bytes:
    /// a batch accumulates entries until adding the next would exceed this.
    /// A single file larger than the limit is allowed to run alone. Batches
    /// that fail are split in half and retried; single-file failures are
    /// reported as zeroed stubs without retry. Default: 256 MiB.
    pub batch_bytes: Option<u64>,
}

pub struct ResolvedConfig {
    pub st: StConfig,
    /// Final gateway section (st.gateway or defaults).
    pub gateway: GatewaySection,
    /// Resolved absolute-ish data root.
    pub data_root: PathBuf,
    /// Node port derived from st.port or 8000.
    pub node_port: u16,
}

fn default_node_port(st_port: Option<u16>) -> u16 {
    st_port.unwrap_or(8000)
}

impl ResolvedConfig {
    /// Load with an explicit path (CLI arg or env).
    pub fn load_explicit(path: &str) -> Self {
        Self::from_text(
            &std::fs::read_to_string(path)
                .unwrap_or_else(|e| panic!("cannot read config file {path}: {e}")),
            std::path::Path::new(path)
                .parent()
                .map(|p| p.to_path_buf())
                .unwrap_or_default(),
        )
    }

    /// Load from the default `./config.yaml` (cwd = start.sh directory).
    pub fn load_default() -> Self {
        let path = std::path::Path::new("config.yaml");
        match std::fs::read_to_string(path) {
            Ok(text) => Self::from_text(&text, PathBuf::from(".")),
            Err(e) => panic!(
                "no config file found at ./config.yaml (cwd): {e}. Run from the SillyTavern root (start.sh directory) or pass --config <path>."
            ),
        }
    }

    /// Parse config text directly (exposed for golden tests).
    pub fn from_text(text: &str, base: PathBuf) -> Self {
        let st: StConfig = serde_yaml::from_str(text)
            .unwrap_or_else(|e| panic!("failed to parse config.yaml: {e}"));

        let node_port = default_node_port(st.port);
        let data_root = {
            let raw = st.dataRoot.clone().unwrap_or_else(|| "./data".to_string());
            resolve(&base, &raw)
        };

        let gateway = st.gateway.clone().unwrap_or_default();
        Self {
            st,
            gateway,
            data_root,
            node_port,
        }
    }

    /// Effective listen address: explicit gateway.listen_addr or auto-derive
    /// from the ST port.
    pub fn listen_addr(&self) -> String {
        self.gateway
            .listen_addr
            .clone()
            .unwrap_or_else(|| format!("0.0.0.0:{}", self.node_port))
    }

    /// Effective Node address: explicit gateway.node_addr or 127.0.0.1:port+1.
    pub fn node_addr(&self) -> String {
        self.gateway
            .node_addr
            .clone()
            .unwrap_or_else(|| format!("127.0.0.1:{}", self.node_port + 1))
    }

    /// Route mode for an `/api/...` path. Unlisted => Forward.
    pub fn route_mode(&self, api_path: &str) -> RouteMode {
        let map = self.gateway.routes.as_ref();
        let key = api_path
            .strip_prefix("/api/")
            .unwrap_or_else(|| api_path.trim_start_matches('/'));
        if let Some(map) = map {
            if let Some(mode) = map.get(key) {
                return *mode;
            }
            let mut prefix = key;
            while let Some(pos) = prefix.rfind('/') {
                prefix = &prefix[..pos];
                if let Some(mode) = map.get(prefix) {
                    return *mode;
                }
            }
        }
        RouteMode::Forward
    }

    pub fn route_mode_summary(&self) -> Vec<String> {
        self.gateway
            .routes
            .as_ref()
            .map(|m| {
                let mut keys: Vec<_> = m
                    .iter()
                    .filter(|(_, mode)| **mode == RouteMode::Local)
                    .map(|(k, _)| k.clone())
                    .collect();
                keys.sort();
                keys
            })
            .unwrap_or_default()
    }

    pub fn user_dir(&self, handle: &str) -> PathBuf {
        self.data_root.join(handle)
    }

    /// Thumbnail output format is PNG (else JPEG). Mirrors upstream
    /// `thumbnails.format` config read.
    pub fn thumb_format_is_png(&self) -> bool {
        self.st
            .thumbnails
            .as_ref()
            .map(|t| t.format_is_png())
            .unwrap_or(false)
    }

    /// JPEG quality 1-100 (upstream default 95).
    pub fn thumb_quality(&self) -> u8 {
        self.st
            .thumbnails
            .as_ref()
            .map(|t| t.quality())
            .unwrap_or(95)
    }

    /// Whether thumbnails are enabled (upstream `thumbnails.enabled`).
    pub fn thumb_enabled(&self) -> bool {
        self.st
            .thumbnails
            .as_ref()
            .map(|t| t.enabled())
            .unwrap_or(true)
    }

    /// Dimensions for a thumbnail type ("bg"|"avatar"|"persona").
    pub fn thumb_dims(&self, type_name: &str) -> (u32, u32) {
        self.st
            .thumbnails
            .as_ref()
            .map(|t| t.dims(type_name))
            .unwrap_or_else(|| match type_name {
                "bg" => (160, 90),
                _ => (96, 144),
            })
    }

    /// Per-request resolution value returned in thumbnail JSON (w*h).
    pub fn thumb_resolution(&self, type_name: &str) -> u32 {
        let (w, h) = self.thumb_dims(type_name);
        w.saturating_mul(h)
    }

    /// Thumbnail decode concurrency: derived from batch_bytes (the designed
    /// memory budget). Each in-flight decode may hold the full RGBA bitmap of
    /// the source image; worst case is bounded by bitmap-permit sizing:
    /// batch_bytes / 8 MiB (approx bitmap size of a large avatar), min 1,
    /// max 16. This keeps total decode memory under the same budget that
    /// governs /all batching.
    pub fn thumb_concurrency(&self) -> usize {
        let budget = self.gateway.batch_bytes.unwrap_or(256 * 1024 * 1024);
        let permits = (budget / (8 * 1024 * 1024)).max(1) as usize;
        permits.min(16)
    }

    pub fn cookie_secret_path(&self) -> PathBuf {
        self.data_root.join("cookie-secret.txt")
    }
}

fn resolve(base: &std::path::Path, p: &str) -> PathBuf {
    let path = std::path::Path::new(p);
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        base.join(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_full_config() {
        let text = r#"
dataRoot: ./data
port: 8000
enableUserAccounts: false
disableCsrfProtection: false
sessionTimeout: 86400
gateway:
  listen_addr: "127.0.0.1:9000"
  node_addr: "127.0.0.1:5001"
  routes:
    "characters/all": local
    "characters": local
  shadow_compare: false
"#;
        let cfg = ResolvedConfig::from_text(text, PathBuf::from("."));
        assert_eq!(cfg.listen_addr(), "127.0.0.1:9000");
        assert_eq!(cfg.node_addr(), "127.0.0.1:5001");
        assert_eq!(cfg.route_mode("/api/characters/all"), RouteMode::Local);
        assert_eq!(cfg.route_mode("/api/characters/get"), RouteMode::Local);
        assert_eq!(cfg.route_mode("/api/chats/get"), RouteMode::Forward);
        assert_eq!(cfg.node_port, 8000);
        assert_eq!(cfg.data_root, PathBuf::from("./data"));
    }

    #[test]
    fn defaults_derive_from_st_port() {
        let text = r#"
port: 9005
dataRoot: /var/st/data
"#;
        let cfg = ResolvedConfig::from_text(text, PathBuf::from("."));
        // No gateway section: listen auto = 0.0.0.0:9005, node = 127.0.0.1:9006
        assert_eq!(cfg.listen_addr(), "0.0.0.0:9005");
        assert_eq!(cfg.node_addr(), "127.0.0.1:9006");
        assert_eq!(cfg.data_root, PathBuf::from("/var/st/data"));
        assert_eq!(cfg.route_mode("/api/anything"), RouteMode::Forward);
    }

    #[test]
    fn empty_config_uses_defaults() {
        let cfg = ResolvedConfig::from_text("", PathBuf::from("."));
        assert_eq!(cfg.node_port, 8000);
        assert_eq!(cfg.listen_addr(), "0.0.0.0:8000");
        assert_eq!(cfg.node_addr(), "127.0.0.1:8001");
        assert_eq!(cfg.data_root, PathBuf::from("./data"));
    }

    #[test]
    fn cookie_secret_and_user_dirs() {
        let cfg = ResolvedConfig::from_text("dataRoot: ./data", PathBuf::from("."));
        assert_eq!(
            cfg.cookie_secret_path(),
            PathBuf::from("./data/cookie-secret.txt")
        );
        assert_eq!(cfg.user_dir("alice"), PathBuf::from("./data/alice"));
    }
}
