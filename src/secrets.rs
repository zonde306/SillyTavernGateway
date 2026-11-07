//! Secret reading, mirroring `src/endpoints/secrets.js`.
//!
//! Secrets live in `<user root>/secrets.json` as
//! `{ "<secret_key>": [ { id, value, label, active }, ... ] }`.
//! `readSecret(key, id)` returns the first entry matching `id` when an id is
//! given, else the entry with `active === true`; missing file/key/entry
//! yields an empty string (upstream returns `''`, never null).
//!
//! The file is re-read on every call on purpose: upstream does the same, and
//! the alternative (caching + invalidation) would risk serving a rotated key.
//! Reads are tiny (a few KB) so this is not a hot-path concern.

use std::path::Path;

use serde_json::Value;

/// Upstream `SECRET_KEYS` values used by the chat-completions generate route.
pub mod keys {
    pub const OPENAI: &str = "api_key_openai";
    pub const OPENROUTER: &str = "api_key_openrouter";
    pub const CUSTOM: &str = "api_key_custom";
    pub const PERPLEXITY: &str = "api_key_perplexity";
    pub const GROQ: &str = "api_key_groq";
    pub const FIREWORKS: &str = "api_key_fireworks";
    pub const NANOGPT: &str = "api_key_nanogpt";
    pub const POLLINATIONS: &str = "api_key_pollinations";
    pub const MOONSHOT: &str = "api_key_moonshot";
    pub const ZAI: &str = "api_key_zai";
    pub const COMETAPI: &str = "api_key_cometapi";
    pub const SILICONFLOW: &str = "api_key_siliconflow";
    pub const WORKERS_AI: &str = "api_key_workers_ai";
    pub const CLAUDE: &str = "api_key_claude";
    pub const MAKERSUITE: &str = "api_key_makersuite";
    pub const VERTEXAI: &str = "api_key_vertexai";
    pub const AI21: &str = "api_key_ai21";
    pub const MISTRALAI: &str = "api_key_mistralai";
    pub const COHERE: &str = "api_key_cohere";
    pub const DEEPSEEK: &str = "api_key_deepseek";
    pub const AIMLAPI: &str = "api_key_aimlapi";
    pub const XAI: &str = "api_key_xai";
    pub const CHUTES: &str = "api_key_chutes";
    pub const ELECTRONHUB: &str = "api_key_electronhub";
    pub const MINIMAX: &str = "api_key_minimax";
    pub const AZURE_OPENAI: &str = "api_key_azure_openai";
}

/// Filename mirror of `SECRETS_FILE` in secrets.js.
pub const SECRETS_FILE: &str = "secrets.json";

/// Read the raw secrets document. Returns `None` when the file is absent or
/// unparsable (upstream would throw on invalid JSON — a hard error there, but
/// the gateway degrades to "no secret" and lets the upstream 400 surface).
pub fn read_secrets_file(user_root: &Path) -> Option<Value> {
    let text = std::fs::read_to_string(user_root.join(SECRETS_FILE)).ok()?;
    serde_json::from_str(&text).ok()
}

/// The parsed secrets document (empty object when unavailable).
pub fn read_secrets_doc(user_root: &Path) -> Value {
    read_secrets_file(user_root).unwrap_or_else(|| Value::Object(serde_json::Map::new()))
}

/// Mirror of `SecretManager.readSecret(directories, key, id)`.
pub fn read_secret(doc: &Value, key: &str, id: Option<&str>) -> String {
    let Some(arr) = doc.get(key).and_then(|v| v.as_array()) else {
        return String::new();
    };
    if arr.is_empty() {
        return String::new();
    }
    let found = match id {
        // `secretArray.find(s => id ? s.id === id : s.active)`
        Some(want) => arr
            .iter()
            .find(|s| s.get("id").and_then(|v| v.as_str()) == Some(want)),
        None => arr
            .iter()
            .find(|s| s.get("active").and_then(|v| v.as_bool()) == Some(true)),
    };
    found
        .and_then(|s| s.get("value"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string()
}

/// Convenience: read straight from the user root.
pub fn read_secret_from_user(user_root: &Path, key: &str, id: Option<&str>) -> String {
    read_secret(&read_secrets_doc(user_root), key, id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_active_secret_without_id() {
        let doc = serde_json::json!({
            "api_key_openai": [
                { "id": "a", "value": "old", "active": false },
                { "id": "b", "value": "new", "active": true }
            ]
        });
        assert_eq!(read_secret(&doc, keys::OPENAI, None), "new");
    }

    #[test]
    fn reads_secret_by_id_even_if_inactive() {
        let doc = serde_json::json!({
            "api_key_openai": [
                { "id": "a", "value": "old", "active": false },
                { "id": "b", "value": "new", "active": true }
            ]
        });
        assert_eq!(read_secret(&doc, keys::OPENAI, Some("a")), "old");
    }

    #[test]
    fn missing_key_or_entry_is_empty_string() {
        let doc = serde_json::json!({ "api_key_openai": [] });
        assert_eq!(read_secret(&doc, keys::OPENAI, None), "");
        assert_eq!(read_secret(&doc, "nope", None), "");
        assert_eq!(
            read_secret(&serde_json::json!({}), keys::CUSTOM, Some("x")),
            ""
        );
    }

    #[test]
    fn reads_from_disk_like_upstream() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join(SECRETS_FILE),
            r#"{"api_key_groq":[{"id":"1","value":"gk","label":"","active":true}]}"#,
        )
        .unwrap();
        assert_eq!(read_secret_from_user(dir.path(), keys::GROQ, None), "gk");
    }

    #[test]
    fn absent_file_yields_empty() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(read_secret_from_user(dir.path(), keys::OPENAI, None), "");
    }
}
