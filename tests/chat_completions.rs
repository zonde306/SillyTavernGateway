use serde_json::json;
use st_gateway::chat_completions::*;
use st_gateway::config::ResolvedConfig;

fn empty_cfg() -> st_gateway::config::StConfig {
    ResolvedConfig::from_text(
        "dataRoot: ./data\npromptPlaceholder: \"[Start a new chat]\"\n",
        std::path::PathBuf::from("."),
    )
    .st
}

/// Scratch directory holding a `secrets.json` (no external deps).
struct SecretsDir(std::path::PathBuf);

impl SecretsDir {
    fn new(secrets: &str) -> Self {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "stgw-completion-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&path).unwrap();
        std::fs::write(path.join("secrets.json"), secrets).unwrap();
        Self(path)
    }
}

impl Drop for SecretsDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

const SECRETS: &str =
    r#"{"api_key_openai":[{"id":"a","value":"sk-test","label":"","active":true}]}"#;
const NO_SECRET: &str = "";

#[test]
fn openai_request_shape_matches_upstream() {
    let dir = SecretsDir::new(SECRETS);
    let cfg = empty_cfg();
    let mut b = json!({
        "chat_completion_source": "openai",
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.7,
        "max_tokens": 100,
        "stream": false,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "top_p": 1,
        "logprobs": 2,
        "n": 1
    });
    let prepared = prepare(&mut b, &dir.0, &cfg, std::path::Path::new(NO_SECRET)).unwrap();
    assert_eq!(prepared.url, "https://api.openai.com/v1/chat/completions");
    assert!(prepared
        .headers
        .iter()
        .any(|(k, v)| k == "Authorization" && v == "Bearer sk-test"));
    let keys: Vec<&String> = prepared.body.as_object().unwrap().keys().collect();
    assert_eq!(
        keys,
        vec![
            "messages",
            "model",
            "temperature",
            "max_tokens",
            "stream",
            "presence_penalty",
            "frequency_penalty",
            "top_p",
            "n",
            "logprobs",
            "top_logprobs"
        ],
        "undefined fields must be omitted and order must match upstream"
    );
}

#[test]
fn text_completion_uses_completions_endpoint_and_prompt() {
    let dir = SecretsDir::new(SECRETS);
    let cfg = empty_cfg();
    let mut b = json!({
        "chat_completion_source": "openai",
        "model": "gpt-3.5-turbo-instruct",
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"}
        ]
    });
    let prepared = prepare(&mut b, &dir.0, &cfg, std::path::Path::new(NO_SECRET)).unwrap();
    assert_eq!(prepared.url, "https://api.openai.com/v1/completions");
    assert_eq!(prepared.body["prompt"], "System: sys\nuser: hi\nassistant:");
    assert!(prepared.body.get("messages").is_none());
}

#[test]
fn missing_api_key_is_bad_request() {
    let dir = SecretsDir::new("{}");
    let cfg = empty_cfg();
    let mut b = json!({
        "chat_completion_source": "openai",
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}]
    });
    assert!(matches!(
        prepare(&mut b, &dir.0, &cfg, std::path::Path::new(NO_SECRET)),
        Err(PrepareError::BadRequest(_))
    ));
}

#[test]
fn custom_source_excludes_and_merges_yaml() {
    let dir = SecretsDir::new("{}");
    let cfg = empty_cfg();
    let mut b = json!({
        "chat_completion_source": "custom",
        "custom_url": "http://localhost:5000/v1",
        "model": "koboldcpp/x",
        "messages": [{"role": "user", "content": "hi"}],
        "custom_include_body": "top_k: 40\nmax_tokens: 42",
        "custom_exclude_body": "[\"n\", \"seed\"]",
        "custom_include_headers": "X-Custom: yes",
        "n": 1,
        "stop": ["\n"]
    });
    let prepared = prepare(&mut b, &dir.0, &cfg, std::path::Path::new(NO_SECRET)).unwrap();
    assert_eq!(prepared.url, "http://localhost:5000/v1/chat/completions");
    assert_eq!(prepared.body["top_k"], 40);
    assert_eq!(prepared.body["max_tokens"], 42);
    assert!(prepared.body.get("n").is_none());
    assert!(prepared.body.get("seed").is_none());
    assert_eq!(prepared.body["stop"], json!(["\n"]));
    assert!(prepared
        .headers
        .iter()
        .any(|(k, v)| k == "X-Custom" && v == "yes"));
}

#[test]
fn workers_ai_requires_account_id_and_encodes_it() {
    let dir = SecretsDir::new(
        r#"{"api_key_workers_ai":[{"id":"a","value":"cf","label":"","active":true}]}"#,
    );
    let cfg = empty_cfg();
    let mut missing = json!({
        "chat_completion_source": "workers_ai",
        "model": "@cf/x",
        "messages": [{"role": "user", "content": "hi"}]
    });
    assert!(matches!(
        prepare(&mut missing, &dir.0, &cfg, std::path::Path::new(NO_SECRET)),
        Err(PrepareError::BadRequest(_))
    ));

    let mut ok = json!({
        "chat_completion_source": "workers_ai",
        "model": "@cf/x",
        "workers_ai_account_id": "acct 1",
        "messages": [{"role": "user", "content": "hi"}]
    });
    let prepared = prepare(&mut ok, &dir.0, &cfg, std::path::Path::new(NO_SECRET)).unwrap();
    assert_eq!(
        prepared.url,
        "https://api.cloudflare.com/client/v4/accounts/acct%201/ai/v1/chat/completions"
    );
}

#[test]
fn cometapi_is_disabled_like_upstream() {
    let dir = SecretsDir::new("{}");
    let cfg = empty_cfg();
    let mut b = json!({
        "chat_completion_source": "cometapi",
        "messages": [{"role": "user", "content": "hi"}]
    });
    match prepare(&mut b, &dir.0, &cfg, std::path::Path::new(NO_SECRET)) {
        Err(PrepareError::Error(msg)) => {
            assert_eq!(msg, "This provider is temporarily disabled.")
        }
        other => panic!("expected Error, got {other:?}"),
    }
}

#[test]
fn flatten_schema_inlines_defs_and_filters_google_keys() {
    let schema = json!({
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "additionalProperties": false,
        "properties": {"a": {"$ref": "#/$defs/A"}},
        "$defs": {
            "A": {"type": "string", "default": "x", "propertyNames": {"type": "string"}}
        }
    });
    let plain = flatten_schema(&schema, "openai");
    assert!(plain.get("$schema").is_none());
    assert!(plain.get("$defs").is_none());
    assert_eq!(plain["properties"]["a"]["type"], "string");
    assert_eq!(plain["properties"]["a"]["default"], "x");
    assert_eq!(plain["additionalProperties"], false);

    let google = flatten_schema(&schema, "vertexai");
    assert!(google["properties"]["a"].get("default").is_none());
    assert!(google["properties"]["a"].get("propertyNames").is_none());
    assert!(google.get("additionalProperties").is_none());
}

#[test]
fn self_referential_schema_becomes_empty_object() {
    let schema = json!({
        "type": "object",
        "properties": {"node": {"$ref": "#/$defs/Node"}},
        "$defs": {
            "Node": {"type": "object", "properties": {"child": {"$ref": "#/$defs/Node"}}}
        }
    });
    let out = flatten_schema(&schema, "openai");
    assert_eq!(out["properties"]["node"]["properties"]["child"], json!({}));
}

#[test]
fn hmac_affinity_matches_node() {
    // node -e "require('crypto').createHmac('sha256','k').update('chat').digest('hex').slice(0,16)"
    assert_eq!(hmac_sha256_hex16("k", b"chat"), "81d33bd16b7a20cd");
}

#[test]
fn normalize_url_adds_trailing_slash_like_js_url() {
    assert_eq!(
        normalize_url("https://api.openai.com"),
        "https://api.openai.com/"
    );
    assert_eq!(
        normalize_url("https://api.openai.com/v1"),
        "https://api.openai.com/v1"
    );
}

#[test]
fn protocol_detection_covers_every_wire_format() {
    assert_eq!(
        Protocol::from_content_type(Some("text/event-stream; charset=utf-8")),
        Protocol::Sse
    );
    assert_eq!(
        Protocol::from_content_type(Some("application/x-ndjson")),
        Protocol::Ndjson
    );
    assert_eq!(
        Protocol::from_content_type(Some("application/json")),
        Protocol::Json
    );
    assert_eq!(
        Protocol::from_content_type(Some("text/html")),
        Protocol::Unknown
    );
    assert_eq!(Protocol::from_content_type(None), Protocol::Unknown);
    // Streams without a labelled media type are treated as line-delimited.
    assert_eq!(
        Protocol::for_request(Some("text/plain"), true),
        Protocol::Ndjson
    );
    assert_eq!(
        Protocol::for_request(Some("text/plain"), false),
        Protocol::Unknown
    );
}

#[test]
fn limiter_charges_by_size() {
    let limiter = GenerateLimiter::new(256 * 1024 * 1024);
    assert_eq!(limiter.permits_for(0), 1);
    assert_eq!(limiter.permits_for(1), 1);
    assert_eq!(limiter.permits_for(16 * 1024 * 1024), 2);
    assert_eq!(limiter.permits_for(u64::MAX), limiter.total as u32);
}

#[test]
fn inline_source_set_matches_port() {
    for s in [
        "openai",
        "openrouter",
        "custom",
        "perplexity",
        "groq",
        "fireworks",
        "nanogpt",
        "pollinations",
        "moonshot",
        "cometapi",
        "zai",
        "siliconflow",
        "workers_ai",
    ] {
        assert!(is_inline_source(s), "{s} should be inline");
    }
    // Every provider SillyTavern can select is built in-process now; nothing
    // in `CHAT_COMPLETION_SOURCES` is forwarded to Node any more.
    for s in [
        "claude",
        "ai21",
        "makersuite",
        "vertexai",
        "mistralai",
        "cohere",
        "deepseek",
        "aimlapi",
        "xai",
        "chutes",
        "minimax",
        "electronhub",
        "azure_openai",
    ] {
        assert!(is_inline_source(s), "{s} should be handled locally");
    }
    for s in ["", "not-a-provider"] {
        assert!(!is_inline_source(s), "{s} is unknown and still forwards");
    }
}
