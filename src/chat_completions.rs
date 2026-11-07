//! `/api/backends/chat-completions/generate` — the OpenAI-compatible half of
//! the upstream route, mirrored from `src/endpoints/backends/chat-completions.js`.
//!
//! Upstream handles two families of providers in that one handler:
//!
//! * a `switch` dispatching to `send<Provider>Request` for Claude, AI21,
//!   MakerSuite, Vertex AI, Mistral, Cohere, DeepSeek, AI21 (aimlapi), xAI,
//!   Chutes, MiniMax, Electron Hub and Azure OpenAI — all of which are still
//!   forwarded to Node by the gateway (not ported);
//!
//! * an inline `if/else` chain building a plain OpenAI-style JSON body for
//!   openai, openrouter, custom, perplexity, groq, fireworks, nanogpt,
//!   pollinations, moonshot, cometapi (disabled upstream), zai, siliconflow
//!   and workers_ai. Everything below mirrors that chain statement by
//!   statement.
//!
//! The request body is rebuilt field by field in upstream property order so
//! the JSON sent to the provider is byte-identical to what Node produced
//! (`undefined` values are simply omitted, which is what serde's `None` does).

use serde_json::{json, Map, Value};

use crate::prompt;
use crate::secrets::{self, keys};

// ---------------------------------------------------------------------------
// Constants (src/constants.js, src/endpoints/backends/chat-completions.js)
// ---------------------------------------------------------------------------

pub const API_OPENAI: &str = "https://api.openai.com/v1";
pub const API_OPENROUTER: &str = "https://openrouter.ai/api/v1";
pub const API_GROQ: &str = "https://api.groq.com/openai/v1";
pub const API_NANOGPT: &str = "https://nano-gpt.com/api/v1";
pub const API_POLLINATIONS: &str = "https://gen.pollinations.ai/v1";
pub const API_POLLINATIONS_ANON: &str = "https://text.pollinations.ai/v1";
pub const API_MOONSHOT: &str = "https://api.moonshot.ai/v1";
pub const API_FIREWORKS: &str = "https://api.fireworks.ai/inference/v1";
pub const API_ZAI_COMMON: &str = "https://api.z.ai/api/paas/v4";
pub const API_ZAI_CODING: &str = "https://api.z.ai/api/coding/paas/v4";
pub const API_SILICONFLOW: &str = "https://api.siliconflow.com/v1";
pub const API_SILICONFLOW_CN: &str = "https://api.siliconflow.cn/v1";
pub const API_WORKERS_AI: &str = "https://api.cloudflare.com/client/v4/accounts";
pub const API_PERPLEXITY: &str = "https://api.perplexity.ai";

/// OpenRouter requires the Referer / X-Title pair.
pub const OPENROUTER_HEADERS: &[(&str, &str)] = &[
    ("HTTP-Referer", "https://sillytavern.app"),
    ("X-Title", "SillyTavern"),
];

/// `OPENAI_VERBOSITY_MODELS` — `/^(?:gpt-5|gpt-6-)/`.
fn is_verbosity_model(model: &str) -> bool {
    has_prefix_match(model, &["gpt-5", "gpt-6-"])
}

/// `OPENAI_REASONING_EFFORT_MODELS` (src/constants.js) — checked with
/// `Array.includes`, i.e. exact equality, not a prefix match.
const OPENAI_REASONING_EFFORT_MODELS: &[&str] = &[
    "o1",
    "o3",
    "o4-mini",
    "o3-mini",
    "o1-mini",
    "o1-preview",
    "o1-pro",
    "o3-pro",
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-5.1",
    "gpt-5.1-chat-latest",
    "gpt-5.1-codex",
    "gpt-5.1-codex-mini",
    "gpt-5.1-codex-max",
    "gpt-5.2",
    "gpt-5.2-pro",
    "gpt-5.2-codex",
    "gpt-5.3-chat-latest",
    "gpt-5.4",
    "gpt-6",
    "gpt-6-mini",
    "gpt-6-nano",
];

/// `OPENAI_REASONING_EFFORT_MAP` — only `min` is remapped for OpenAI.
fn map_openai_effort(effort: &str) -> &str {
    match effort {
        "min" => "minimal",
        other => other,
    }
}

/// `OPENAI_FIXED_REASONING_EFFORT`.
fn fixed_openai_effort(model: &str) -> Option<&'static str> {
    match model {
        "gpt-5.3-chat-latest" => Some("medium"),
        _ => None,
    }
}

/// `NANOGPT_REASONING_EFFORT_MAP`.
fn map_nanogpt_effort(effort: &str) -> &str {
    match effort {
        "min" => "none",
        "low" => "minimal",
        "medium" => "low",
        "high" => "medium",
        "max" => "high",
        other => other,
    }
}

/// `GEMINI_SAFETY` from src/constants.js.
fn gemini_safety() -> Value {
    json!([
        { "category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE" },
        { "category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE" },
        { "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE" },
        { "category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE" },
        { "category": "HARM_CATEGORY_CIVIC_INTEGRITY", "threshold": "BLOCK_NONE" }
    ])
}

/// TEXT_COMPLETION_MODELS membership (exact match, like `Array.includes`).
const TEXT_COMPLETION_MODELS: &[&str] = &[
    "gpt-3.5-turbo-instruct",
    "gpt-4-bing",
    "gpt-4-turbo-bing",
    "gpt-4.5-preview",
    "gpt-5-instruct",
];

/// Cheap mirror of a JS regex `^prefix`. The upstream regexes are anchored
/// and only ever test a prefix, so `starts_with` is exact.
fn has_prefix_match(value: &str, prefixes: &[&str]) -> bool {
    prefixes.iter().any(|p| value.starts_with(p))
}

/// Mirror of `new URL(x).toString()` for the URLs this route touches:
/// lowercase scheme/host, drop default ports, add "/" to a bare origin, keep
/// the path as-is.
pub fn normalize_url(raw: &str) -> String {
    match url::Url::parse(raw) {
        Ok(u) => {
            let mut out = u.to_string();
            if !out.ends_with('/') && u.path() == "/" {
                out.push('/');
            }
            out
        }
        Err(_) => raw.to_string(),
    }
}

// ---------------------------------------------------------------------------
// Request-side helpers
// ---------------------------------------------------------------------------

/// Providers handled by the inline chain below. Everything else is forwarded
/// to Node (the `send*Request` family).
pub fn is_inline_source(source: &str) -> bool {
    matches!(
        source,
        "openai"
            | "openrouter"
            | "custom"
            | "perplexity"
            | "groq"
            | "fireworks"
            | "nanogpt"
            | "pollinations"
            | "moonshot"
            | "cometapi"
            | "zai"
            | "siliconflow"
            | "workers_ai"
    )
}

fn get_str<'a>(body: &'a Value, key: &str) -> Option<&'a str> {
    body.get(key).and_then(|v| v.as_str())
}

fn get_bool(body: &Value, key: &str) -> bool {
    body.get(key).and_then(|v| v.as_bool()).unwrap_or(false)
}

/// `mergeObjectWithYaml(obj, yamlString)` — parse the YAML text and
/// `Object.assign` the result. Arrays contribute each object item; parse
/// failures are swallowed, exactly like upstream.
pub fn merge_object_with_yaml(obj: &mut Map<String, Value>, yaml_text: Option<&str>) {
    let Some(text) = yaml_text.filter(|t| !t.is_empty()) else {
        return;
    };
    let Ok(parsed) = serde_yaml::from_str::<Value>(text) else {
        return;
    };
    merge_assign(obj, &parsed);
}

fn merge_assign(obj: &mut Map<String, Value>, parsed: &Value) {
    match parsed {
        Value::Array(items) => {
            for item in items {
                if let Value::Object(map) = item {
                    for (k, v) in map {
                        obj.insert(k.clone(), v.clone());
                    }
                }
            }
        }
        Value::Object(map) => {
            for (k, v) in map {
                obj.insert(k.clone(), v.clone());
            }
        }
        _ => {}
    }
}

/// `excludeKeysByYaml(obj, yamlString)`.
pub fn exclude_keys_by_yaml(obj: &mut Map<String, Value>, yaml_text: Option<&str>) {
    let Some(text) = yaml_text.filter(|t| !t.is_empty()) else {
        return;
    };
    let Ok(parsed) = serde_yaml::from_str::<Value>(text) else {
        return;
    };
    match parsed {
        Value::Array(items) => {
            for item in items {
                if let Value::String(key) = item {
                    obj.remove(&key);
                }
            }
        }
        Value::Object(map) => {
            for key in map.keys() {
                obj.remove(key);
            }
        }
        Value::String(key) => {
            obj.remove(&key);
        }
        _ => {}
    }
}

/// `flattenSchema(schema, api)` from src/util.js.
pub fn flatten_schema(schema: &Value, source: &str) -> Value {
    if !schema.is_object() {
        return schema.clone();
    }
    let google = matches!(source, "vertexai" | "makersuite");
    let definitions = schema.get("$defs").cloned().unwrap_or_else(|| json!({}));
    let mut copy = schema.clone();
    if let Some(obj) = copy.as_object_mut() {
        obj.remove("$defs");
    }

    fn resolve(value: &Value, definitions: &Value, google: bool, parents: &[String]) -> Value {
        match value {
            Value::Array(items) => Value::Array(
                items
                    .iter()
                    .map(|i| resolve(i, definitions, google, parents))
                    .collect(),
            ),
            Value::Object(map) => {
                if let Some(reference) = map.get("$ref").and_then(|v| v.as_str()) {
                    if let Some(name) = reference.strip_prefix("#/$defs/") {
                        let name = name.rsplit('/').next().unwrap_or(name).to_string();
                        if parents.iter().any(|p| p == &name) {
                            return json!({});
                        }
                        if let Some(def) = definitions.get(&name) {
                            let mut next: Vec<String> = parents.to_vec();
                            next.push(name);
                            return resolve(def, definitions, google, &next);
                        }
                    }
                }
                let mut out = Map::new();
                for (key, val) in map {
                    if google
                        && matches!(
                            key.as_str(),
                            "default"
                                | "additionalProperties"
                                | "exclusiveMinimum"
                                | "propertyNames"
                        )
                    {
                        continue;
                    }
                    out.insert(key.clone(), resolve(val, definitions, google, parents));
                }
                Value::Object(out)
            }
            other => other.clone(),
        }
    }

    let mut flattened = resolve(&copy, &definitions, google, &[]);
    if let Some(obj) = flattened.as_object_mut() {
        obj.remove("$schema");
    }
    flattened
}

// ---------------------------------------------------------------------------
// Prepared request
// ---------------------------------------------------------------------------

/// Everything needed to talk to the provider once the body is built.
#[derive(Debug, Clone)]
pub struct PreparedRequest {
    pub url: String,
    /// Headers to send upstream (`Content-Type`/`Authorization` included).
    pub headers: Vec<(String, String)>,
    pub body: Value,
    pub stream: bool,
}

/// Why a request cannot proceed. `BadRequest` maps to `400 {error:true}`,
/// `Error` maps to the 502 `{error:{message}}` shape.
#[derive(Debug)]
pub enum PrepareError {
    BadRequest(&'static str),
    Error(String),
}

/// Build the upstream request. `user_root` is the authenticated user's
/// directory (where `secrets.json` lives); `cfg` is the parsed ST config for
/// `getConfigValue`-style lookups.
pub fn prepare(
    body: &mut Value,
    user_root: &std::path::Path,
    cfg: &crate::config::StConfig,
    cookie_secret_path: &std::path::Path,
) -> Result<PreparedRequest, PrepareError> {
    if !body.is_object() {
        return Err(PrepareError::BadRequest("no request body"));
    }

    // --- global pre-processing ------------------------------------------------
    let post_processing = get_str(body, "custom_prompt_post_processing")
        .map(str::to_string)
        .filter(|s| !s.is_empty());
    if get_bool_ctx(body, "messages_is_array") && post_processing.is_some() {
        // handled by the caller (see `post_process_if_requested`)
    }

    let schema_source = get_str(body, "chat_completion_source")
        .unwrap_or("")
        .to_string();
    if let Some(schema) = body.get_mut("json_schema") {
        let has_value = schema.get("value").is_some_and(|v| !v.is_null());
        if has_value {
            if let Some(value) = schema.get_mut("value") {
                *value = flatten_schema(value, &schema_source);
            }
        }
    }

    let source = get_str(body, "chat_completion_source")
        .unwrap_or("")
        .to_string();
    let model = get_str(body, "model").unwrap_or("").to_string();
    let is_text_completion = (!model.is_empty()
        && TEXT_COMPLETION_MODELS.contains(&model.as_str()))
        || body.get("messages").is_some_and(|m| m.is_string());

    let secrets_doc = secrets::read_secrets_doc(user_root);
    let secret_id = get_str(body, "secret_id");
    let read_secret = |key: &str| secrets::read_secret(&secrets_doc, key, secret_id);

    let mut headers: Vec<(String, String)> = Vec::new();
    let api_url: String;
    let api_key: String;
    let mut body_params: Map<String, Value> = Map::new();

    match source.as_str() {
        "openai" => {
            let reverse_proxy = get_str(body, "reverse_proxy").filter(|s| !s.is_empty());
            api_url = normalize_url(reverse_proxy.unwrap_or(API_OPENAI));
            api_key = if reverse_proxy.is_some() {
                get_str(body, "proxy_password").unwrap_or("").to_string()
            } else {
                read_secret(keys::OPENAI)
            };
            // JSON.stringify drops undefined values, so an absent logprobs
            // never reaches the provider.
            let logprobs = body.get("logprobs").cloned().filter(|v| !v.is_null());
            if let Some(logprobs) = &logprobs {
                // Upstream sends `{ logprobs: <n>, top_logprobs: undefined }`,
                // and flips `logprobs` to `true` afterwards so the serialized
                // key order is `logprobs, top_logprobs`.
                body_params.insert("logprobs".into(), logprobs.clone());
                body_params.insert("top_logprobs".into(), Value::Null);
                if !is_text_completion && logprobs.as_f64().unwrap_or(0.0) > 0.0 {
                    body_params.insert("top_logprobs".into(), logprobs.clone());
                    body_params.insert("logprobs".into(), Value::Bool(true));
                }
            }
            if config_bool(cfg, "openai.randomizeUserId", false) {
                body_params.insert("user".into(), Value::String(uuid_v4()));
            }
            if let Some(messages) = body.get_mut("messages") {
                prompt::embed_open_router_media(messages, true, false);
            }
        }
        "openrouter" => {
            api_url = API_OPENROUTER.to_string();
            api_key = read_secret(keys::OPENROUTER);
            for (k, v) in OPENROUTER_HEADERS {
                headers.push((k.to_string(), v.to_string()));
            }
            let include_reasoning = get_bool(body, "include_reasoning");
            let transforms = match get_str(body, "middleout") {
                Some("on") => json!(["middle-out"]),
                Some("off") => json!([]),
                _ => Value::Null,
            };
            let mut plugins: Vec<Value> = Vec::new();
            if get_bool(body, "enable_web_search") {
                plugins.push(json!({ "id": "web" }));
            }
            body_params.insert("transforms".into(), transforms);
            body_params.insert("plugins".into(), Value::Array(plugins));
            body_params.insert("reasoning".into(), json!({ "exclude": !include_reasoning }));

            let logprobs = body.get("logprobs").and_then(|v| v.as_f64()).unwrap_or(0.0);
            if logprobs > 0.0 {
                body_params.insert("top_logprobs".into(), body["logprobs"].clone());
                body_params.insert("logprobs".into(), Value::Bool(true));
            }
            if let Some(v) = body.get("min_p").filter(|v| !v.is_null()) {
                body_params.insert("min_p".into(), v.clone());
            }
            if let Some(v) = body.get("top_a").filter(|v| !v.is_null()) {
                body_params.insert("top_a".into(), v.clone());
            }
            if let Some(v) = body.get("repetition_penalty").filter(|v| !v.is_null()) {
                body_params.insert("repetition_penalty".into(), v.clone());
            }
            if let Some(provider) = body.get("provider").and_then(|v| v.as_array()) {
                if !provider.is_empty() {
                    let allow_fallbacks = body
                        .get("allow_fallbacks")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(true);
                    body_params.insert(
                        "provider".into(),
                        json!({ "allow_fallbacks": allow_fallbacks, "order": provider }),
                    );
                }
            }
            if let Some(quantizations) = body.get("quantizations").and_then(|v| v.as_array()) {
                if !quantizations.is_empty() {
                    let entry = body_params
                        .entry("provider".to_string())
                        .or_insert_with(|| json!({}));
                    if let Some(obj) = entry.as_object_mut() {
                        obj.insert("quantizations".into(), Value::Array(quantizations.clone()));
                    }
                }
            }
            if get_bool(body, "use_fallback") {
                body_params.insert("route".into(), json!("fallback"));
            }
            let effort = get_str(body, "reasoning_effort").filter(|s| !s.is_empty());
            if let Some(effort) = effort {
                if let Some(reasoning) = body_params
                    .get_mut("reasoning")
                    .and_then(|r| r.as_object_mut())
                {
                    reasoning.insert("effort".into(), Value::String(effort.to_string()));
                }
            }
            let verbosity = get_str(body, "verbosity").filter(|s| !s.is_empty());
            if let Some(verbosity) = verbosity {
                body_params.insert("verbosity".into(), Value::String(verbosity.to_string()));
            }
            if let Some(schema) = json_schema_of(body) {
                body_params.insert(
                    "response_format".into(),
                    json!({
                        "type": "json_schema",
                        "json_schema": {
                            "name": schema.get("name").cloned().unwrap_or(Value::Null),
                            "strict": schema.get("strict").and_then(|v| v.as_bool()).unwrap_or(true),
                            "schema": schema.get("value").cloned().unwrap_or(Value::Null),
                        }
                    }),
                );
            }

            let is_claude = has_prefix_match(&model, &["anthropic/claude"]);
            let is_gemini = model.contains("google/gemini");
            let enable_gemini_system_prompt_cache =
                config_bool(cfg, "gemini.enableSystemPromptCache", false);

            if body.get("messages").is_some_and(|m| m.is_array()) {
                let enable_signatures = config_bool(cfg, "gemini.thoughtSignatures", true);
                let claude_ttl = if config_bool(cfg, "claude.extendedTTL", false) {
                    Some("1h")
                } else {
                    Some("5m")
                };
                let enable_system_prompt_cache =
                    config_bool(cfg, "claude.enableSystemPromptCache", false);
                let caching_at_depth = config_i64(cfg, "claude.cachingAtDepth", -1);
                let messages = body.get_mut("messages").expect("checked above");
                prompt::embed_open_router_media(messages, true, true);
                prompt::add_open_router_signatures(messages, &model, enable_signatures);
                if is_claude {
                    if enable_system_prompt_cache {
                        prompt::caching_system_prompt_for_open_router(messages, claude_ttl);
                    }
                    if caching_at_depth >= 0 {
                        prompt::caching_at_depth_for_open_router_claude(
                            messages,
                            caching_at_depth,
                            claude_ttl,
                        );
                    }
                }
                if is_gemini && enable_gemini_system_prompt_cache {
                    prompt::caching_system_prompt_for_open_router(messages, None);
                }
            }
            if is_gemini {
                body_params.insert("safety_settings".into(), gemini_safety());
            }
        }
        "custom" => {
            api_url = get_str(body, "custom_url").unwrap_or("").to_string();
            api_key = read_secret(keys::CUSTOM);
            // JSON.stringify drops undefined values, so an absent logprobs
            // never reaches the provider.
            let logprobs = body.get("logprobs").cloned().filter(|v| !v.is_null());
            if let Some(logprobs) = &logprobs {
                // Upstream sends `{ logprobs: <n>, top_logprobs: undefined }`,
                // and flips `logprobs` to `true` afterwards so the serialized
                // key order is `logprobs, top_logprobs`.
                body_params.insert("logprobs".into(), logprobs.clone());
                body_params.insert("top_logprobs".into(), Value::Null);
                if !is_text_completion && logprobs.as_f64().unwrap_or(0.0) > 0.0 {
                    body_params.insert("top_logprobs".into(), logprobs.clone());
                    body_params.insert("logprobs".into(), Value::Bool(true));
                }
            }
            merge_object_with_yaml(&mut body_params, get_str(body, "custom_include_body"));
            if let Some(include_headers) = get_str(body, "custom_include_headers") {
                let mut header_map = Map::new();
                merge_object_with_yaml(&mut header_map, Some(include_headers));
                for (k, v) in header_map {
                    if let Some(text) = value_to_header_string(&v) {
                        headers.push((k, text));
                    }
                }
            }
            if let Some(messages) = body.get_mut("messages") {
                prompt::embed_open_router_media(messages, true, false);
            }
            if let Some(schema) = json_schema_of(body) {
                body_params.insert(
                    "response_format".into(),
                    json!({
                        "type": "json_schema",
                        "json_schema": {
                            "name": schema.get("name").cloned().unwrap_or(Value::Null),
                            "strict": schema.get("strict").and_then(|v| v.as_bool()).unwrap_or(true),
                            "schema": schema.get("value").cloned().unwrap_or(Value::Null),
                        }
                    }),
                );
            }
        }
        "perplexity" => {
            api_url = API_PERPLEXITY.to_string();
            api_key = read_secret(keys::PERPLEXITY);
            body_params.insert(
                "reasoning_effort".into(),
                body.get("reasoning_effort").cloned().unwrap_or(Value::Null),
            );
            // Perplexity always forces a strict prompt.
            let names = prompt::PromptNames::from_body(body);
            if let Some(messages) = body.get_mut("messages") {
                let placeholder = config_str(cfg, "promptPlaceholder")
                    .unwrap_or_else(|| prompt::DEFAULT_PROMPT_PLACEHOLDER.to_string());
                prompt::post_process_prompt(
                    messages,
                    prompt::PromptProcessingType::Strict,
                    &names,
                    &placeholder,
                );
            }
            if let Some(schema) = json_schema_of(body) {
                body_params.insert(
                    "response_format".into(),
                    json!({
                        "type": "json_schema",
                        "json_schema": {
                            "schema": schema.get("value").cloned().unwrap_or(Value::Null),
                        }
                    }),
                );
            }
        }
        "groq" => {
            api_url = API_GROQ.to_string();
            api_key = read_secret(keys::GROQ);
            if let Some(schema) = json_schema_of(body) {
                body_params.insert(
                    "response_format".into(),
                    json!({
                        "type": "json_schema",
                        "json_schema": {
                            "name": schema.get("name").cloned().unwrap_or(Value::Null),
                            "description": schema.get("description").cloned().unwrap_or(Value::Null),
                            "schema": schema.get("value").cloned().unwrap_or(Value::Null),
                            "strict": schema.get("strict").and_then(|v| v.as_bool()).unwrap_or(true),
                        }
                    }),
                );
            }
        }
        "fireworks" => {
            api_url = API_FIREWORKS.to_string();
            api_key = read_secret(keys::FIREWORKS);
            if let Some(effort) = get_str(body, "reasoning_effort").filter(|s| !s.is_empty()) {
                body_params.insert("reasoning_effort".into(), json!(effort));
            }
            if let Some(schema) = json_schema_of(body) {
                body_params.insert(
                    "response_format".into(),
                    json!({
                        "type": "json_schema",
                        "json_schema": {
                            "name": schema.get("name").cloned().unwrap_or(Value::Null),
                            "description": schema.get("description").cloned().unwrap_or(Value::Null),
                            "schema": schema.get("value").cloned().unwrap_or(Value::Null),
                            "strict": schema.get("strict").and_then(|v| v.as_bool()).unwrap_or(true),
                        }
                    }),
                );
            }
            if let Some(chat_id) = get_str(body, "chat_id").filter(|s| !s.is_empty()) {
                // x-session-affinity = HMAC-SHA256(instance cookie secret, chatId)
                // hex[:16], mirroring `getAffinityKey()` upstream.
                let key = std::fs::read_to_string(cookie_secret_path)
                    .unwrap_or_default()
                    .trim_end_matches(['\n', '\r'])
                    .to_string();
                let hmac = hmac_sha256_hex16(&key, chat_id.as_bytes());
                headers.push(("x-session-affinity".to_string(), hmac));
            }
        }
        "nanogpt" => {
            api_url = API_NANOGPT.to_string();
            api_key = read_secret(keys::NANOGPT);
            if let Some(provider) = get_str(body, "nanogpt_provider").filter(|s| !s.is_empty()) {
                headers.push(("X-Provider".to_string(), provider.to_string()));
            }
            if get_bool(body, "nanogpt_payg_override") {
                headers.push(("X-Billing-Mode".to_string(), "paygo".to_string()));
                body_params.insert("billing_mode".into(), json!("paygo"));
            }
            let online = model.ends_with(":online");
            if get_bool(body, "enable_web_search") && !online {
                let suffixed = format!("{model}:online");
                if let Some(obj) = body.as_object_mut() {
                    obj.insert("model".into(), Value::String(suffixed));
                }
            }
            if let Some(v) = body.get("min_p").filter(|v| !v.is_null()) {
                body_params.insert("min_p".into(), v.clone());
            }
            if let Some(v) = body.get("top_a").filter(|v| !v.is_null()) {
                body_params.insert("top_a".into(), v.clone());
            }
            if let Some(v) = body.get("repetition_penalty").filter(|v| !v.is_null()) {
                body_params.insert("repetition_penalty".into(), v.clone());
            }
            if let Some(effort) = get_str(body, "reasoning_effort").filter(|s| !s.is_empty()) {
                body_params.insert(
                    "reasoning".into(),
                    json!({ "effort": map_nanogpt_effort(effort) }),
                );
            }
            let is_claude = model.contains("claude-") || model.contains("claude_");
            let enable_system_prompt_cache =
                config_bool(cfg, "claude.enableSystemPromptCache", false);
            if enable_system_prompt_cache && is_claude {
                let ttl = if config_bool(cfg, "claude.extendedTTL", false) {
                    "1h"
                } else {
                    "5m"
                };
                body_params.insert(
                    "cache_control".into(),
                    json!({ "enabled": true, "ttl": ttl }),
                );
            }
        }
        "pollinations" => {
            let anonymous = get_str(body, "pollinations_endpoint") == Some("anonymous");
            api_url = if anonymous {
                API_POLLINATIONS_ANON.to_string()
            } else {
                API_POLLINATIONS.to_string()
            };
            api_key = if anonymous {
                "anonymous".to_string()
            } else {
                read_secret(keys::POLLINATIONS)
            };
            let seed = body
                .get("seed")
                .filter(|v| !v.is_null())
                .cloned()
                .unwrap_or_else(|| json!(rand_u32(99_999_999)));
            body_params.insert("seed".into(), seed);
            if !anonymous {
                body_params.insert(
                    "reasoning_effort".into(),
                    body.get("reasoning_effort").cloned().unwrap_or(Value::Null),
                );
                if let Some(schema) = json_schema_of(body) {
                    body_params.insert(
                        "response_format".into(),
                        json!({
                            "type": "json_schema",
                            "json_schema": {
                                "schema": schema.get("value").cloned().unwrap_or(Value::Null),
                            }
                        }),
                    );
                }
            }
        }
        "moonshot" => {
            let reverse_proxy = get_str(body, "reverse_proxy").filter(|s| !s.is_empty());
            api_url = normalize_url(reverse_proxy.unwrap_or(API_MOONSHOT));
            api_key = if reverse_proxy.is_some() {
                get_str(body, "proxy_password").unwrap_or("").to_string()
            } else {
                read_secret(keys::MOONSHOT)
            };
            let include_reasoning = get_bool(body, "include_reasoning");
            body_params.insert(
                "thinking".into(),
                json!({ "type": if include_reasoning { "enabled" } else { "disabled" } }),
            );
            if let Some(schema) = json_schema_of(body) {
                let schema = Value::Object(schema.clone());
                if let Some(messages) = body.get_mut("messages") {
                    prompt::set_json_object_format(&mut body_params, messages, &schema);
                }
            } else if let Some(messages) = body.get_mut("messages") {
                prompt::add_assistant_prefix(messages, Some(&json!([])), "partial");
            }
        }
        "cometapi" => {
            // Upstream throws here; the route maps it to 502 like any other
            // generation failure.
            return Err(PrepareError::Error(
                "This provider is temporarily disabled.".to_string(),
            ));
        }
        "zai" => {
            let default_url = if get_str(body, "zai_endpoint") == Some("coding") {
                API_ZAI_CODING
            } else {
                API_ZAI_COMMON
            };
            let reverse_proxy = get_str(body, "reverse_proxy").filter(|s| !s.is_empty());
            api_url = normalize_url(reverse_proxy.unwrap_or(default_url));
            api_key = if reverse_proxy.is_some() {
                get_str(body, "proxy_password").unwrap_or("").to_string()
            } else {
                read_secret(keys::ZAI)
            };
            headers.push(("Accept-Language".to_string(), "en-US,en".to_string()));
            let include_reasoning = get_bool(body, "include_reasoning");
            body_params.insert(
                "thinking".into(),
                json!({ "type": if include_reasoning { "enabled" } else { "disabled" } }),
            );
            if let Some(schema) = json_schema_of(body) {
                let schema = Value::Object(schema.clone());
                if let Some(messages) = body.get_mut("messages") {
                    prompt::set_json_object_format(&mut body_params, messages, &schema);
                }
            }
        }
        "siliconflow" => {
            api_url = if get_str(body, "siliconflow_endpoint") == Some("cn") {
                API_SILICONFLOW_CN.to_string()
            } else {
                API_SILICONFLOW.to_string()
            };
            api_key = read_secret(keys::SILICONFLOW);
            if let Some(schema) = json_schema_of(body) {
                let schema = Value::Object(schema.clone());
                if let Some(messages) = body.get_mut("messages") {
                    prompt::set_json_object_format(&mut body_params, messages, &schema);
                }
            }
        }
        "workers_ai" => {
            api_key = read_secret(keys::WORKERS_AI);
            let account_id = get_str(body, "workers_ai_account_id")
                .unwrap_or("")
                .trim()
                .to_string();
            if account_id.is_empty() {
                tracing::warn!("Cloudflare Workers AI Account ID is missing.");
                return Err(PrepareError::BadRequest("missing workers ai account id"));
            }
            api_url = format!(
                "{API_WORKERS_AI}/{}/ai/v1",
                encode_uri_component(&account_id)
            );
            body_params.insert(
                "repetition_penalty".into(),
                body.get("repetition_penalty")
                    .cloned()
                    .unwrap_or(Value::Null),
            );
            if let Some(schema) = json_schema_of(body) {
                body_params.insert(
                    "response_format".into(),
                    json!({
                        "type": "json_schema",
                        "json_schema": schema.get("value").cloned().unwrap_or(Value::Null),
                    }),
                );
            }
        }
        other => {
            tracing::warn!(source = other, "chat completion source not handled inline");
            return Err(PrepareError::BadRequest(
                "unsupported chat completion source",
            ));
        }
    }

    // --- OpenAI/Custom reasoning effort + verbosity ---------------------------
    let effort = get_str(body, "reasoning_effort")
        .filter(|s| !s.is_empty())
        .map(str::to_string);
    if let Some(effort) = effort {
        if source == "custom" || source == "openai" {
            if OPENAI_REASONING_EFFORT_MODELS.contains(&model.as_str()) {
                let mapped = fixed_openai_effort(&model)
                    .map(str::to_string)
                    .unwrap_or_else(|| map_openai_effort(&effort).to_string());
                body_params.insert("reasoning_effort".into(), json!(mapped));
            }
            if source == "custom" {
                if let Some(rest) = model.strip_prefix("koboldcpp/") {
                    if !rest.is_empty() {
                        body_params.insert("reasoning_effort".into(), json!(effort));
                    }
                }
            }
        }
    }
    if let Some(verbosity) = get_str(body, "verbosity").filter(|s| !s.is_empty()) {
        if (source == "custom" || source == "openai") && is_verbosity_model(&model) {
            body_params.insert("verbosity".into(), json!(verbosity));
        }
    }

    let reverse_proxy_present = get_str(body, "reverse_proxy")
        .map(|s| !s.is_empty())
        .unwrap_or(false);
    if api_key.is_empty() && !reverse_proxy_present && source != "custom" {
        tracing::warn!("OpenAI API key is missing.");
        return Err(PrepareError::BadRequest("missing api key"));
    }

    // --- custom stop sequences ------------------------------------------------
    if let Some(stop) = body.get("stop").and_then(|v| v.as_array()) {
        if !stop.is_empty() {
            body_params.insert("stop".into(), Value::Array(stop.clone()));
        }
    }

    let text_prompt = if is_text_completion {
        prompt::convert_text_completion_prompt(body.get("messages").unwrap_or(&Value::Null))
    } else {
        String::new()
    };
    let endpoint_url = if is_text_completion && source != "openrouter" {
        format!("{api_url}/completions")
    } else {
        format!("{api_url}/chat/completions")
    };

    if !is_text_completion {
        if let Some(tools) = body.get("tools").and_then(|t| t.as_array()) {
            if !tools.is_empty() {
                body_params.insert("tools".into(), Value::Array(tools.clone()));
                body_params.insert(
                    "tool_choice".into(),
                    body.get("tool_choice").cloned().unwrap_or(Value::Null),
                );
            }
        }
    }

    if let Some(schema) = json_schema_of(body) {
        if !body_params.contains_key("response_format") {
            body_params.insert(
                "response_format".into(),
                json!({
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema.get("name").cloned().unwrap_or(Value::Null),
                        "strict": schema.get("strict").and_then(|v| v.as_bool()).unwrap_or(true),
                        "schema": schema.get("value").cloned().unwrap_or(Value::Null),
                    }
                }),
            );
        }
    }

    // --- request body, in upstream property order -----------------------------
    let mut request_body = Map::new();
    if is_text_completion {
        request_body.insert("messages".into(), Value::Null);
        request_body.insert("prompt".into(), Value::String(text_prompt));
    } else {
        request_body.insert(
            "messages".into(),
            body.get("messages").cloned().unwrap_or(Value::Null),
        );
        request_body.insert("prompt".into(), Value::Null);
    }
    let mut put = |key: &str, value: Value| {
        request_body.insert(key.to_string(), value);
    };
    put("model", body.get("model").cloned().unwrap_or(Value::Null));
    put(
        "temperature",
        body.get("temperature").cloned().unwrap_or(Value::Null),
    );
    put(
        "max_tokens",
        body.get("max_tokens").cloned().unwrap_or(Value::Null),
    );
    put(
        "max_completion_tokens",
        body.get("max_completion_tokens")
            .cloned()
            .unwrap_or(Value::Null),
    );
    put("stream", body.get("stream").cloned().unwrap_or(Value::Null));
    put(
        "presence_penalty",
        body.get("presence_penalty").cloned().unwrap_or(Value::Null),
    );
    put(
        "frequency_penalty",
        body.get("frequency_penalty")
            .cloned()
            .unwrap_or(Value::Null),
    );
    put("top_p", body.get("top_p").cloned().unwrap_or(Value::Null));
    put("top_k", body.get("top_k").cloned().unwrap_or(Value::Null));
    put(
        "stop",
        if is_text_completion {
            Value::Null
        } else {
            body.get("stop").cloned().unwrap_or(Value::Null)
        },
    );
    put(
        "logit_bias",
        body.get("logit_bias").cloned().unwrap_or(Value::Null),
    );
    put("seed", body.get("seed").cloned().unwrap_or(Value::Null));
    put("n", body.get("n").cloned().unwrap_or(Value::Null));
    for (k, v) in &body_params {
        request_body.insert(k.clone(), v.clone());
    }

    if source == "custom" {
        let exclude = get_str(body, "custom_exclude_body").map(str::to_string);
        exclude_keys_by_yaml(&mut request_body, exclude.as_deref());
    }

    // `JSON.stringify` drops undefined values; serde_json::Value::Null is the
    // placeholder used above, so strip the ones upstream left undefined.
    request_body.retain(|_, v| !v.is_null());

    let mut final_headers: Vec<(String, String)> = vec![
        ("Content-Type".to_string(), "application/json".to_string()),
        ("Authorization".to_string(), format!("Bearer {api_key}")),
    ];
    final_headers.extend(headers);

    let stream = get_bool(body, "stream");

    Ok(PreparedRequest {
        url: endpoint_url,
        headers: final_headers,
        body: Value::Object(request_body),
        stream,
    })
}

fn json_schema_of(body: &Value) -> Option<&Map<String, Value>> {
    body.get("json_schema")
        .filter(|v| v.is_object())
        .and_then(|v| v.as_object())
}

fn value_to_header_string(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.clone()),
        Value::Number(n) => Some(n.to_string()),
        Value::Bool(b) => Some(b.to_string()),
        _ => None,
    }
}

/// `encodeURIComponent` — everything except `A-Za-z0-9-_.!~*'()` is escaped.
pub fn encode_uri_component(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for byte in input.bytes() {
        let keep = byte.is_ascii_alphanumeric()
            || matches!(
                byte,
                b'-' | b'_' | b'.' | b'!' | b'~' | b'*' | b'\'' | b'(' | b')'
            );
        if keep {
            out.push(byte as char);
        } else {
            out.push_str(&format!("%{byte:02X}"));
        }
    }
    out
}

// --- config value helpers (mirror of getConfigValue) ------------------------

fn env_override(key: &str) -> Option<String> {
    let env_key = format!("SILLYTAVERN_{}", key.to_uppercase().replace('.', "_"));
    std::env::var(env_key).ok()
}

fn config_raw(cfg: &crate::config::StConfig, key: &str) -> Option<serde_yaml::Value> {
    let (section, item) = key.split_once('.')?;
    let section = cfg.raw_section(section)?;
    section.get(item).cloned()
}

fn config_bool(cfg: &crate::config::StConfig, key: &str, default: bool) -> bool {
    if let Some(raw) = env_override(key) {
        return parse_bool_text(&raw);
    }
    match config_raw(cfg, key) {
        Some(serde_yaml::Value::Bool(b)) => b,
        Some(serde_yaml::Value::String(s)) => parse_bool_text(&s),
        _ => default,
    }
}

fn parse_bool_text(text: &str) -> bool {
    match text.trim().to_ascii_lowercase().as_str() {
        "true" | "1" | "yes" | "on" => true,
        "false" | "0" | "no" | "off" | "" => false,
        _ => true,
    }
}

fn config_i64(cfg: &crate::config::StConfig, key: &str, default: i64) -> i64 {
    if let Some(raw) = env_override(key) {
        return raw.trim().parse().unwrap_or(default);
    }
    match config_raw(cfg, key) {
        Some(serde_yaml::Value::Number(n)) => n.as_i64().unwrap_or(default),
        Some(serde_yaml::Value::String(s)) => s.trim().parse().unwrap_or(default),
        _ => default,
    }
}

fn config_str(cfg: &crate::config::StConfig, key: &str) -> Option<String> {
    if let Some(raw) = env_override(key) {
        return Some(raw);
    }
    match config_raw(cfg, key)? {
        serde_yaml::Value::String(s) => Some(s),
        other => serde_yaml::to_string(&other)
            .ok()
            .map(|s| s.trim().to_string()),
    }
}

// --- small utilities --------------------------------------------------------

fn hex_encode(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push_str(&format!("{b:02x}"));
    }
    out
}

/// HMAC-SHA256(key, data) hex, truncated to 16 chars (upstream
/// `createHmac('sha256', key).update(id).digest('hex').slice(0, 16)`).
pub fn hmac_sha256_hex16(key: &str, data: &[u8]) -> String {
    use hmac::Mac;
    let mut mac = hmac::Hmac::<sha2::Sha256>::new_from_slice(key.as_bytes())
        .expect("HMAC accepts keys of any size");
    mac.update(data);
    let digest = mac.finalize().into_bytes();
    hex_encode(&digest)[..16].to_string()
}

fn uuid_v4() -> String {
    // Openssl-free, dependency-free v4 layout: 16 random bytes with the
    // version/variant bits patched, formatted like `uuidv4()`.
    let mut bytes = [0u8; 16];
    let mut seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
        ^ (std::process::id() as u64) << 32;
    for chunk in bytes.chunks_mut(8) {
        // xorshift64*
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        let le = seed.to_le_bytes();
        chunk.copy_from_slice(&le[..chunk.len()]);
    }
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    let h = hex_encode(&bytes);
    format!(
        "{}-{}-{}-{}-{}",
        &h[0..8],
        &h[8..12],
        &h[12..16],
        &h[16..20],
        &h[20..32]
    )
}

fn rand_u32(max_exclusive: u32) -> u32 {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    nanos % max_exclusive.max(1)
}

/// `get_bool_ctx` is only used for the "messages is an array" pre-check.
fn get_bool_ctx(body: &Value, key: &str) -> bool {
    body.get(key).and_then(|v| v.as_bool()).unwrap_or(false)
}

// ---------------------------------------------------------------------------
// Concurrency gate
// ---------------------------------------------------------------------------

/// Size-weighted generation gate.
///
/// Node kept every request body alive (plus its JSON.stringify copy) for the
/// whole generation, so N concurrent image-heavy chats multiplied the peak
/// RSS. The gateway keeps the same guarantee without the copies by charging
/// each request `1 + body_bytes / unit` permits against a shared semaphore
/// sized from `gateway.batch_bytes`; the permit is released the moment the
/// upstream response finishes streaming.
pub struct GenerateLimiter {
    semaphore: std::sync::Arc<tokio::sync::Semaphore>,
    pub total: usize,
    unit: usize,
}

impl GenerateLimiter {
    /// `budget_bytes` (gateway.batch_bytes, default 256 MiB) is divided into
    /// `unit`-sized shares; at least one request always fits.
    pub fn new(budget_bytes: u64) -> Self {
        let unit = 16 * 1024 * 1024;
        let total = (budget_bytes / unit).clamp(1, 256) as usize;
        Self {
            semaphore: std::sync::Arc::new(tokio::sync::Semaphore::new(total)),
            total,
            unit: unit as usize,
        }
    }

    /// Permits a body of `body_bytes` costs (always >= 1).
    pub fn permits_for(&self, body_bytes: u64) -> u32 {
        let shares = (body_bytes as usize) / self.unit;
        (shares + 1).min(self.total) as u32
    }

    /// Acquire admission. Returns `None` when the caller must wait for the
    /// semaphore to be closed (never happens in practice).
    pub async fn acquire(&self, body_bytes: u64) -> Option<tokio::sync::OwnedSemaphorePermit> {
        let permits = self.permits_for(body_bytes);
        let sem = self.semaphore.clone();
        sem.acquire_many_owned(permits).await.ok()
    }
}

impl Default for GenerateLimiter {
    fn default() -> Self {
        Self::new(256 * 1024 * 1024)
    }
}

// ---------------------------------------------------------------------------
// Response protocol detection
// ---------------------------------------------------------------------------

/// Wire protocol of an upstream chat-completions response, decided from its
/// content type. Only SSE has an in-band comment syntax; every
/// line-delimited format tolerates blank lines; a whole-JSON body tolerates
/// leading whitespace.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Protocol {
    /// `text/event-stream` — `: keep-alive` comment lines.
    Sse,
    /// `application/x-ndjson`, `application/jsonl`, ... — blank lines.
    Ndjson,
    /// One JSON document — leading whitespace before the body only.
    Json,
    /// Content type missing/unknown: stay maximally compatible.
    #[default]
    Unknown,
}

impl Protocol {
    /// Classify by the upstream `Content-Type` header value.
    pub fn from_content_type(value: Option<&str>) -> Self {
        let Some(value) = value else {
            return Protocol::Unknown;
        };
        let mime = value
            .split(';')
            .next()
            .unwrap_or("")
            .trim()
            .to_ascii_lowercase();
        match mime.as_str() {
            "text/event-stream" => Protocol::Sse,
            "application/x-ndjson"
            | "application/jsonl"
            | "application/ndjson"
            | "text/x-ndjson" => Protocol::Ndjson,
            "application/json" => Protocol::Json,
            // Providers that stream bare JSON lines under a text-ish media
            // type are only treated as streamed when the caller asked for it.
            "text/plain" | "text/json" => Protocol::Unknown,
            _ => Protocol::Unknown,
        }
    }

    /// Classify with knowledge of the request mode: a `stream: true` request
    /// answered with a text-ish but unlabelled type is line-delimited.
    pub fn for_request(content_type: Option<&str>, stream_requested: bool) -> Self {
        match Self::from_content_type(content_type) {
            Protocol::Unknown if stream_requested => Protocol::Ndjson,
            other => other,
        }
    }
}
