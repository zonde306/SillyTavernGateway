//! Non-OpenAI chat-completion providers.
//!
//! Upstream dispatches these from a `switch` in
//! `src/endpoints/backends/chat-completions.js`, each to its own
//! `send<Provider>Request`. They used to be forwarded to Node verbatim; this
//! module builds the provider request in-process so the gateway never needs
//! Node for `/api/backends/chat-completions/generate`.
//!
//! Two shapes live here:
//!
//! * [`OpenAiFlavour`] — providers that speak plain OpenAI chat completions
//!   and only differ in base URL, auth header and a few body fields
//!   (DeepSeek, xAI, AI/ML API, Electron Hub, Chutes, MiniMax, Mistral,
//!   Azure OpenAI). Their response is passed through untouched.
//! * bespoke builders for Claude, Gemini (AI Studio + Vertex), AI21 and
//!   Cohere, whose request *and* non-streaming response differ enough to need
//!   translation.
//!
//! Streaming never needs translation in any of these: every provider emits
//! SSE (or, for Gemini non-SSE, a JSON array) that the client already
//! understands, so the response is piped through byte-for-byte.

use serde_json::{json, Map, Value};

use crate::chat_completions::{flatten_schema, normalize_url, PrepareError, PreparedRequest};
use crate::converters;
use crate::prompt::{self, PromptNames};
use crate::secrets::{self, keys};

// --- endpoints (src/endpoints/backends/chat-completions.js:74-99) ----------

pub const API_CLAUDE: &str = "https://api.anthropic.com/v1";
pub const API_MISTRAL: &str = "https://api.mistral.ai/v1";
pub const API_COHERE_V2: &str = "https://api.cohere.ai/v2";
pub const API_MAKERSUITE: &str = "https://generativelanguage.googleapis.com";
pub const API_VERTEX_AI: &str = "https://us-central1-aiplatform.googleapis.com";
pub const API_AI21: &str = "https://api.ai21.com/studio/v1";
pub const API_CHUTES: &str = "https://llm.chutes.ai/v1";
pub const API_ELECTRONHUB: &str = "https://api.electronhub.ai/v1";
pub const API_DEEPSEEK: &str = "https://api.deepseek.com/beta";
pub const API_XAI: &str = "https://api.x.ai/v1";
pub const API_AIMLAPI: &str = "https://api.aimlapi.com/v1";
pub const API_MINIMAX: &str = "https://api.minimax.io/v1";
pub const API_MINIMAX_CN: &str = "https://api.minimaxi.com/v1";
pub const API_VERTEX_OAUTH: &str = "https://oauth2.googleapis.com/token";

/// `AIMLAPI_HEADERS` (src/constants.js:367).
const AIMLAPI_HEADERS: &[(&str, &str)] = &[
    ("HTTP-Referer", "https://sillytavern.app"),
    ("X-Title", "SillyTavern"),
];

/// `AZURE_OPENAI_KEYS` (src/constants.js:440) — the only body fields Azure
/// accepts, copied in this order.
const AZURE_OPENAI_KEYS: &[&str] = &[
    "messages",
    "temperature",
    "frequency_penalty",
    "presence_penalty",
    "top_p",
    "max_tokens",
    "max_completion_tokens",
    "stream",
    "logit_bias",
    "stop",
    "n",
    "logprobs",
    "seed",
    "tools",
    "tool_choice",
    "reasoning_effort",
];

/// `VERTEX_SAFETY` (src/constants.js:164) — Vertex adds image-category blocks.
fn vertex_safety() -> Value {
    json!([
        { "category": "HARM_CATEGORY_IMAGE_HATE", "threshold": "OFF" },
        { "category": "HARM_CATEGORY_IMAGE_DANGEROUS_CONTENT", "threshold": "OFF" },
        { "category": "HARM_CATEGORY_IMAGE_HARASSMENT", "threshold": "OFF" },
        { "category": "HARM_CATEGORY_IMAGE_SEXUALLY_EXPLICIT", "threshold": "OFF" },
        { "category": "HARM_CATEGORY_JAILBREAK", "threshold": "OFF" }
    ])
}

/// `GEMINI_SAFETY` (src/constants.js:141) — "OFF", not "BLOCK_NONE".
pub fn gemini_safety() -> Value {
    json!([
        { "category": "HARM_CATEGORY_HARASSMENT", "threshold": "OFF" },
        { "category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "OFF" },
        { "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "OFF" },
        { "category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "OFF" },
        { "category": "HARM_CATEGORY_CIVIC_INTEGRITY", "threshold": "OFF" }
    ])
}

// --- helpers ---------------------------------------------------------------

fn get_str<'a>(body: &'a Value, key: &str) -> Option<&'a str> {
    body.get(key).and_then(|v| v.as_str())
}

fn get_bool(body: &Value, key: &str) -> bool {
    body.get(key).and_then(|v| v.as_bool()).unwrap_or(false)
}

fn get_f64(body: &Value, key: &str) -> Option<f64> {
    body.get(key).and_then(|v| v.as_f64())
}

/// `JSON.stringify(value, null, 4)` — `serde_json` only offers 2-space pretty
/// printing, so re-indent it.
fn pretty_json_4(value: &Value) -> String {
    let pretty = serde_json::to_string_pretty(value).unwrap_or_default();
    pretty
        .lines()
        .map(|line| {
            // Each nesting level already contributed two spaces.
            let depth = line.len() - line.trim_start().len();
            let mut out = String::with_capacity(line.len() + depth);
            out.push_str(&" ".repeat(depth * 2));
            out.push_str(line.trim_start());
            out
        })
        .collect::<Vec<_>>()
        .join(
            "
",
        )
}

/// A number that renders as an integer when it is one. `serde_json` prints
/// `1524.0` for an `f64`, while JavaScript prints `1524`; providers compare
/// these against JSON schema types, so the difference matters.
pub fn num(value: f64) -> Value {
    if value.fract() == 0.0 && value.abs() < 9e15 {
        Value::from(value as i64)
    } else {
        Value::from(value)
    }
}

/// Copy `from` into `to` only when the source key exists and is not null.
/// Mirrors `JSON.stringify` dropping `undefined`.
fn copy_if_present(to: &mut Map<String, Value>, key: &str, from: &Value) {
    if let Some(value) = from.get(key) {
        if !value.is_null() {
            to.insert(key.to_string(), value.clone());
        }
    }
}

/// Resolve a provider's base URL and key the way every `send*Request` does:
/// a `reverse_proxy` switches both the URL and where the key comes from.
fn resolve_endpoint(
    body: &Value,
    default_url: &str,
    secret_key: &str,
    secrets_doc: &Value,
    secret_id: Option<&str>,
) -> (String, String) {
    let reverse_proxy = get_str(body, "reverse_proxy").filter(|s| !s.is_empty());
    let url = normalize_url(reverse_proxy.unwrap_or(default_url));
    let key = if reverse_proxy.is_some() {
        get_str(body, "proxy_password").unwrap_or("").to_string()
    } else {
        secrets::read_secret(secrets_doc, secret_key, secret_id)
    };
    (url, key)
}

/// The credential a provider needs, so the caller can produce upstream's
/// `400 {error:true}` before building anything.
pub struct ProviderRequest {
    pub prepared: PreparedRequest,
    /// Non-streaming responses to rewrite before the client sees them.
    pub transform: ResponseTransform,
}

/// How to fix up a non-streaming JSON response. Streaming is always a raw
/// byte pipe: every provider already emits client-compatible SSE.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResponseTransform {
    /// Provider already returns OpenAI-shaped JSON.
    None,
    /// Anthropic: wrap `content[0].text` into `choices[0].message.content`.
    Claude,
    /// Gemini: wrap `candidates[0].content` into `choices[0].message.content`.
    Gemini,
    /// AI21 / Cohere / Mistral return their own JSON; upstream passes it
    /// through unchanged, so the client is expected to understand it.
    Passthrough,
}

// ---------------------------------------------------------------------------
// OpenAI-compatible providers
// ---------------------------------------------------------------------------

/// Which OpenAI-style flavour to build. Each entry documents the only real
/// differences from the inline chain: endpoint, auth, and the extra body
/// fields upstream adds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenAiFlavour {
    DeepSeek,
    Xai,
    AimlApi,
    ElectronHub,
    Chutes,
    MiniMax,
    Mistral,
    AzureOpenAi,
}

impl OpenAiFlavour {
    /// Parse a `chat_completion_source` value.
    pub fn parse(source: &str) -> Option<Self> {
        Some(match source {
            "deepseek" => Self::DeepSeek,
            "xai" => Self::Xai,
            "aimlapi" => Self::AimlApi,
            "electronhub" => Self::ElectronHub,
            "chutes" => Self::Chutes,
            "minimax" => Self::MiniMax,
            "mistralai" => Self::Mistral,
            "azure_openai" => Self::AzureOpenAi,
            _ => return None,
        })
    }
}

/// `postProcessPrompt(messages, MERGE_TOOLS, names)` — MiniMax rejects two
/// consecutive same-role turns.
fn merge_tools_body(body: &Value, names: &PromptNames, placeholder: &str) -> Value {
    let mut messages = body.get("messages").cloned().unwrap_or(json!([]));
    prompt::post_process_prompt(
        &mut messages,
        prompt::PromptProcessingType::MergeTools,
        names,
        placeholder,
    );
    messages
}

/// `addReasoningContentToToolCalls`: DeepSeek wants `reasoning_content` on
/// tool-call messages.
fn add_reasoning_content(messages: &mut Value) {
    let Some(arr) = messages.as_array_mut() else {
        return;
    };
    for message in arr.iter_mut() {
        let Some(obj) = message.as_object_mut() else {
            continue;
        };
        let has_calls = obj.get("tool_calls").is_some_and(|t| t.is_array());
        if !has_calls || obj.contains_key("reasoning_content") {
            continue;
        }
        obj.insert("reasoning_content".into(), json!(""));
    }
}

/// Build an OpenAI-compatible provider request.
pub fn build_openai_flavour(
    flavour: OpenAiFlavour,
    body: &Value,
    secrets_doc: &Value,
    cfg: &crate::config::StConfig,
    names: &PromptNames,
    placeholder: &str,
) -> Result<ProviderRequest, PrepareError> {
    let secret_id = get_str(body, "secret_id");
    let model = get_str(body, "model").unwrap_or("").to_string();
    let stream = get_bool(body, "stream");

    // --- Azure is a different URL shape and header, handle it first ---------
    if matches!(flavour, OpenAiFlavour::AzureOpenAi) {
        let base_url = get_str(body, "azure_base_url").unwrap_or("");
        let deployment = get_str(body, "azure_deployment_name").unwrap_or("");
        let api_version = get_str(body, "azure_api_version").unwrap_or("");
        let api_key = secrets::read_secret(secrets_doc, keys::AZURE_OPENAI, secret_id);
        if base_url.is_empty()
            || deployment.is_empty()
            || api_version.is_empty()
            || api_key.is_empty()
        {
            return Err(PrepareError::Message(
                "Azure OpenAI configuration is incomplete. Please provide Base URL, Deployment Name, \
                 API Version, and API Key in the connection settings."
                    .to_string(),
            ));
        }
        let url = format!(
            "{}/openai/deployments/{}/chat/completions?api-version={}",
            base_url.trim_end_matches('/'),
            crate::chat_completions::encode_uri_component(deployment),
            crate::chat_completions::encode_uri_component(api_version)
        );

        let mut request_body = Map::new();
        for key in AZURE_OPENAI_KEYS {
            copy_if_present(&mut request_body, key, body);
        }
        if let Some(schema) = body.get("json_schema").filter(|v| v.is_object()) {
            request_body.insert(
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
        // Azure follows the OpenAI spec: `logprobs` becomes a boolean plus
        // `top_logprobs`.
        if let Some(logprobs) = get_f64(body, "logprobs").filter(|v| *v > 0.0) {
            request_body.insert("top_logprobs".into(), num(logprobs));
            request_body.insert("logprobs".into(), json!(true));
        }
        // `.includes(model)` — absent model must still be dropped.
        if crate::chat_completions::is_reasoning_effort_model(&model) {
            let effort = crate::chat_completions::resolve_openai_effort(&model, body);
            if let Some(effort) = effort {
                request_body.insert("reasoning_effort".into(), json!(effort));
            }
        } else {
            request_body.remove("reasoning_effort");
        }

        return Ok(ProviderRequest {
            prepared: PreparedRequest {
                url,
                headers: vec![
                    ("Content-Type".into(), "application/json".into()),
                    ("api-key".into(), api_key),
                ],
                body: Value::Object(request_body),
                stream,
                transform: ResponseTransform::None,
            },
            transform: ResponseTransform::None,
        });
    }

    // --- the rest are reverse_proxy + secret key + `/chat/completions` ------
    let (default_url, secret_key) = match flavour {
        OpenAiFlavour::DeepSeek => (API_DEEPSEEK, keys::DEEPSEEK),
        OpenAiFlavour::Xai => (API_XAI, keys::XAI),
        OpenAiFlavour::AimlApi => (API_AIMLAPI, keys::AIMLAPI),
        OpenAiFlavour::ElectronHub => (API_ELECTRONHUB, keys::ELECTRONHUB),
        OpenAiFlavour::Chutes => (API_CHUTES, keys::CHUTES),
        // The CN endpoint is a separate host, chosen before the key is read.
        OpenAiFlavour::MiniMax => (
            if get_str(body, "minimax_endpoint") == Some("cn") {
                API_MINIMAX_CN
            } else {
                API_MINIMAX
            },
            keys::MINIMAX,
        ),
        OpenAiFlavour::Mistral => (API_MISTRAL, keys::MISTRALAI),
        OpenAiFlavour::AzureOpenAi => unreachable!("handled above"),
    };

    // AI/ML API, Electron Hub, Chutes and MiniMax ignore `reverse_proxy`
    // upstream (their URL is a constant); the others honour it.
    let honours_reverse_proxy = matches!(
        flavour,
        OpenAiFlavour::DeepSeek | OpenAiFlavour::Xai | OpenAiFlavour::Mistral
    );
    let (url, key) = if honours_reverse_proxy {
        resolve_endpoint(body, default_url, secret_key, secrets_doc, secret_id)
    } else {
        (
            normalize_url(default_url),
            secrets::read_secret(secrets_doc, secret_key, secret_id),
        )
    };
    // Every one of these providers needs a key (upstream answers
    // `400 {error:true}`, which is what `BadRequest` renders as).
    if key.is_empty() {
        return Err(PrepareError::BadRequest("missing api key"));
    }

    let mut body_params = Map::new();
    // DeepSeek processes messages off `request.body.messages`, so the
    // schema hint, the prompt post-processing and the assistant prefix all
    // stack on one mutable array — in that order.
    let mut messages = match flavour {
        OpenAiFlavour::MiniMax => merge_tools_body(body, names, placeholder),
        OpenAiFlavour::Mistral => converters::convert_mistral_messages(
            &body.get("messages").cloned().unwrap_or(json!([])),
            names,
            crate::chat_completions::config_bool(cfg, "mistral.enablePrefix", false),
        ),
        OpenAiFlavour::Xai => converters::convert_xai_messages(
            &body.get("messages").cloned().unwrap_or(json!([])),
            names,
        ),
        _ => body.get("messages").cloned().unwrap_or(json!([])),
    };

    if matches!(flavour, OpenAiFlavour::DeepSeek) {
        // JSON-schema support is emulated with a trailing user message.
        if let Some(schema) = body.get("json_schema").filter(|v| v.is_object()) {
            body_params.insert("response_format".into(), json!({ "type": "json_object" }));
            // `JSON.stringify(value, null, 4)` — four-space indentation.
            let hint = format!(
                "JSON schema for the response:
{}",
                pretty_json_4(schema.get("value").unwrap_or(&Value::Null))
            );
            if let Some(arr) = messages.as_array_mut() {
                arr.push(json!({ "role": "user", "content": hint }));
            }
        }
        prompt::post_process_prompt(
            &mut messages,
            prompt::PromptProcessingType::SemiTools,
            names,
            placeholder,
        );
        prompt::add_assistant_prefix(&mut messages, body_params.get("tools"), "prefix");
        add_reasoning_content(&mut messages);
    }

    // --- per-provider body_params -----------------------------------------
    match flavour {
        OpenAiFlavour::DeepSeek => {
            if get_f64(body, "logprobs").is_some_and(|v| v > 0.0) {
                body_params.insert(
                    "top_logprobs".into(),
                    body.get("logprobs").cloned().unwrap(),
                );
                body_params.insert("logprobs".into(), json!(true));
            }
            if let Some(tools) = body.get("tools").and_then(|t| t.as_array()) {
                if !tools.is_empty() {
                    // DeepSeek rejects empty `required` arrays.
                    let mut tools = tools.clone();
                    for tool in tools.iter_mut() {
                        if let Some(required) = tool
                            .pointer_mut("/function/parameters/required")
                            .and_then(|r| r.as_array_mut())
                        {
                            if required.is_empty() {
                                if let Some(params) = tool
                                    .pointer_mut("/function/parameters")
                                    .and_then(|p| p.as_object_mut())
                                {
                                    params.remove("required");
                                }
                            }
                        }
                    }
                    body_params.insert("tools".into(), Value::Array(tools));
                    body_params.insert(
                        "tool_choice".into(),
                        body.get("tool_choice").cloned().unwrap_or(Value::Null),
                    );
                }
            }
        }
        OpenAiFlavour::Xai => {
            if get_f64(body, "logprobs").is_some_and(|v| v > 0.0) {
                body_params.insert(
                    "top_logprobs".into(),
                    body.get("logprobs").cloned().unwrap(),
                );
                body_params.insert("logprobs".into(), json!(true));
            }
            if let Some(tools) = body.get("tools").and_then(|t| t.as_array()) {
                if !tools.is_empty() {
                    body_params.insert("tools".into(), Value::Array(tools.clone()));
                    body_params.insert(
                        "tool_choice".into(),
                        body.get("tool_choice").cloned().unwrap_or(Value::Null),
                    );
                }
            }
            if let Some(stop) = body.get("stop").and_then(|s| s.as_array()) {
                if !stop.is_empty() {
                    body_params.insert("stop".into(), Value::Array(stop.clone()));
                }
            }
            // xAI accepts only 'high' or 'low'.
            if let Some(effort) = get_str(body, "reasoning_effort").filter(|s| !s.is_empty()) {
                let mapped = if effort == "high" { "high" } else { "low" };
                body_params.insert("reasoning_effort".into(), json!(mapped));
            }
            if let Some(schema) = body.get("json_schema").filter(|v| v.is_object()) {
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
        OpenAiFlavour::AimlApi => {
            if get_f64(body, "logprobs").is_some_and(|v| v > 0.0) {
                body_params.insert(
                    "top_logprobs".into(),
                    body.get("logprobs").cloned().unwrap(),
                );
                body_params.insert("logprobs".into(), json!(true));
            }
            copy_tools(&mut body_params, body);
            copy_stop(&mut body_params, body);
            if let Some(effort) = get_str(body, "reasoning_effort").filter(|s| !s.is_empty()) {
                body_params.insert("reasoning_effort".into(), json!(effort));
            }
            insert_json_schema_verbose(&mut body_params, body, true);
        }
        OpenAiFlavour::ElectronHub => {
            if get_bool(body, "enable_web_search") {
                body_params.insert("web_search".into(), json!(true));
            }
            copy_tools(&mut body_params, body);
            if let Some(effort) = get_str(body, "reasoning_effort").filter(|s| !s.is_empty()) {
                body_params.insert("reasoning_effort".into(), json!(effort));
            }
            insert_json_schema_verbose(&mut body_params, body, true);

            // Prompt caching is only meaningful for Claude models proxied
            // through Electron Hub; upstream applies the breakpoints to
            // `request.body.messages` before serializing, so mutating our
            // copy has the same effect.
            if model.starts_with("claude-") {
                let ttl = crate::chat_completions::claude_cache_ttl(cfg);
                if crate::chat_completions::config_bool(
                    cfg,
                    "claude.enableSystemPromptCache",
                    false,
                ) {
                    prompt::caching_system_prompt_for_open_router(&mut messages, ttl.as_deref());
                }
                let depth = crate::chat_completions::claude_caching_at_depth(cfg);
                if depth != -1 {
                    prompt::caching_at_depth_for_open_router_claude(
                        &mut messages,
                        depth,
                        ttl.as_deref(),
                    );
                }
            }
        }
        OpenAiFlavour::Chutes => {
            copy_tools(&mut body_params, body);
            if get_f64(body, "logprobs").is_some_and(|v| v > 0.0) {
                body_params.insert(
                    "top_logprobs".into(),
                    body.get("logprobs").cloned().unwrap(),
                );
                body_params.insert("logprobs".into(), json!(true));
            }
            insert_json_schema_verbose(&mut body_params, body, true);
        }
        OpenAiFlavour::MiniMax => {
            copy_tools(&mut body_params, body);
        }
        OpenAiFlavour::Mistral => {
            copy_tools(&mut body_params, body);
            insert_json_schema_verbose(&mut body_params, body, true);
        }
        OpenAiFlavour::AzureOpenAi => unreachable!(),
    }

    let mut request_body = build_common_body(flavour, body, &messages);
    // MiniMax caps `M2-her` output at 2048 tokens.
    if matches!(flavour, OpenAiFlavour::MiniMax) && model == "M2-her" {
        if let Some(max_tokens) = get_f64(body, "max_tokens") {
            request_body.insert("max_tokens".into(), json!(max_tokens.min(2048.0)));
        }
    }
    request_body.extend(body_params);

    let mut headers = vec![
        ("Content-Type".to_string(), "application/json".to_string()),
        ("Authorization".to_string(), format!("Bearer {key}")),
    ];
    if matches!(flavour, OpenAiFlavour::AimlApi) {
        for (k, v) in AIMLAPI_HEADERS {
            headers.push((k.to_string(), v.to_string()));
        }
    }

    Ok(ProviderRequest {
        prepared: PreparedRequest {
            url: format!("{}/chat/completions", url.trim_end_matches('/')),
            headers,
            body: Value::Object(request_body),
            stream,
            transform: ResponseTransform::None,
        },
        transform: ResponseTransform::None,
    })
}

/// The shared `{messages, model, temperature, ...}` skeleton, in upstream
/// property order (which is what ends up in the JSON the provider parses).
fn build_common_body(flavour: OpenAiFlavour, body: &Value, messages: &Value) -> Map<String, Value> {
    let mut out = Map::new();
    out.insert("messages".into(), messages.clone());
    copy_if_present(&mut out, "model", body);

    // Field order matches each `send*Request`'s literal.
    let fields: &[&str] = match flavour {
        OpenAiFlavour::DeepSeek => &[
            "temperature",
            "max_tokens",
            "stream",
            "presence_penalty",
            "frequency_penalty",
            "top_p",
            "stop",
            "seed",
        ],
        OpenAiFlavour::Xai => &[
            "temperature",
            "max_tokens",
            "max_completion_tokens",
            "stream",
            "presence_penalty",
            "frequency_penalty",
            "top_p",
            "seed",
            "n",
        ],
        OpenAiFlavour::AimlApi => &[
            "temperature",
            "max_tokens",
            "stream",
            "presence_penalty",
            "frequency_penalty",
            "top_p",
            "seed",
            "n",
        ],
        OpenAiFlavour::ElectronHub => &[
            "temperature",
            "max_tokens",
            "stream",
            "presence_penalty",
            "frequency_penalty",
            "top_p",
            "top_k",
            "logit_bias",
            "seed",
        ],
        OpenAiFlavour::Chutes => &[
            "temperature",
            "max_tokens",
            "stream",
            "presence_penalty",
            "frequency_penalty",
            "repetition_penalty",
            "min_p",
            "top_p",
            "top_k",
            "seed",
            "stop",
            "reasoning_effort",
            "logit_bias",
        ],
        OpenAiFlavour::MiniMax => &["temperature", "max_tokens", "stream", "top_p", "stop"],
        OpenAiFlavour::Mistral => &[
            "temperature",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "max_tokens",
            "stream",
            "safe_prompt",
        ],
        OpenAiFlavour::AzureOpenAi => &[],
    };
    for field in fields {
        copy_if_present(&mut out, field, body);
    }
    // DeepSeek always sends an explicit `thinking` flag.
    if matches!(flavour, OpenAiFlavour::DeepSeek) {
        out.insert(
            "thinking".into(),
            json!({ "type": if get_bool(body, "include_reasoning") { "enabled" } else { "disabled" } }),
        );
        if get_bool(body, "include_reasoning") {
            if let Some(effort) = get_str(body, "reasoning_effort").filter(|s| !s.is_empty()) {
                out.insert("reasoning_effort".into(), json!(effort));
            }
        }
    }
    // Mistral calls the seed `random_seed`, and `-1` means "unset".
    if matches!(flavour, OpenAiFlavour::Mistral) {
        if let Some(seed) = get_f64(body, "seed").filter(|v| *v != -1.0) {
            out.insert("random_seed".into(), num(seed));
        }
        if body
            .get("stop")
            .and_then(|s| s.as_array())
            .is_some_and(|s| !s.is_empty())
        {
            out.insert("stop".into(), body.get("stop").cloned().unwrap());
        }
    }
    out
}

fn copy_tools(out: &mut Map<String, Value>, body: &Value) {
    if let Some(tools) = body.get("tools").and_then(|t| t.as_array()) {
        if !tools.is_empty() {
            out.insert("tools".into(), Value::Array(tools.clone()));
            out.insert(
                "tool_choice".into(),
                body.get("tool_choice").cloned().unwrap_or(Value::Null),
            );
        }
    }
}

fn copy_stop(out: &mut Map<String, Value>, body: &Value) {
    if let Some(stop) = body.get("stop").and_then(|s| s.as_array()) {
        if !stop.is_empty() {
            out.insert("stop".into(), Value::Array(stop.clone()));
        }
    }
}

/// `{type:"json_schema", json_schema:{name, [description], schema, strict}}`
fn insert_json_schema_verbose(out: &mut Map<String, Value>, body: &Value, with_description: bool) {
    let Some(schema) = body.get("json_schema").filter(|v| v.is_object()) else {
        return;
    };
    let mut inner = Map::new();
    inner.insert(
        "name".into(),
        schema.get("name").cloned().unwrap_or(Value::Null),
    );
    if with_description {
        if let Some(description) = schema.get("description").filter(|v| !v.is_null()) {
            inner.insert("description".into(), description.clone());
        }
    }
    inner.insert(
        "schema".into(),
        schema.get("value").cloned().unwrap_or(Value::Null),
    );
    inner.insert(
        "strict".into(),
        json!(schema
            .get("strict")
            .and_then(|v| v.as_bool())
            .unwrap_or(true)),
    );
    out.insert(
        "response_format".into(),
        json!({ "type": "json_schema", "json_schema": inner }),
    );
}

// ---------------------------------------------------------------------------
// Claude (Anthropic Messages API)
// ---------------------------------------------------------------------------

/// The per-model capability gates `sendClaudeRequest` computes from regexes.
struct ClaudeFlags {
    use_thinking: bool,
    use_web_search: bool,
    is_limited_sampling: bool,
    use_verbosity: bool,
    no_prefill: bool,
    is_adaptive: bool,
    no_sampling: bool,
    use_native_json: bool,
    is_claude5: bool,
    is_fable: bool,
}

/// Mirror of the model regexes at the top of `sendClaudeRequest`. All of them
/// are anchored `^claude-...`, except `/claude-fable/` which is unanchored so
/// prefixed ids through proxies still match.
fn claude_model_flags(model: &str, adaptive_enabled: bool) -> ClaudeFlags {
    let is_fable = model.contains("claude-fable");
    let use_native_json = model.contains("claude-fable-5-1") || model.contains("claude-opus-5-5");
    let is_claude5 = model.contains("claude-opus-5") || model.contains("claude-sonnet-5");
    let starts = |prefixes: &[&str]| prefixes.iter().any(|p| model.starts_with(p));

    ClaudeFlags {
        use_thinking: is_fable
            || is_claude5
            || starts(&[
                "claude-3-7",
                "claude-opus-4",
                "claude-sonnet-4",
                "claude-haiku-4-5",
                "claude-opus-4-5",
                "claude-opus-4-6",
                "claude-sonnet-4-6",
                "claude-opus-4-7",
            ]),
        use_web_search: is_fable
            || is_claude5
            || starts(&[
                "claude-3-5",
                "claude-3-7",
                "claude-opus-4",
                "claude-sonnet-4",
                "claude-haiku-4-5",
                "claude-opus-4-5",
                "claude-opus-4-6",
                "claude-sonnet-4-6",
                "claude-opus-4-7",
            ]),
        is_limited_sampling: starts(&[
            "claude-opus-4-1",
            "claude-sonnet-4-5",
            "claude-haiku-4-5",
            "claude-opus-4-5",
            "claude-opus-4-6",
            "claude-sonnet-4-6",
        ]),
        use_verbosity: is_fable
            || is_claude5
            || starts(&[
                "claude-opus-4-5",
                "claude-opus-4-6",
                "claude-sonnet-4-6",
                "claude-opus-4-7",
                "claude-opus-4-8",
            ]),
        no_prefill: is_fable
            || is_claude5
            || starts(&[
                "claude-opus-4-6",
                "claude-sonnet-4-6",
                "claude-opus-4-7",
                "claude-opus-4-8",
            ]),
        is_adaptive: is_fable
            || is_claude5
            || starts(&["claude-opus-4-7", "claude-opus-4-8"])
            || (adaptive_enabled && starts(&["claude-opus-4-6", "claude-sonnet-4-6"])),
        no_sampling: is_fable || is_claude5 || starts(&["claude-opus-4-7", "claude-opus-4-8"]),
        use_native_json,
        is_claude5,
        is_fable,
    }
}

/// Build the Anthropic `/messages` request (mirror of `sendClaudeRequest`).
pub fn build_claude(
    body: &Value,
    secrets_doc: &Value,
    cfg: &crate::config::StConfig,
    names: &PromptNames,
    placeholder: &str,
) -> Result<ProviderRequest, PrepareError> {
    let secret_id = get_str(body, "secret_id");
    let (api_url, api_key) =
        resolve_endpoint(body, API_CLAUDE, keys::CLAUDE, secrets_doc, secret_id);
    if api_key.is_empty() {
        return Err(PrepareError::BadRequest("missing api key"));
    }

    let model = get_str(body, "model").unwrap_or("").to_string();
    let stream = get_bool(body, "stream");
    let use_tools = body
        .get("tools")
        .and_then(|t| t.as_array())
        .is_some_and(|t| !t.is_empty());
    let use_system_prompt = get_bool(body, "use_sysprompt");
    let adaptive_enabled =
        crate::chat_completions::config_bool(cfg, "claude.enableAdaptiveThinking", true);
    let flags = claude_model_flags(&model, adaptive_enabled);

    let prefill = get_str(body, "assistant_prefill").map(str::to_string);
    let (messages, system_prompt) = converters::convert_claude_messages(
        &body.get("messages").cloned().unwrap_or(json!([])),
        prefill.as_deref(),
        use_system_prompt,
        use_tools,
        names,
        placeholder,
    );

    let ttl = crate::chat_completions::claude_cache_ttl(cfg);
    let enable_system_cache =
        crate::chat_completions::config_bool(cfg, "claude.enableSystemPromptCache", false);
    let caching_depth = crate::chat_completions::claude_caching_at_depth(cfg);

    let stop_sequences: Vec<Value> = body
        .get("stop")
        .and_then(|s| s.as_array())
        .map(|s| s.to_vec())
        .unwrap_or_default();

    let mut beta_headers: Vec<&str> = vec!["output-128k-2025-02-19", "context-1m-2025-08-07"];

    let mut request_body = Map::new();
    request_body.insert("system".into(), json!([]));
    request_body.insert("messages".into(), messages);
    copy_if_present(&mut request_body, "model", body);
    copy_if_present(&mut request_body, "max_tokens", body);
    request_body.insert("stop_sequences".into(), Value::Array(stop_sequences));
    copy_if_present(&mut request_body, "temperature", body);
    copy_if_present(&mut request_body, "top_p", body);
    copy_if_present(&mut request_body, "top_k", body);
    request_body.insert("stream".into(), json!(stream));

    if use_system_prompt {
        let mut system = system_prompt;
        if enable_system_cache {
            if let Some(last) = system
                .as_array_mut()
                .and_then(|arr| arr.last_mut())
                .and_then(|v| v.as_object_mut())
            {
                last.insert("cache_control".into(), cache_control(ttl.as_deref()));
            }
        }
        request_body.insert("system".into(), system);
    } else {
        // Anthropic rejects an empty `system`; upstream deletes the key.
        request_body.remove("system");
    }

    if use_tools {
        beta_headers.push("tools-2024-05-16");
        request_body.insert(
            "tool_choice".into(),
            json!({ "type": get_str(body, "tool_choice").unwrap_or("") }),
        );
        let mut tools: Vec<Value> = body
            .get("tools")
            .and_then(|t| t.as_array())
            .map(|tools| {
                tools
                    .iter()
                    .filter(|t| t.get("type").and_then(|v| v.as_str()) == Some("function"))
                    .filter_map(|t| t.get("function"))
                    .map(|f| {
                        json!({
                            "name": f.get("name").cloned().unwrap_or(Value::Null),
                            "description": f.get("description").cloned().unwrap_or(Value::Null),
                            "input_schema": flatten_schema(
                                f.get("parameters").unwrap_or(&Value::Null),
                                "claude",
                            ),
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();
        if enable_system_cache {
            // The cache breakpoint rides on the last tool block, so the whole
            // tool list is cached with the system prompt.
            if let Some(last) = tools.last_mut().and_then(|t| t.as_object_mut()) {
                last.insert("cache_control".into(), cache_control(ttl.as_deref()));
            }
        }
        request_body.insert("tools".into(), Value::Array(tools));
    }

    // JSON schema: native output config on the newest models, forced tool
    // everywhere else.
    if let Some(schema) = body.get("json_schema").filter(|v| v.is_object()) {
        if flags.use_native_json {
            request_body.insert(
                "output_config".into(),
                json!({
                    "format": {
                        "type": "json_schema",
                        "schema": schema.get("value").cloned().unwrap_or(Value::Null),
                    }
                }),
            );
        } else {
            let json_tool = json!({
                "name": schema.get("name").cloned().unwrap_or(Value::Null),
                "description": schema
                    .get("description")
                    .and_then(|v| v.as_str())
                    .filter(|s| !s.is_empty())
                    .unwrap_or("Well-formed JSON object"),
                "input_schema": schema.get("value").cloned().unwrap_or(Value::Null),
            });
            let mut tools = request_body
                .get("tools")
                .and_then(|t| t.as_array())
                .cloned()
                .unwrap_or_default();
            tools.push(json_tool);
            request_body.insert("tools".into(), Value::Array(tools));
            request_body.insert(
                "tool_choice".into(),
                json!({ "type": "tool", "name": schema.get("name").cloned().unwrap_or(Value::Null) }),
            );
        }
    }

    if flags.use_web_search && get_bool(body, "enable_web_search") {
        let mut tools = vec![json!({ "type": "web_search_20250305", "name": "web_search" })];
        if let Some(existing) = request_body.get("tools").and_then(|t| t.as_array()) {
            tools.extend(existing.iter().cloned());
        }
        request_body.insert("tools".into(), Value::Array(tools));
    }

    if caching_depth != -1 {
        let mut msgs = request_body
            .get("messages")
            .cloned()
            .unwrap_or_else(|| json!([]));
        prompt::caching_at_depth_for_open_router_claude(&mut msgs, caching_depth, ttl.as_deref());
        request_body.insert("messages".into(), msgs);
    }

    if enable_system_cache || caching_depth != -1 {
        beta_headers.push("prompt-caching-2024-07-31");
        beta_headers.push("extended-cache-ttl-2025-04-11");
    }

    // Limited-sampling models accept exactly one of temperature/top_p.
    if flags.is_limited_sampling {
        if get_f64(body, "top_p").unwrap_or(1.0) < 1.0 {
            request_body.remove("temperature");
        } else {
            request_body.remove("top_p");
        }
    }
    if flags.no_sampling {
        request_body.remove("temperature");
        request_body.remove("top_p");
        request_body.remove("top_k");
    }

    let reasoning_effort = get_str(body, "reasoning_effort");
    let include_reasoning = get_bool(body, "include_reasoning");
    let max_tokens = get_f64(body, "max_tokens").unwrap_or(0.0);
    let budget =
        converters::claude_budget_tokens(max_tokens, reasoning_effort, stream, flags.is_adaptive);

    let mut fix_thinking_prefill = false;
    // `useThinking` gates the whole block: a model without thinking support
    // must not receive a `thinking` field at all, even with a budget computed.
    let budget = if flags.use_thinking { budget } else { None };
    match budget {
        // Adaptive thinking: an effort level string instead of a token budget.
        Some(Value::String(effort)) => {
            fix_thinking_prefill = true;
            let mut thinking = Map::new();
            thinking.insert("type".into(), json!("adaptive"));
            if flags.no_sampling && include_reasoning {
                thinking.insert("display".into(), json!("summarized"));
            }
            request_body.insert("thinking".into(), Value::Object(thinking));
            let mut output_config = request_body
                .get("output_config")
                .and_then(|o| o.as_object())
                .cloned()
                .unwrap_or_default();
            output_config.insert("effort".into(), Value::String(effort));
            request_body.insert("output_config".into(), Value::Object(output_config));
            // Anthropic rejects `top_k` in adaptive mode.
            request_body.remove("top_k");
        }
        // Traditional thinking: a numeric budget, which also forbids sampling.
        Some(Value::Number(budget)) => {
            fix_thinking_prefill = true;
            const MIN_THINK_TOKENS: f64 = 1024.0;
            if max_tokens <= MIN_THINK_TOKENS {
                request_body.insert("max_tokens".into(), num(max_tokens + MIN_THINK_TOKENS));
            }
            request_body.insert(
                "thinking".into(),
                json!({ "type": "enabled", "budget_tokens": num(budget.as_f64().unwrap_or(0.0)) }),
            );
            request_body.remove("temperature");
            request_body.remove("top_p");
            request_body.remove("top_k");
        }
        _ => {
            // Fable / Claude 5 think by default; readable summaries still need
            // an explicit display request.
            if (flags.is_fable || flags.is_claude5)
                && reasoning_effort == Some("auto")
                && include_reasoning
            {
                fix_thinking_prefill = true;
                request_body.insert(
                    "thinking".into(),
                    json!({ "type": "adaptive", "display": "summarized" }),
                );
            }
        }
    }

    // A trailing assistant turn cannot be a prefill once thinking is on.
    if fix_thinking_prefill || flags.no_prefill {
        if let Some(last) = request_body
            .get_mut("messages")
            .and_then(|m| m.as_array_mut())
            .and_then(|m| m.last_mut())
            .and_then(|m| m.as_object_mut())
        {
            if last.get("role").and_then(|r| r.as_str()) == Some("assistant") {
                last.insert("role".into(), json!("user"));
            }
        }
    }

    if flags.use_verbosity {
        if let Some(verbosity) = get_str(body, "verbosity").filter(|s| !s.is_empty()) {
            let already_set = request_body
                .get("output_config")
                .and_then(|o| o.get("effort"))
                .is_some_and(|e| !e.is_null());
            if !already_set {
                beta_headers.push("effort-2025-11-24");
                let mut output_config = request_body
                    .get("output_config")
                    .and_then(|o| o.as_object())
                    .cloned()
                    .unwrap_or_default();
                output_config.insert("effort".into(), json!(verbosity));
                request_body.insert("output_config".into(), Value::Object(output_config));
            }
        }
    }

    let mut headers = vec![
        ("Content-Type".to_string(), "application/json".to_string()),
        ("anthropic-version".to_string(), "2023-06-01".to_string()),
        ("x-api-key".to_string(), api_key),
    ];
    if !beta_headers.is_empty() {
        headers.push(("anthropic-beta".to_string(), beta_headers.join(",")));
    }

    Ok(ProviderRequest {
        prepared: PreparedRequest {
            url: format!("{}/messages", api_url.trim_end_matches('/')),
            headers,
            body: Value::Object(request_body),
            stream,
            transform: ResponseTransform::Claude,
        },
        transform: ResponseTransform::Claude,
    })
}

/// `{type:"ephemeral"[ , ttl]}` for the prompt-caching beta.
fn cache_control(ttl: Option<&str>) -> Value {
    match ttl {
        Some(ttl) => json!({ "type": "ephemeral", "ttl": ttl }),
        None => json!({ "type": "ephemeral" }),
    }
}

// ---------------------------------------------------------------------------
// Gemini (AI Studio + Vertex AI)
// ---------------------------------------------------------------------------

/// `GEMINI_SAFETY` + `VERTEX_SAFETY` when the request goes to Vertex.
fn gemini_safety_for(vertex: bool) -> Value {
    let mut all = gemini_safety().as_array().cloned().unwrap_or_default();
    if vertex {
        all.extend(vertex_safety().as_array().cloned().unwrap_or_default());
    }
    Value::Array(all)
}

/// `isThinkingConfigModel` from `getGeminiBody`.
fn gemini_is_thinking_model(model: &str) -> bool {
    let flash_or_pro = (model.starts_with("gemini-2.5-flash")
        || model.starts_with("gemini-2.5-pro"))
        && !model.ends_with("-image")
        && !model.ends_with("-image-preview");
    flash_or_pro || model.starts_with("gemini-3")
}

/// `noSamplingModel` — Gemini 3.6+ flash / 3.5 flash-lite reject sampling.
fn gemini_no_sampling(model: &str) -> bool {
    model.contains("gemini-3.6-flash")
        || model.contains("gemini-3.7-flash")
        || model.contains("gemini-3.8-flash")
        || model.contains("gemini-3.9-flash")
        || model.contains("gemini-3.5-flash-lite")
}

const GEMINI_IMAGE_MODELS: &[&str] = &[
    "gemini-2.0-flash-exp",
    "gemini-2.0-flash-exp-image-generation",
    "gemini-2.0-flash-preview-image-generation",
    "gemini-2.5-flash-image-preview",
    "gemini-2.5-flash-image",
    "gemini-3-pro-image",
    "gemini-3.1-flash-image",
];

const GEMINI_NO_SEARCH_MODELS: &[&str] = &[
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash-lite-001",
    "gemini-2.0-flash-lite-preview-02-05",
    "gemini-robotics-er-1.5-preview",
];

/// Build the `{contents, safetySettings, generationConfig, ...}` body shared by
/// AI Studio and Vertex (`getGeminiBody`).
fn gemini_request_body(
    body: &Value,
    model: &str,
    vertex: bool,
    names: &PromptNames,
    cfg: &crate::config::StConfig,
) -> Value {
    let request_images = get_bool(body, "request_images");
    let include_reasoning = get_bool(body, "include_reasoning");
    let reasoning_effort = get_str(body, "reasoning_effort").unwrap_or("");
    let aspect_ratio = get_str(body, "request_image_aspect_ratio").unwrap_or("");
    let image_size = get_str(body, "request_image_resolution").unwrap_or("");
    let is_gemma3 = model.contains("gemma-3");
    let is_learnlm = model.contains("learnlm");

    // `responseMimeType ?? (json_schema ? 'application/json' : undefined)`
    let response_mime = get_str(body, "responseMimeType")
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .or_else(|| {
            body.get("json_schema")
                .is_some_and(|v| !v.is_null())
                .then(|| "application/json".to_string())
        });
    let response_schema = body
        .get("responseSchema")
        .cloned()
        .filter(|v| !v.is_null())
        .or_else(|| {
            body.get("json_schema")
                .and_then(|s| s.get("value"))
                .cloned()
        });

    let no_sampling = gemini_no_sampling(model);

    let mut generation_config = Map::new();
    if let Some(stop) = body.get("stop").and_then(|s| s.as_array()) {
        if !stop.is_empty() {
            generation_config.insert("stopSequences".into(), Value::Array(stop.clone()));
        }
    }
    if !no_sampling {
        generation_config.insert("candidateCount".into(), json!(1));
    }
    copy_if_present(&mut generation_config, "max_tokens", body);
    if !no_sampling {
        copy_if_present(&mut generation_config, "temperature", body);
        copy_if_present(&mut generation_config, "top_p", body);
        // `top_k || undefined` — a literal 0 is dropped.
        if get_f64(body, "top_k").is_some_and(|v| v != 0.0) {
            generation_config.insert("topK".into(), body.get("top_k").cloned().unwrap());
        }
    }
    if let Some(mime) = response_mime {
        generation_config.insert("responseMimeType".into(), json!(mime));
    }
    if let Some(schema) = response_schema {
        generation_config.insert("responseSchema".into(), schema);
    }
    copy_if_present(&mut generation_config, "seed", body);
    // The wire names Gemini wants are camelCase; `max_tokens` becomes
    // `maxOutputTokens`.
    if let Some(value) = generation_config.remove("max_tokens") {
        generation_config.insert("maxOutputTokens".into(), value);
    }
    // `top_p`/`top_k` were copied under their OpenAI names above.
    for (gemini_key, openai_key) in [("topP", "top_p"), ("topK", "top_k")] {
        if let Some(value) = generation_config.remove(openai_key) {
            generation_config.insert(gemini_key.into(), value);
        }
    }

    let enable_image_modality = request_images && GEMINI_IMAGE_MODELS.contains(&model);
    if enable_image_modality {
        generation_config.insert("responseModalities".into(), json!(["text", "image"]));
        if !aspect_ratio.is_empty() || !image_size.is_empty() {
            let mut image_config = Map::new();
            if !image_size.is_empty() && model.starts_with("gemini-3") {
                image_config.insert("imageSize".into(), json!(image_size));
            }
            if !aspect_ratio.is_empty() {
                image_config.insert("aspectRatio".into(), json!(aspect_ratio));
            }
            generation_config.insert("imageConfig".into(), Value::Object(image_config));
        }
    }

    let use_system_prompt = !enable_image_modality && !is_gemma3 && get_bool(body, "use_sysprompt");
    let (contents, system_instruction) = converters::convert_google_prompt(
        &body.get("messages").cloned().unwrap_or(json!([])),
        model,
        use_system_prompt,
        names,
        crate::chat_completions::config_bool(cfg, "gemini.thoughtSignatures", true),
    );

    let mut tools: Vec<Value> = Vec::new();
    if let Some(list) = body.get("tools").and_then(|t| t.as_array()) {
        if !list.is_empty() && !enable_image_modality && !is_gemma3 {
            let mut function_declarations: Vec<Value> = Vec::new();
            let mut custom_tools: Vec<Value> = Vec::new();
            for tool in list {
                let kind = tool.get("type").and_then(|t| t.as_str()).unwrap_or("");
                if kind == "function" {
                    if let Some(function) = tool.get("function") {
                        let mut function = function.clone();
                        if let Some(params) = function
                            .get_mut("parameters")
                            .and_then(|p| p.as_object_mut())
                        {
                            params.remove("$schema");
                        }
                        // An empty `properties` object is rejected.
                        let empty_properties = function
                            .get("parameters")
                            .and_then(|p| p.get("properties"))
                            .and_then(|p| p.as_object())
                            .is_some_and(|p| p.is_empty());
                        if empty_properties {
                            if let Some(obj) = function.as_object_mut() {
                                obj.remove("parameters");
                            }
                        }
                        function_declarations.push(function);
                    }
                } else if let Some(custom) = tool.get(kind) {
                    let mut entry = Map::new();
                    entry.insert(kind.to_string(), custom.clone());
                    custom_tools.push(Value::Object(entry));
                }
            }
            if !function_declarations.is_empty() {
                tools.push(json!({ "function_declarations": function_declarations }));
            } else if !custom_tools.is_empty() {
                // Custom tools only apply when there is no function calling.
                tools.extend(custom_tools);
            }
        }
    }

    if get_bool(body, "enable_web_search")
        && !enable_image_modality
        && !is_gemma3
        && !is_learnlm
        && !GEMINI_NO_SEARCH_MODELS.contains(&model)
    {
        // Function calling and google_search cannot be combined.
        let has_functions = tools
            .iter()
            .any(|t| t.get("function_declarations").is_some());
        if !has_functions {
            tools.push(json!({ "google_search": {} }));
        }
    }

    if gemini_is_thinking_model(model) {
        let mut thinking_config = Map::new();
        thinking_config.insert("includeThoughts".into(), json!(include_reasoning));
        let max_output = get_f64(body, "max_tokens").unwrap_or(0.0);
        let budget = converters::google_budget_tokens(max_output, Some(reasoning_effort), model);
        match &budget {
            Some(Value::Number(number)) => {
                if let Some(budget) = number.as_f64() {
                    thinking_config.insert("thinkingBudget".into(), num(budget));
                }
            }
            Some(Value::String(level)) if !level.is_empty() => {
                thinking_config.insert("thinkingLevel".into(), budget.clone().unwrap());
            }
            _ => {}
        }
        // Vertex rejects `thinkingBudget: 0` together with includeThoughts.
        if vertex && budget.as_ref() == Some(&json!(0)) && include_reasoning {
            thinking_config.insert("includeThoughts".into(), json!(false));
        }
        generation_config.insert("thinkingConfig".into(), Value::Object(thinking_config));
    }

    let mut out = Map::new();
    out.insert("contents".into(), contents);
    out.insert("safetySettings".into(), gemini_safety_for(vertex));
    out.insert("generationConfig".into(), Value::Object(generation_config));

    let has_system = system_instruction
        .get("parts")
        .and_then(|p| p.as_array())
        .is_some_and(|p| !p.is_empty());
    if use_system_prompt && has_system {
        out.insert("systemInstruction".into(), system_instruction);
    }

    if !tools.is_empty() {
        out.insert("tools".into(), Value::Array(tools));
        // Translate OpenAI's `tool_choice` into Gemini's functionCallingConfig.
        let function_calling_config = match body.get("tool_choice") {
            Some(Value::String(mode)) => match mode.as_str() {
                "none" => Some(json!({ "mode": "NONE" })),
                "required" => Some(json!({ "mode": "ANY" })),
                "auto" => Some(json!({ "mode": "AUTO" })),
                _ => None,
            },
            Some(choice) => choice
                .pointer("/function/name")
                .and_then(|n| n.as_str())
                .map(|name| json!({ "mode": "ANY", "allowedFunctionNames": [name] })),
            None => None,
        };
        if let Some(config) = function_calling_config {
            out.insert(
                "toolConfig".into(),
                json!({ "functionCallingConfig": config }),
            );
        }
    }

    Value::Object(out)
}

/// Vertex authentication, mirroring `getVertexAIAuth`.
struct VertexAuth {
    /// `express` (API key), `full` (service account) or `proxy`.
    kind: &'static str,
    header: String,
    api_key: Option<String>,
}

/// Build a Gemini request for AI Studio (`makersuite`) or Vertex AI.
pub async fn build_gemini(
    body: &Value,
    secrets_doc: &Value,
    cfg: &crate::config::StConfig,
    names: &PromptNames,
    vertex: bool,
) -> Result<ProviderRequest, PrepareError> {
    let secret_id = get_str(body, "secret_id");
    let model = get_str(body, "model").unwrap_or("").to_string();
    let stream = get_bool(body, "stream");
    let response_type = if stream {
        "streamGenerateContent"
    } else {
        "generateContent"
    };

    let request_body = gemini_request_body(body, &model, vertex, names, cfg);
    let api_version = crate::chat_completions::config_str(cfg, "gemini.apiVersion")
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| "v1beta".to_string());
    let reverse_proxy = get_str(body, "reverse_proxy").filter(|s| !s.is_empty());
    let alt = if stream { "?alt=sse" } else { "" };
    let encoded_model = crate::chat_completions::encode_uri_component(&model);

    let json_headers = |extra: Option<(String, String)>| {
        let mut headers = vec![("Content-Type".to_string(), "application/json".to_string())];
        if let Some((k, v)) = extra {
            headers.push((k, v));
        }
        headers
    };

    let (url, headers) = if vertex {
        let auth = vertex_auth(body, secrets_doc, secret_id).await?;
        let region = get_str(body, "vertexai_region").unwrap_or("us-central1");
        match auth.kind {
            "express" => {
                let key = auth.api_key.clone().unwrap_or_default();
                let encoded_key = crate::chat_completions::encode_uri_component(&key);
                let project_id = get_str(body, "vertexai_express_project_id")
                    .filter(|s| !s.is_empty())
                    .map(str::to_string);
                let base = if region == "global" {
                    "https://aiplatform.googleapis.com".to_string()
                } else {
                    format!("https://{region}-aiplatform.googleapis.com")
                };
                // With a project id the URL is project-scoped and always uses
                // the global host, exactly like upstream.
                let url = match project_id {
                    Some(project_id) => format!(
                        "https://aiplatform.googleapis.com/v1/projects/{}/locations/{}/publishers/google/models/{encoded_model}:{response_type}?key={encoded_key}{}",
                        crate::chat_completions::encode_uri_component(&project_id),
                        crate::chat_completions::encode_uri_component(region),
                        if stream { "&alt=sse" } else { "" },
                    ),
                    None => format!(
                        "{base}/v1/publishers/google/models/{encoded_model}:{response_type}?key={encoded_key}{alt}"
                    ),
                };
                (url, json_headers(None))
            }
            "full" => {
                let project_id = vertex_project_id(secrets_doc, secret_id)
                    .ok_or(PrepareError::BadRequest("missing api key"))?;
                let base = if region == "global" {
                    "https://aiplatform.googleapis.com".to_string()
                } else {
                    format!("https://{region}-aiplatform.googleapis.com")
                };
                (
                    format!(
                        "{base}/v1/projects/{project_id}/locations/{region}/publishers/google/models/{encoded_model}:{response_type}{alt}"
                    ),
                    json_headers(Some(("Authorization".to_string(), auth.header))),
                )
            }
            _ => {
                let base = normalize_url(reverse_proxy.unwrap_or(API_VERTEX_AI));
                (
                    format!(
                        "{}/v1/publishers/google/models/{encoded_model}:{response_type}{alt}",
                        base.trim_end_matches('/')
                    ),
                    json_headers(Some(("Authorization".to_string(), auth.header))),
                )
            }
        }
    } else {
        let base = normalize_url(reverse_proxy.unwrap_or(API_MAKERSUITE));
        let api_key = match reverse_proxy {
            Some(_) => get_str(body, "proxy_password").unwrap_or("").to_string(),
            None => secrets::read_secret(secrets_doc, keys::MAKERSUITE, secret_id),
        };
        if api_key.is_empty() && reverse_proxy.is_none() {
            return Err(PrepareError::BadRequest("missing api key"));
        }
        (
            format!(
                "{}/{api_version}/models/{encoded_model}:{response_type}?key={}{}",
                base.trim_end_matches('/'),
                crate::chat_completions::encode_uri_component(&api_key),
                if stream { "&alt=sse" } else { "" }
            ),
            json_headers(None),
        )
    };

    Ok(ProviderRequest {
        prepared: PreparedRequest {
            url,
            headers,
            body: request_body,
            stream,
            transform: ResponseTransform::Gemini,
        },
        transform: ResponseTransform::Gemini,
    })
}

/// Mirror of `getVertexAIAuth`.
async fn vertex_auth(
    body: &Value,
    secrets_doc: &Value,
    secret_id: Option<&str>,
) -> Result<VertexAuth, PrepareError> {
    // A reverse proxy short-circuits everything: its password is the bearer.
    if get_str(body, "reverse_proxy").is_some_and(|s| !s.is_empty()) {
        let password = get_str(body, "proxy_password").unwrap_or("");
        return Ok(VertexAuth {
            kind: "proxy",
            header: format!("Bearer {password}"),
            api_key: None,
        });
    }

    let mode = get_str(body, "vertexai_auth_mode").unwrap_or("express");
    if mode == "express" {
        let api_key = secrets::read_secret(secrets_doc, keys::VERTEXAI, secret_id);
        if api_key.is_empty() {
            return Err(PrepareError::Message(
                "API key is required for Vertex AI Express mode".to_string(),
            ));
        }
        return Ok(VertexAuth {
            kind: "express",
            header: format!("Bearer {api_key}"),
            api_key: Some(api_key),
        });
    }

    if mode == "full" {
        let service_account =
            secrets::read_secret(secrets_doc, keys::VERTEXAI_SERVICE_ACCOUNT, secret_id);
        if service_account.is_empty() {
            return Err(PrepareError::Message(
                "Service Account JSON is required for Vertex AI Full mode".to_string(),
            ));
        }
        let token = vertex_access_token(&service_account)
            .await
            .map_err(PrepareError::Message)?;
        return Ok(VertexAuth {
            kind: "full",
            header: format!("Bearer {token}"),
            api_key: None,
        });
    }

    Err(PrepareError::Message(format!(
        "Unsupported Vertex AI authentication mode: {mode}"
    )))
}

/// Project id from the service account JSON (Full-mode URLs).
fn vertex_project_id(secrets_doc: &Value, secret_id: Option<&str>) -> Option<String> {
    let raw = secrets::read_secret(secrets_doc, keys::VERTEXAI_SERVICE_ACCOUNT, secret_id);
    let parsed: Value = serde_json::from_str(&raw).ok()?;
    parsed
        .get("project_id")
        .and_then(|p| p.as_str())
        .filter(|s| !s.is_empty())
        .map(str::to_string)
}

/// Sign a Google service-account JWT and exchange it for an access token.
///
/// This is the only provider setup that needs the network: the JWT is signed
/// locally with `rsa`, then Google's token endpoint validates it. Node did the
/// same with `crypto.createSign` inside `getVertexAIAuth`.
async fn vertex_access_token(service_account_json: &str) -> Result<String, String> {
    use base64::Engine;
    let account: Value = serde_json::from_str(service_account_json)
        .map_err(|e| format!("Invalid service account JSON: {e}"))?;
    let client_email = account
        .get("client_email")
        .and_then(|v| v.as_str())
        .ok_or("client_email missing from service account JSON")?
        .to_string();
    let private_key = account
        .get("private_key")
        .and_then(|v| v.as_str())
        .ok_or("private_key missing from service account JSON")?
        .to_string();

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let header = json!({ "alg": "RS256", "typ": "JWT" });
    let payload = json!({
        "iss": client_email,
        "scope": "https://www.googleapis.com/auth/cloud-platform",
        "aud": API_VERTEX_OAUTH,
        "iat": now,
        "exp": now + 3600,
    });
    let b64 = base64::engine::general_purpose::URL_SAFE_NO_PAD;
    let signing_input = format!(
        "{}.{}",
        b64.encode(header.to_string()),
        b64.encode(payload.to_string())
    );

    // `crypto.createSign('RSA-SHA256')` == PKCS#1 v1.5 with SHA-256.
    use rsa::pkcs1::DecodeRsaPrivateKey;
    use rsa::pkcs8::DecodePrivateKey;
    use rsa::signature::{SignatureEncoding, Signer};
    let key = rsa::RsaPrivateKey::from_pkcs8_pem(&private_key)
        .or_else(|_| rsa::RsaPrivateKey::from_pkcs1_pem(&private_key))
        .map_err(|e| format!("Could not parse the service account private key: {e}"))?;
    // `SigningKey::new` wants `AssociatedOid`, which the `sha2` feature of
    // `rsa` provides; using the explicit prefix constructor avoids the bound.
    let signing_key = rsa::pkcs1v15::SigningKey::<sha2::Sha256>::new_unprefixed(key);
    let signature = signing_key.sign(signing_input.as_bytes());
    let jwt = format!("{signing_input}.{}", b64.encode(signature.to_bytes()));

    let form =
        format!("grant_type=urn%3Aietf%3Aparams%3Aoauth%3Agrant-type%3Ajwt-bearer&assertion={jwt}");
    let response = crate::forward::post_form(API_VERTEX_OAUTH, &form)
        .await
        .map_err(|e| format!("Failed to get access token: {e}"))?;
    let parsed: Value =
        serde_json::from_str(&response).map_err(|e| format!("Unexpected token response: {e}"))?;
    parsed
        .get("access_token")
        .and_then(|t| t.as_str())
        .map(str::to_string)
        .ok_or_else(|| format!("Failed to get access token: {response}"))
}

// ---------------------------------------------------------------------------
// AI21 + Cohere
// ---------------------------------------------------------------------------

/// Mirror of `sendAI21Request`.
pub fn build_ai21(
    body: &Value,
    secrets_doc: &Value,
    names: &PromptNames,
    placeholder: &str,
) -> Result<ProviderRequest, PrepareError> {
    let secret_id = get_str(body, "secret_id");
    let api_key = secrets::read_secret(secrets_doc, keys::AI21, secret_id);
    if api_key.is_empty() {
        return Err(PrepareError::BadRequest("missing api key"));
    }
    let stream = get_bool(body, "stream");

    let mut body_params = Map::new();
    let mut messages = body.get("messages").cloned().unwrap_or(json!([]));
    // JSON schema is emulated with a trailing user message.
    if let Some(schema) = body.get("json_schema").filter(|v| v.is_object()) {
        body_params.insert("response_format".into(), json!({ "type": "json_object" }));
        let hint = format!(
            "JSON schema for the response:\n{}",
            serde_json::to_string_pretty(schema.get("value").unwrap_or(&Value::Null))
                .unwrap_or_default()
        );
        if let Some(arr) = messages.as_array_mut() {
            arr.push(json!({ "role": "user", "content": hint }));
        }
    }
    let converted = converters::convert_ai21_messages(&messages, names, placeholder);

    let mut request_body = Map::new();
    request_body.insert("messages".into(), converted);
    copy_if_present(&mut request_body, "model", body);
    copy_if_present(&mut request_body, "max_tokens", body);
    copy_if_present(&mut request_body, "temperature", body);
    copy_if_present(&mut request_body, "top_p", body);
    copy_if_present(&mut request_body, "stop", body);
    request_body.insert("stream".into(), json!(stream));
    if let Some(tools) = body.get("tools").filter(|t| !t.is_null()) {
        request_body.insert("tools".into(), tools.clone());
    }
    request_body.extend(body_params);

    Ok(ProviderRequest {
        prepared: PreparedRequest {
            url: format!("{API_AI21}/chat/completions"),
            headers: vec![
                ("accept".into(), "application/json".into()),
                ("content-type".into(), "application/json".into()),
                ("Authorization".into(), format!("Bearer {api_key}")),
            ],
            body: Value::Object(request_body),
            stream,
            transform: ResponseTransform::Passthrough,
        },
        transform: ResponseTransform::Passthrough,
    })
}

/// Mirror of `sendCohereRequest`.
pub fn build_cohere(
    body: &Value,
    secrets_doc: &Value,
    names: &PromptNames,
    placeholder: &str,
) -> Result<ProviderRequest, PrepareError> {
    let secret_id = get_str(body, "secret_id");
    let api_key = secrets::read_secret(secrets_doc, keys::COHERE, secret_id);
    if api_key.is_empty() {
        return Err(PrepareError::BadRequest("missing api key"));
    }
    let stream = get_bool(body, "stream");
    let model = get_str(body, "model").unwrap_or("").to_string();

    let converted = converters::convert_cohere_messages(
        &body.get("messages").cloned().unwrap_or(json!([])),
        names,
        placeholder,
    );

    let mut tools: Vec<Value> = Vec::new();
    if let Some(list) = body.get("tools").and_then(|t| t.as_array()) {
        tools.extend(list.iter().cloned());
        // Cohere rejects `$schema` inside a function's parameter schema.
        for tool in tools.iter_mut() {
            if let Some(params) = tool
                .pointer_mut("/function/parameters")
                .and_then(|p| p.as_object_mut())
            {
                params.remove("$schema");
            }
        }
    }

    // https://docs.cohere.com/reference/chat
    let mut request_body = Map::new();
    request_body.insert("stream".into(), json!(stream));
    copy_if_present(&mut request_body, "model", body);
    request_body.insert("messages".into(), converted);
    copy_if_present(&mut request_body, "temperature", body);
    copy_if_present(&mut request_body, "max_tokens", body);
    if let Some(top_k) = body.get("top_k").filter(|v| !v.is_null()) {
        request_body.insert("k".into(), top_k.clone());
    }
    if let Some(top_p) = body.get("top_p").filter(|v| !v.is_null()) {
        request_body.insert("p".into(), top_p.clone());
    }
    copy_if_present(&mut request_body, "seed", body);
    if let Some(stop) = body.get("stop").filter(|v| !v.is_null()) {
        request_body.insert("stop_sequences".into(), stop.clone());
    }
    copy_if_present(&mut request_body, "frequency_penalty", body);
    copy_if_present(&mut request_body, "presence_penalty", body);
    request_body.insert("documents".into(), json!([]));
    request_body.insert("tools".into(), Value::Array(tools));

    // Only the 08-2024 models accept an explicit safety mode.
    if model.ends_with("08-2024") {
        request_body.insert("safety_mode".into(), json!("OFF"));
    }
    if let Some(schema) = body.get("json_schema").filter(|v| v.is_object()) {
        request_body.insert(
            "response_format".into(),
            json!({
                "type": "json_schema",
                "schema": schema.get("value").cloned().unwrap_or(Value::Null),
            }),
        );
    }

    Ok(ProviderRequest {
        prepared: PreparedRequest {
            url: format!("{API_COHERE_V2}/chat"),
            headers: vec![
                ("Content-Type".into(), "application/json".into()),
                ("Authorization".into(), format!("Bearer {api_key}")),
            ],
            body: Value::Object(request_body),
            stream,
            transform: ResponseTransform::Passthrough,
        },
        transform: ResponseTransform::Passthrough,
    })
}

// ---------------------------------------------------------------------------
// Non-streaming response translation
// ---------------------------------------------------------------------------

/// Rewrite a non-streaming provider response into the shape the frontend
/// expects. Streaming bodies are never touched: the provider's own SSE already
/// matches what the client parses, which is why `forwardFetchResponse` upstream
/// is a pure pipe.
///
/// Returns `None` when the body should go out byte-for-byte.
pub fn apply_transform(transform: ResponseTransform, body: &[u8]) -> Option<Vec<u8>> {
    match transform {
        ResponseTransform::None | ResponseTransform::Passthrough => None,
        ResponseTransform::Claude => {
            let parsed: Value = serde_json::from_slice(body).ok()?;
            let text = parsed
                .pointer("/content/0/text")
                .and_then(|t| t.as_str())
                .unwrap_or("");
            // Upstream also keeps the raw content blocks so thought
            // signatures survive into the next turn.
            let reply = json!({
                "choices": [{ "message": { "content": text } }],
                "content": parsed.get("content").cloned().unwrap_or(Value::Null),
            });
            serde_json::to_vec(&reply).ok()
        }
        ResponseTransform::Gemini => {
            let parsed: Value = serde_json::from_slice(body).ok()?;
            let candidates = parsed
                .get("candidates")
                .and_then(|c| c.as_array())
                .filter(|c| !c.is_empty());
            let Some(candidates) = candidates else {
                let mut message = "API returned no candidate".to_string();
                if let Some(reason) = parsed
                    .pointer("/promptFeedback/blockReason")
                    .and_then(|r| r.as_str())
                {
                    message.push_str(&format!("\nPrompt was blocked due to : {reason}"));
                }
                return serde_json::to_vec(&json!({ "error": { "message": message } })).ok();
            };

            let content = candidates[0]
                .get("content")
                .or_else(|| candidates[0].get("output"))
                .cloned()
                .unwrap_or(Value::Null);
            let parts = candidates[0]
                .pointer("/content/parts")
                .and_then(|p| p.as_array())
                .cloned()
                .unwrap_or_default();
            let has_function_call = parts.iter().any(|p| p.get("functionCall").is_some());
            let has_inline_data = parts.iter().any(|p| p.get("inlineData").is_some());

            let text = match &content {
                Value::String(s) => s.clone(),
                _ => parts
                    .iter()
                    .filter(|part| part.get("thought").and_then(|t| t.as_bool()) != Some(true))
                    .filter_map(|part| part.get("text").and_then(|t| t.as_str()))
                    .collect::<Vec<_>>()
                    .join("\n\n"),
            };
            if text.is_empty() && !has_function_call && !has_inline_data {
                return serde_json::to_vec(
                    &json!({ "error": { "message": "Candidate text empty" } }),
                )
                .ok();
            }
            let reply = json!({
                "choices": [{ "message": { "content": text } }],
                "responseContent": content,
            });
            serde_json::to_vec(&reply).ok()
        }
    }
}
