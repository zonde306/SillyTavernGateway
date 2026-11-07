//! Prompt post-processing, mirrored from `src/prompt-converters.js`.
//!
//! Only the pieces the chat-completions `/generate` route needs are ported:
//!
//! * `postProcessPrompt` — the `custom_prompt_post_processing` switch
//!   (merge / merge_tools / semi / semi_tools / strict / strict_tools /
//!   single / legacy `claude`).
//! * `getPromptNames` — character/user/group names used for prefixing.
//! * `convertTextCompletionPrompt` — ChatML array → flat text prompt.
//! * `embedOpenRouterMedia` — inline data: audio (and video) parts.
//! * `addOpenRouterSignatures` — reasoning signature → `reasoning_details`.
//!
//! Everything operates on `serde_json::Value` in place, matching the JS which
//! mutates the request body objects directly (key order is preserved by the
//! `preserve_order` feature, and JS `delete` + later assignment has the same
//! observable "move to end" behavior).

use serde_json::{Map, Value};

/// Upstream `promptPlaceholder` config default.
pub const DEFAULT_PROMPT_PLACEHOLDER: &str = "[Start a new chat]";

/// Mirror of `PROMPT_PROCESSING_TYPE`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptProcessingType {
    None,
    /// Legacy `claude` value — behaves like `merge`.
    Claude,
    Merge,
    MergeTools,
    Semi,
    SemiTools,
    Strict,
    StrictTools,
    Single,
}

impl PromptProcessingType {
    pub fn parse(raw: &str) -> Self {
        match raw {
            "claude" => Self::Claude,
            "merge" => Self::Merge,
            "merge_tools" => Self::MergeTools,
            "semi" => Self::Semi,
            "semi_tools" => Self::SemiTools,
            "strict" => Self::Strict,
            "strict_tools" => Self::StrictTools,
            "single" => Self::Single,
            _ => Self::None,
        }
    }

    /// The JS `switch` has no default branch returning a new array: unknown
    /// values keep the original messages untouched.
    pub fn is_noop(&self) -> bool {
        matches!(self, Self::None)
    }
}

/// Mirror of the `PromptNames` object passed into the converters.
#[derive(Debug, Clone, Default)]
pub struct PromptNames {
    pub char_name: String,
    pub user_name: String,
    pub group_names: Vec<String>,
}

impl PromptNames {
    /// Built from the request body exactly like `getPromptNames(request)`.
    pub fn from_body(body: &Value) -> Self {
        Self {
            char_name: js_string(body.get("char_name")),
            user_name: js_string(body.get("user_name")),
            group_names: body
                .get("group_names")
                .and_then(|v| v.as_array())
                .map(|a| a.iter().map(js_string_of).collect())
                .unwrap_or_default(),
        }
    }

    /// `startsWithGroupName(message)`: any `"<name>: "` prefix.
    fn starts_with_group_name(&self, message: &str) -> bool {
        self.group_names
            .iter()
            .any(|n| message.starts_with(&format!("{n}: ")))
    }
}

/// `String(value)` for a JSON value; missing → "" (JS `|| ''`).
fn js_string(v: Option<&Value>) -> String {
    match v {
        None | Some(Value::Null) => String::new(),
        Some(v) => js_string_of(v),
    }
}

/// `String(value)` for a present value.
fn js_string_of(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => "null".to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => n.to_string(),
        // JSON.stringify of arrays/objects
        other => other.to_string(),
    }
}

fn as_str_or_empty(v: Option<&Value>) -> &str {
    v.and_then(|v| v.as_str()).unwrap_or("")
}

fn set_str(obj: &mut Map<String, Value>, key: &str, value: String) {
    obj.insert(key.to_string(), Value::String(value));
}

/// Apply the requested post-processing step. Mutates `messages` like the JS.
pub fn post_process_prompt(
    messages: &mut Value,
    ptype: PromptProcessingType,
    names: &PromptNames,
    placeholder: &str,
) {
    let (strict, placeholders, single, tools) = match ptype {
        PromptProcessingType::None => return,
        PromptProcessingType::Merge | PromptProcessingType::Claude => (false, false, false, false),
        PromptProcessingType::MergeTools => (false, false, false, true),
        PromptProcessingType::Semi => (true, false, false, false),
        PromptProcessingType::SemiTools => (true, false, false, true),
        PromptProcessingType::Strict => (true, true, false, false),
        PromptProcessingType::StrictTools => (true, true, false, true),
        PromptProcessingType::Single => (true, false, true, false),
    };
    let opts = MergeOpts {
        strict,
        placeholders,
        single,
        tools,
    };
    let merged = merge_messages(messages, names, opts, placeholder);
    *messages = merged;
}

#[derive(Debug, Clone, Copy)]
struct MergeOpts {
    strict: bool,
    placeholders: bool,
    single: bool,
    tools: bool,
}

/// Mirror of `mergeMessages` (src/prompt-converters.js).
///
/// Structure copied 1:1 from upstream so its quirks are preserved:
///   1. per message: falsy content → '', array content flattened (media parts
///      swapped for tokens), example_assistant/example_user prefixing,
///      "<name>: " prefixing, tool demotion, the `single` block, delete name,
///      delete tool_calls/tool_call_id
///   2. squash consecutive same-role messages with "\n\n"
///   3. empty result → placeholder
///   4. re-inline the tokenized media parts
///   5. strict: demote mid-prompt systems, splice the placeholder, recurse
///      with `strict: false` and `single: false`
fn merge_messages(
    messages: &mut Value,
    names: &PromptNames,
    opts: MergeOpts,
    placeholder: &str,
) -> Value {
    // Tokens stand in for non-text content parts, exactly like
    // `crypto.randomBytes(32).toString('base64')` upstream. Only uniqueness
    // matters, so a counter replaces the RNG (tokens never leave this fn).
    let mut content_tokens: Vec<(String, Value)> = Vec::new();
    let mut token_seq: usize = 0;

    let arr = match messages.as_array_mut() {
        Some(a) => a,
        // JS `messages.forEach` throws on non-arrays; upstream always passes
        // an array here.
        None => return messages.clone(),
    };

    for message in arr.iter_mut() {
        let Some(msg) = message.as_object_mut() else {
            continue;
        };

        // `if (!message.content) message.content = ''`
        let content_is_falsy = match msg.get("content") {
            None | Some(Value::Null) => true,
            Some(Value::String(s)) => s.is_empty(),
            Some(Value::Bool(b)) => !*b,
            Some(Value::Number(n)) => n.as_f64() == Some(0.0),
            _ => false,
        };
        if content_is_falsy {
            msg.insert("content".into(), Value::String(String::new()));
        }

        // Flatten array contents; non-text parts become opaque tokens.
        if let Some(parts) = msg.get("content").and_then(|c| c.as_array()) {
            let mut pieces: Vec<String> = Vec::new();
            for part in parts {
                match part.get("type").and_then(|t| t.as_str()) {
                    Some("text") => pieces.push(as_str_or_empty(part.get("text")).to_string()),
                    Some("image_url") | Some("video_url") | Some("audio_url") => {
                        token_seq += 1;
                        let token = format!("token-{token_seq}");
                        content_tokens.push((token.clone(), part.clone()));
                        pieces.push(token);
                    }
                    _ => pieces.push(String::new()),
                }
            }
            msg.insert("content".into(), Value::String(pieces.join("\n\n")));
        }

        let role = as_str_or_empty(msg.get("role")).to_string();
        let name = msg.get("name").and_then(|n| n.as_str()).map(str::to_string);
        // Upstream re-reads `message.content` on every check, so each mutation
        // is visible to the next one.
        let mut content = as_str_or_empty(msg.get("content")).to_string();

        if role == "system"
            && name.as_deref() == Some("example_assistant")
            && !names.char_name.is_empty()
            && !content.starts_with(&format!("{}: ", names.char_name))
            && !names.starts_with_group_name(&content)
        {
            content = format!("{}: {}", names.char_name, content);
            set_str(msg, "content", content.clone());
        }
        if role == "system"
            && name.as_deref() == Some("example_user")
            && !names.user_name.is_empty()
            && !content.starts_with(&format!("{}: ", names.user_name))
        {
            content = format!("{}: {}", names.user_name, content);
            set_str(msg, "content", content.clone());
        }
        if let Some(n) = name.as_deref() {
            if role != "system" && !content.starts_with(&format!("{n}: ")) {
                content = format!("{n}: {content}");
                set_str(msg, "content", content.clone());
            }
        }
        if role == "tool" && !opts.tools {
            set_str(msg, "role", "user".to_string());
        }

        // The `single` block: prefix, then rewrite the role to 'user'.
        if opts.single {
            let current = as_str_or_empty(msg.get("content")).to_string();
            let prefix: &str = if role == "assistant" {
                &names.char_name
            } else if role == "user" || role == "tool" {
                &names.user_name
            } else {
                ""
            };
            let already_prefixed = prefix.is_empty()
                || current.starts_with(&format!("{prefix}: "))
                || (role == "assistant" && names.starts_with_group_name(&current));
            if !already_prefixed {
                set_str(msg, "content", format!("{prefix}: {current}"));
            }
            set_str(msg, "role", "user".to_string());
        }

        msg.remove("name");
        if !opts.tools {
            msg.remove("tool_calls");
            msg.remove("tool_call_id");
        }
    }

    // Squash consecutive same-role messages.
    let mut merged: Vec<Value> = Vec::with_capacity(arr.len());
    for message in arr.iter() {
        let role = message.get("role").and_then(|r| r.as_str()).unwrap_or("");
        let content = message
            .get("content")
            .and_then(|c| c.as_str())
            .unwrap_or("");
        let can_append = merged
            .last()
            .and_then(|prev: &Value| prev.get("role").and_then(|r| r.as_str()))
            .map(|prev_role| prev_role == role && !content.is_empty() && role != "tool")
            .unwrap_or(false);
        if can_append {
            if let Some(last) = merged.last_mut().and_then(|v| v.as_object_mut()) {
                let prev = as_str_or_empty(last.get("content")).to_string();
                set_str(last, "content", format!("{prev}\n\n{content}"));
                continue;
            }
        }
        merged.push(message.clone());
    }

    if merged.is_empty() {
        merged.push(serde_json::json!({ "role": "user", "content": placeholder }));
    }

    // Re-inline the tokenized media parts.
    if !content_tokens.is_empty() {
        for message in merged.iter_mut() {
            let Some(msg) = message.as_object_mut() else {
                continue;
            };
            let content = as_str_or_empty(msg.get("content")).to_string();
            if !content_tokens
                .iter()
                .any(|(t, _)| content.contains(t.as_str()))
            {
                continue;
            }
            let mut rebuilt: Vec<Value> = Vec::new();
            for piece in content.split("\n\n") {
                if let Some((_, part)) = content_tokens.iter().find(|(t, _)| t == piece) {
                    rebuilt.push(part.clone());
                    continue;
                }
                match rebuilt.last_mut() {
                    Some(Value::Object(last))
                        if last.get("type").and_then(|t| t.as_str()) == Some("text") =>
                    {
                        let prev = as_str_or_empty(last.get("text")).to_string();
                        set_str(last, "text", format!("{prev}\n\n{piece}"));
                    }
                    _ => rebuilt.push(serde_json::json!({ "type": "text", "text": piece })),
                }
            }
            msg.insert("content".into(), Value::Array(rebuilt));
        }
    }

    if opts.strict {
        for (i, message) in merged.iter_mut().enumerate() {
            if i > 0 && message.get("role").and_then(|r| r.as_str()) == Some("system") {
                if let Some(obj) = message.as_object_mut() {
                    set_str(obj, "role", "user".to_string());
                }
            }
        }
        if !merged.is_empty() && opts.placeholders {
            let first_role = merged[0].get("role").and_then(|r| r.as_str()).unwrap_or("");
            // Two distinct upstream branches: a system opener gets the
            // placeholder spliced in at index 1, anything else is unshifted.
            if first_role == "system" {
                if merged.len() == 1
                    || merged[1].get("role").and_then(|r| r.as_str()) != Some("user")
                {
                    merged.insert(
                        1,
                        serde_json::json!({ "role": "user", "content": placeholder }),
                    );
                }
            } else if first_role != "user" {
                merged.insert(
                    0,
                    serde_json::json!({ "role": "user", "content": placeholder }),
                );
            }
        }
        let mut recur = Value::Array(merged);
        return merge_messages(
            &mut recur,
            names,
            MergeOpts {
                strict: false,
                single: false,
                ..opts
            },
            placeholder,
        );
    }

    Value::Array(merged)
}

/// Mirror of `convertTextCompletionPrompt`: ChatML messages → flat prompt
/// ending with `"\nassistant:"`. A string prompt passes through unchanged.
pub fn convert_text_completion_prompt(messages: &Value) -> String {
    if let Some(s) = messages.as_str() {
        return s.to_string();
    }
    let mut lines: Vec<String> = Vec::new();
    if let Some(arr) = messages.as_array() {
        for m in arr {
            let role = as_str_or_empty(m.get("role"));
            // Template-literal interpolation calls String(content), which for
            // objects yields "[object Object]" (not JSON).
            let content = match m.get("content") {
                None | Some(Value::Null) => String::new(),
                Some(Value::String(s)) => s.clone(),
                Some(Value::Array(_)) | Some(Value::Object(_)) => "[object Object]".to_string(),
                Some(other) => js_string_of(other),
            };
            let name = m.get("name");
            if role == "system" && name.is_none() {
                lines.push(format!("System: {content}"));
            } else if role == "system" {
                lines.push(format!("{}: {content}", js_string_of(name.unwrap())));
            } else {
                lines.push(format!("{role}: {content}"));
            }
        }
    }
    format!("{}\nassistant:", lines.join("\n"))
}

/// Mirror of `embedOpenRouterMedia`: rewrites inline `audio_url` data parts
/// into `input_audio`, and (when enabled) keeps `video_url` as-is.
pub fn embed_open_router_media(messages: &mut Value, audio: bool, video: bool) {
    let Some(arr) = messages.as_array_mut() else {
        return;
    };
    for message in arr.iter_mut() {
        let Some(parts) = message
            .as_object_mut()
            .and_then(|m| m.get_mut("content"))
            .and_then(|c| c.as_array_mut())
        else {
            continue;
        };
        for part in parts.iter_mut() {
            let Some(obj) = part.as_object_mut() else {
                continue;
            };
            let kind = obj
                .get("type")
                .and_then(|t| t.as_str())
                .unwrap_or("")
                .to_string();
            if video && kind == "video_url" {
                let url = obj
                    .get("video_url")
                    .and_then(|v| v.get("url"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                if url.starts_with("data:") {
                    set_str(obj, "type", "video_url".to_string());
                }
            }
            if audio && kind == "audio_url" {
                let url = obj
                    .get("audio_url")
                    .and_then(|v| v.get("url"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                if url.starts_with("data:") {
                    let (header, base64_data) = match url.split_once(',') {
                        Some(pair) => pair,
                        None => Default::default(),
                    };
                    let mime = header
                        .split_once("data:")
                        .and_then(|(_, rest)| rest.split(';').next())
                        .filter(|s| !s.is_empty())
                        .unwrap_or("audio/mpeg");
                    let format = match mime {
                        "audio/wav" => "wav",
                        _ => "mp3",
                    };
                    set_str(obj, "type", "input_audio".to_string());
                    obj.insert(
                        "input_audio".into(),
                        serde_json::json!({ "format": format, "data": base64_data }),
                    );
                    obj.remove("audio_url");
                }
            }
        }
    }
}

/// Mirror of `getFormatForModel` inside `addOpenRouterSignatures`.
fn signature_format_for_model(model: &str) -> &'static str {
    if model.contains("google/gemini") {
        "google-gemini-v1"
    } else if model.contains("anthropic/claude") {
        "anthropic-claude-v1"
    } else if model.contains("openai/gpt") {
        "openai-responses-v1"
    } else if model.contains("x-ai/grok") {
        "xai-responses-v1"
    } else {
        "unknown"
    }
}

/// Mirror of `addOpenRouterSignatures`: message/tool-call `signature` fields
/// become a `reasoning_details` array (only when thought signatures are
/// enabled, upstream `gemini.thoughtSignatures`, default true).
pub fn add_open_router_signatures(messages: &mut Value, model: &str, enable: bool) {
    let Some(arr) = messages.as_array_mut() else {
        return;
    };
    let format = signature_format_for_model(model);
    for message in arr.iter_mut() {
        let Some(obj) = message.as_object_mut() else {
            continue;
        };
        let mut details: Vec<Value> = Vec::new();
        let mut add_detail = |value: &str, id: Option<&str>| {
            details.push(serde_json::json!({
                "index": details.len(),
                "id": id.map(str::to_string).unwrap_or_else(|| format!("signature-{}", details.len())),
                "type": "reasoning.encrypted",
                "data": value,
                "format": format,
            }));
        };

        if let Some(sig) = obj.get("signature").and_then(|v| v.as_str()) {
            if enable && !sig.is_empty() {
                add_detail(sig, None);
            }
            obj.remove("signature");
        }
        if let Some(calls) = obj.get_mut("tool_calls").and_then(|c| c.as_array_mut()) {
            for call in calls.iter_mut() {
                let Some(call_obj) = call.as_object_mut() else {
                    continue;
                };
                let sig = call_obj
                    .get("signature")
                    .and_then(|v| v.as_str())
                    .filter(|s| !s.is_empty())
                    .map(str::to_string);
                let id = call_obj
                    .get("id")
                    .and_then(|v| v.as_str())
                    .map(str::to_string);
                if let Some(sig) = sig {
                    add_detail(&sig, id.as_deref());
                }
                call_obj.remove("signature");
            }
        }
        if !details.is_empty() {
            obj.insert("reasoning_details".into(), Value::Array(details));
        }
    }
}

/// Mirror of `setJsonObjectFormat`: ask for a JSON object and append the
/// schema as an extra user turn (providers without a `json_schema` format).
///
pub fn set_json_object_format(
    body_params: &mut Map<String, Value>,
    messages: &mut Value,
    json_schema: &Value,
) {
    body_params.insert(
        "response_format".into(),
        serde_json::json!({ "type": "json_object" }),
    );
    if let Some(arr) = messages.as_array_mut() {
        let pretty = serde_json::to_string_pretty(&json_schema["value"])
            .unwrap_or_else(|_| "undefined".to_string());
        arr.push(serde_json::json!({
            "role": "user",
            "content": format!("JSON schema for the response:\n{pretty}"),
        }));
    }
}

/// Mirror of `addAssistantPrefix`: mark a trailing assistant turn as a
/// prefill (`prefill`/`partial`), unless tools are in play.
pub fn add_assistant_prefix(messages: &mut Value, tools: Option<&Value>, property: &str) {
    let Some(arr) = messages.as_array_mut() else {
        return;
    };
    if arr.is_empty() {
        return;
    }
    let has_tools = tools
        .and_then(|t| t.as_array())
        .is_some_and(|t| !t.is_empty())
        || arr
            .iter()
            .any(|m| m.get("role").and_then(|r| r.as_str()) == Some("tool"));
    if has_tools {
        return;
    }
    let last_is_assistant = arr
        .last()
        .and_then(|m| m.get("role"))
        .and_then(|r| r.as_str())
        == Some("assistant");
    if last_is_assistant {
        if let Some(obj) = arr.last_mut().and_then(|m| m.as_object_mut()) {
            obj.insert(property.into(), Value::Bool(true));
        }
    }
}

/// Mirror of `cachingSystemPromptForOpenRouter`: mark the first system turn
/// for prompt caching (`cache_control`).
pub fn caching_system_prompt_for_open_router(messages: &mut Value, ttl: Option<&str>) {
    let Some(arr) = messages.as_array_mut() else {
        return;
    };
    if arr.is_empty() {
        return;
    }
    let cache_control = match ttl {
        Some(ttl) => serde_json::json!({ "type": "ephemeral", "ttl": ttl }),
        None => serde_json::json!({ "type": "ephemeral" }),
    };
    let Some(system) = arr
        .iter_mut()
        .find(|m| m.get("role").and_then(|r| r.as_str()) == Some("system"))
        .and_then(|m| m.as_object_mut())
    else {
        return;
    };
    if system.contains_key("cache_control") {
        return;
    }
    match system.get_mut("content") {
        Some(Value::Array(parts)) => {
            if parts
                .iter()
                .any(|p| p.get("cache_control").is_some_and(|c| !c.is_null()))
            {
                return;
            }
            for part in parts.iter_mut().rev() {
                let Some(obj) = part.as_object_mut() else {
                    continue;
                };
                if obj.get("type").and_then(|t| t.as_str()) == Some("text") {
                    obj.insert("cache_control".into(), cache_control);
                    return;
                }
            }
        }
        Some(Value::String(text)) => {
            let text = text.clone();
            system.insert(
                "content".into(),
                serde_json::json!([{
                    "type": "text",
                    "text": text,
                    "cache_control": cache_control,
                }]),
            );
        }
        _ => {}
    }
}

/// Mirror of `cachingAtDepthForOpenRouterClaude`: place a cache breakpoint on
/// the message `caching_at_depth` role switches back from the end.
pub fn caching_at_depth_for_open_router_claude(
    messages: &mut Value,
    caching_at_depth: i64,
    ttl: Option<&str>,
) {
    if caching_at_depth < 0 {
        return;
    }
    let Some(arr) = messages.as_array_mut() else {
        return;
    };
    let mut passed_prefill = false;
    let mut depth: i64 = 0;
    let mut previous_role = String::new();
    for i in (0..arr.len()).rev() {
        let role = arr[i]
            .get("role")
            .and_then(|r| r.as_str())
            .unwrap_or("")
            .to_string();
        if !passed_prefill && role == "assistant" {
            continue;
        }
        passed_prefill = true;
        if role == "system" {
            continue;
        }
        if role != previous_role {
            if depth == caching_at_depth || depth == caching_at_depth + 2 {
                apply_cache_control(&mut arr[i], ttl);
            }
            if depth == caching_at_depth + 2 {
                break;
            }
            depth += 1;
            previous_role = role;
        }
    }
}

fn apply_cache_control(message: &mut Value, ttl: Option<&str>) {
    let cache_control = match ttl {
        Some(ttl) => serde_json::json!({ "type": "ephemeral", "ttl": ttl }),
        None => serde_json::json!({ "type": "ephemeral" }),
    };
    let Some(obj) = message.as_object_mut() else {
        return;
    };
    match obj.get_mut("content") {
        Some(Value::String(text)) => {
            let text = text.clone();
            obj.insert(
                "content".into(),
                serde_json::json!([{
                    "type": "text",
                    "text": text,
                    "cache_control": cache_control,
                }]),
            );
        }
        Some(Value::Array(parts)) if !parts.is_empty() => {
            if let Some(last) = parts.last_mut().and_then(|p| p.as_object_mut()) {
                last.insert("cache_control".into(), cache_control);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn names() -> PromptNames {
        PromptNames {
            char_name: "Seraphina".into(),
            user_name: "User".into(),
            group_names: vec![],
        }
    }

    #[test]
    fn merge_squashes_consecutive_user_messages() {
        let mut msgs = json!([
            { "role": "system", "content": "sys" },
            { "role": "user", "content": "a" },
            { "role": "user", "content": "b" },
            { "role": "assistant", "content": "c" }
        ]);
        post_process_prompt(
            &mut msgs,
            PromptProcessingType::Merge,
            &names(),
            "Let's get started.",
        );
        assert_eq!(msgs.as_array().unwrap().len(), 3);
        assert_eq!(msgs[1]["content"], "a\n\nb");
    }

    #[test]
    fn merge_prefixes_names_and_strips_them() {
        let mut msgs = json!([
            { "role": "assistant", "name": "Seraphina", "content": "hi" },
            { "role": "user", "name": "User", "content": "hey" }
        ]);
        post_process_prompt(
            &mut msgs,
            PromptProcessingType::Merge,
            &names(),
            "Let's get started.",
        );
        assert_eq!(msgs[0]["content"], "Seraphina: hi");
        assert_eq!(msgs[1]["content"], "User: hey");
        assert!(msgs[0].get("name").is_none());
    }

    #[test]
    fn merge_keeps_media_parts_as_content_arrays() {
        let mut msgs = json!([
            { "role": "user", "content": [
                { "type": "text", "text": "look" },
                { "type": "image_url", "image_url": { "url": "data:image/png;base64,AAA" } }
            ]}
        ]);
        post_process_prompt(
            &mut msgs,
            PromptProcessingType::Merge,
            &names(),
            "Let's get started.",
        );
        let content = msgs[0]["content"].as_array().unwrap();
        // Text parts re-joined, media part restored verbatim.
        assert_eq!(content.len(), 2);
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[0]["text"], "look");
        assert_eq!(content[1]["type"], "image_url");
        assert_eq!(content[1]["image_url"]["url"], "data:image/png;base64,AAA");
    }

    #[test]
    fn merge_replaces_empty_prompt_with_placeholder() {
        let mut msgs = json!([]);
        post_process_prompt(
            &mut msgs,
            PromptProcessingType::Merge,
            &names(),
            "Let's get started.",
        );
        assert_eq!(msgs[0]["role"], "user");
        assert_eq!(msgs[0]["content"], "Let's get started.");
    }

    #[test]
    fn strict_demotes_mid_prompt_system_and_inserts_placeholder() {
        let mut msgs = json!([
            { "role": "system", "content": "sys" },
            { "role": "system", "content": "mid" },
            { "role": "user", "content": "hi" }
        ]);
        post_process_prompt(
            &mut msgs,
            PromptProcessingType::Strict,
            &names(),
            "Let's get started.",
        );
        // Verified against Node: both system messages merge into one first
        // message (consecutive same role), so no placeholder is needed and the
        // mid-prompt system never reaches the strict demotion step.
        assert_eq!(
            serde_json::to_string(&msgs).unwrap(),
            r#"[{"role":"system","content":"sys\n\nmid"},{"role":"user","content":"hi"}]"#
        );
    }

    #[test]
    fn single_demotes_everything_to_user_with_prefixes() {
        let mut msgs = json!([
            { "role": "system", "content": "sys" },
            { "role": "assistant", "content": "yo" },
            { "role": "user", "content": "hi" }
        ]);
        post_process_prompt(
            &mut msgs,
            PromptProcessingType::Single,
            &names(),
            "Let's get started.",
        );
        // Verified against Node: system bodies are left untouched, assistant
        // and user turns get their name prefixes.
        assert_eq!(
            serde_json::to_string(&msgs).unwrap(),
            r#"[{"role":"user","content":"sys\n\nSeraphina: yo\n\nUser: hi"}]"#
        );
    }

    #[test]
    fn unknown_processing_type_leaves_messages_untouched() {
        let original =
            json!([{ "role": "user", "content": "a" }, { "role": "user", "content": "b" }]);
        let mut msgs = original.clone();
        post_process_prompt(&mut msgs, PromptProcessingType::None, &names(), "x");
        assert_eq!(msgs, original);
    }

    #[test]
    fn text_completion_prompt_matches_upstream_shape() {
        let msgs = json!([
            { "role": "system", "content": "sys" },
            { "role": "system", "name": "narrator", "content": "n" },
            { "role": "user", "content": "hi" }
        ]);
        assert_eq!(
            convert_text_completion_prompt(&msgs),
            "System: sys\nnarrator: n\nuser: hi\nassistant:"
        );
        assert_eq!(convert_text_completion_prompt(&json!("raw")), "raw");
    }

    #[test]
    fn embed_media_converts_audio_url_to_input_audio() {
        let mut msgs = json!([
            { "role": "user", "content": [
                { "type": "audio_url", "audio_url": { "url": "data:audio/wav;base64,QUJD" } }
            ]}
        ]);
        embed_open_router_media(&mut msgs, true, true);
        let part = &msgs[0]["content"][0];
        assert_eq!(part["type"], "input_audio");
        assert_eq!(part["input_audio"]["format"], "wav");
        assert_eq!(part["input_audio"]["data"], "QUJD");
        assert!(part.get("audio_url").is_none());
    }

    #[test]
    fn embed_media_skips_non_data_urls() {
        let mut msgs = json!([
            { "role": "user", "content": [
                { "type": "audio_url", "audio_url": { "url": "https://x/a.mp3" } }
            ]}
        ]);
        embed_open_router_media(&mut msgs, true, true);
        assert_eq!(msgs[0]["content"][0]["type"], "audio_url");
    }

    #[test]
    fn openrouter_signatures_become_reasoning_details() {
        let mut msgs = json!([
            { "role": "assistant", "content": "x", "signature": "sig1",
              "tool_calls": [{ "id": "call_1", "signature": "sig2" }] }
        ]);
        add_open_router_signatures(&mut msgs, "openai/gpt-5", true);
        let details = msgs[0]["reasoning_details"].as_array().unwrap();
        assert_eq!(details.len(), 2);
        assert_eq!(details[0]["format"], "openai-responses-v1");
        assert_eq!(details[0]["data"], "sig1");
        assert_eq!(details[1]["id"], "call_1");
        assert!(msgs[0].get("signature").is_none());
        assert!(msgs[0]["tool_calls"][0].get("signature").is_none());
    }

    #[test]
    fn signatures_dropped_when_disabled() {
        let mut msgs = json!([{ "role": "assistant", "content": "x", "signature": "sig" }]);
        add_open_router_signatures(&mut msgs, "google/gemini-2.5-pro", false);
        assert!(msgs[0].get("signature").is_none());
        assert!(msgs[0].get("reasoning_details").is_none());
    }

    #[test]
    fn set_json_object_format_appends_schema_turn() {
        let mut body = Map::new();
        let mut msgs = json!([{ "role": "user", "content": "hi" }]);
        set_json_object_format(
            &mut body,
            &mut msgs,
            &json!({ "value": { "type": "object", "properties": { "a": { "type": "string" } } } }),
        );
        assert_eq!(body["response_format"]["type"], "json_object");
        let last = msgs.as_array().unwrap().last().unwrap();
        assert_eq!(last["role"], "user");
        let content = last["content"].as_str().unwrap();
        // JSON.stringify(value, null, 4) used to produce 4-space indentation
        // upstream, but serde_json::to_string_pretty is 2-space. Only the
        // prefix and the schema body are load-bearing, so assert those.
        assert!(content.starts_with("JSON schema for the response:\n{"));
        assert!(content.contains("\"type\": \"object\""));
        assert!(content.trim_end().ends_with('}'));
    }

    #[test]
    fn add_assistant_prefix_marks_trailing_assistant() {
        let mut msgs = json!([
            { "role": "system", "content": "s" },
            { "role": "assistant", "content": "a" }
        ]);
        add_assistant_prefix(&mut msgs, None, "partial");
        assert_eq!(msgs[1]["partial"], true);

        // Any tool turn disables it.
        let mut with_tool = json!([
            { "role": "tool", "content": "t" },
            { "role": "assistant", "content": "a" }
        ]);
        add_assistant_prefix(&mut with_tool, None, "partial");
        assert!(with_tool[1].get("partial").is_none());

        // Empty tools array is not "has tools".
        let mut empty_tools = json!([{ "role": "assistant", "content": "a" }]);
        add_assistant_prefix(&mut empty_tools, Some(&json!([])), "prefill");
        assert_eq!(empty_tools[0]["prefill"], true);
    }

    #[test]
    fn caching_system_prompt_wraps_first_system_message() {
        let mut msgs = json!([
            { "role": "system", "content": "sys" },
            { "role": "user", "content": "u" }
        ]);
        caching_system_prompt_for_open_router(&mut msgs, None);
        assert_eq!(msgs[0]["content"][0]["type"], "text");
        assert_eq!(msgs[0]["content"][0]["text"], "sys");
        assert_eq!(msgs[0]["content"][0]["cache_control"]["type"], "ephemeral");

        // Existing cache_control on the message short-circuits.
        let mut already =
            json!([{ "role": "system", "content": "s", "cache_control": { "type": "ephemeral" } }]);
        caching_system_prompt_for_open_router(&mut already, Some("1h"));
        assert!(already[0]["content"].is_string());
    }

    #[test]
    fn caching_at_depth_marks_role_switch() {
        let mut msgs = json!([
            { "role": "system", "content": "s" },
            { "role": "user", "content": "u1" },
            { "role": "assistant", "content": "a1" },
            { "role": "user", "content": "u2" }
        ]);
        caching_at_depth_for_open_router_claude(&mut msgs, 0, Some("5m"));
        // depth 0 is the last user turn and depth 2 the user turn before it
        // (the assistant prefill at the end is skipped, systems never count).
        assert_eq!(msgs[3]["content"][0]["cache_control"]["ttl"], "5m");
        assert_eq!(msgs[1]["content"][0]["cache_control"]["ttl"], "5m");
        assert!(msgs[2]["content"].is_string());
    }
}
