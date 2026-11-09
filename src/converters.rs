//! Message converters — a port of the relevant half of
//! `src/prompt-converters.js`.
//!
//! Every provider that speaks something other than the OpenAI chat format
//! needs the ChatML history rewritten first. The functions here mirror the
//! upstream ones statement by statement, including their quirks, because the
//! provider sees the result verbatim and a silently different prompt is a
//! behaviour change users notice as "the bot got worse".
//!
//! Naming follows the JS: `convertClaudeMessages` → [`convert_claude_messages`]
//! and so on. Inputs are borrowed, outputs are fresh values, so callers can
//! keep the original body around for logging.

use serde_json::{json, Value};

use crate::prompt::PromptNames;

/// The upstream *default* placeholder. The live value comes from
/// `promptPlaceholder` in config.yaml and is passed in by the caller.
pub const PROMPT_PLACEHOLDER: &str = "Let's get started.";

// --- small JS helpers ------------------------------------------------------

/// `String(x ?? '')` for content-ish values.
pub fn js_str(value: Option<&Value>) -> String {
    match value {
        None | Some(Value::Null) => String::new(),
        Some(Value::String(s)) => s.clone(),
        Some(Value::Bool(b)) => b.to_string(),
        Some(Value::Number(n)) => n.to_string(),
        // `String([{...}])` is "[object Object]" for upstream, but content is
        // always a string here in practice; JSON is the honest fallback.
        Some(other) => other.to_string(),
    }
}

fn role_of(message: &Value) -> &str {
    message.get("role").and_then(|r| r.as_str()).unwrap_or("")
}

fn name_of(message: &Value) -> Option<&str> {
    message.get("name").and_then(|n| n.as_str())
}

/// `typeof str === 'string' ? JSON.parse(str) : str`, with the JS throw
/// swallowed: upstream lets it propagate, but a malformed tool call must not
/// take down the whole request here.
fn parse_tool_arguments(value: &Value) -> Value {
    match value {
        Value::String(s) => serde_json::from_str(s).unwrap_or(Value::Null),
        other => other.clone(),
    }
}

/// `message.content.startsWith(prefix)` guarded by truthiness of the message.
fn content_starts_with(message: &Value, prefix: &str) -> bool {
    message
        .get("content")
        .and_then(|c| c.as_str())
        .is_some_and(|c| c.starts_with(prefix))
}

/// Prepend `"<prefix>"` to a string content field when needed.
fn prefix_content(message: &mut Value, prefix: &str) {
    let Some(obj) = message.as_object_mut() else {
        return;
    };
    let current = obj
        .get("content")
        .cloned()
        .unwrap_or(Value::String(String::new()));
    let text = match current {
        Value::String(s) => s,
        other => js_str(Some(&other)),
    };
    obj.insert("content".into(), Value::String(format!("{prefix}{text}")));
}

/// The `example_user` / `example_assistant` naming pass shared by the Claude,
/// AI21, Mistral and Cohere converters. Returns the (possibly rewritten)
/// message with its `name` field removed when the JS would have removed it.
fn apply_example_names(message: &mut Value, names: &PromptNames, remove_name_role_system: bool) {
    let role = role_of(message).to_string();
    let name = name_of(message).unwrap_or("").to_string();

    if names.user_name.is_empty() {
        // fall through: the JS only prefixes when a name exists
    } else if name == "example_user"
        && !content_starts_with(message, &format!("{}: ", names.user_name))
    {
        prefix_content(message, &format!("{}: ", names.user_name));
    }

    if !names.char_name.is_empty() && name == "example_assistant" {
        let already = content_starts_with(message, &format!("{}: ", names.char_name))
            || message
                .get("content")
                .and_then(|c| c.as_str())
                .is_some_and(|c| names.starts_with_group_name(c));
        if !already {
            prefix_content(message, &format!("{}: ", names.char_name));
        }
    }

    if remove_name_role_system && role == "system" {
        if let Some(obj) = message.as_object_mut() {
            obj.remove("name");
        }
    }
}

// ---------------------------------------------------------------------------
// Claude
// ---------------------------------------------------------------------------

/// Mirror of `convertClaudeMessages`.
///
/// Returns `(messages, system_prompt)`. `system_prompt` is the array of
/// `{type:"text",text}` blocks derived from the leading system messages, which
/// the caller sends as Anthropic's top-level `system` field.
pub fn convert_claude_messages(
    messages: &Value,
    prefill_string: Option<&str>,
    use_sys_prompt: bool,
    use_tools: bool,
    names: &PromptNames,
    placeholder: &str,
) -> (Value, Value) {
    let mut messages: Vec<Value> = messages.as_array().cloned().unwrap_or_default();
    let mut system_prompt: Vec<Value> = Vec::new();

    if use_sys_prompt {
        // Collect leading system messages, then drop them from the array.
        let mut i = 0;
        while i < messages.len() {
            if role_of(&messages[i]) != "system" {
                break;
            }
            if !names.user_name.is_empty()
                && name_of(&messages[i]) == Some("example_user")
                && !content_starts_with(&messages[i], &format!("{}: ", names.user_name))
            {
                prefix_content(&mut messages[i], &format!("{}: ", names.user_name));
            }
            if !names.char_name.is_empty() && name_of(&messages[i]) == Some("example_assistant") {
                let already = content_starts_with(&messages[i], &format!("{}: ", names.char_name))
                    || messages[i]
                        .get("content")
                        .and_then(|c| c.as_str())
                        .is_some_and(|c| names.starts_with_group_name(c));
                if !already {
                    prefix_content(&mut messages[i], &format!("{}: ", names.char_name));
                }
            }
            let text = messages[i]
                .get("content")
                .cloned()
                .unwrap_or(Value::String(String::new()));
            system_prompt.push(json!({ "type": "text", "text": text }));
            i += 1;
        }
        messages.drain(0..i);

        if messages.is_empty() {
            messages.insert(0, json!({ "role": "user", "content": placeholder }));
        }
    }

    for message in messages.iter_mut() {
        // assistant tool_calls → tool_use content blocks
        if role_of(message) == "assistant" {
            if let Some(calls) = message.get("tool_calls").and_then(|t| t.as_array()) {
                let blocks: Vec<Value> = calls
                    .iter()
                    .map(|tc| {
                        json!({
                            "type": "tool_use",
                            "id": tc.get("id").cloned().unwrap_or(Value::Null),
                            "name": tc.pointer("/function/name").cloned().unwrap_or(Value::Null),
                            "input": parse_tool_arguments(
                                tc.pointer("/function/arguments").unwrap_or(&Value::Null)
                            ),
                        })
                    })
                    .collect();
                if let Some(obj) = message.as_object_mut() {
                    obj.insert("content".into(), Value::Array(blocks));
                }
            }
        }

        // tool result → user message with a tool_result block
        if role_of(message) == "tool" {
            let content = message.get("content").cloned().unwrap_or(Value::Null);
            let call_id = message.get("tool_call_id").cloned().unwrap_or(Value::Null);
            if let Some(obj) = message.as_object_mut() {
                obj.insert("role".into(), json!("user"));
                obj.insert(
                    "content".into(),
                    json!([{
                        "type": "tool_result",
                        "tool_use_id": call_id,
                        "content": content,
                    }]),
                );
            }
        }

        if role_of(message) == "system" {
            apply_example_names(message, names, false);
            if let Some(obj) = message.as_object_mut() {
                obj.insert("role".into(), json!("user"));
                obj.remove("name");
            }
        }

        // Content → array of parts.
        match message.get("content").cloned() {
            Some(Value::String(text)) => {
                let text = if let Some(name) = name_of(message).map(str::to_string) {
                    format!("{name}: {text}")
                } else {
                    text
                };
                if let Some(obj) = message.as_object_mut() {
                    obj.insert("content".into(), json!([{ "type": "text", "text": text }]));
                }
            }
            Some(Value::Array(parts)) => {
                let msg_name = name_of(message).map(str::to_string);
                let converted: Vec<Value> = parts
                    .iter()
                    .map(|part| {
                        match part.get("type").and_then(|t| t.as_str()) {
                            Some("image_url") => {
                                let url = part
                                    .pointer("/image_url/url")
                                    .and_then(|u| u.as_str())
                                    .unwrap_or("");
                                let media_type = url
                                    .split(';')
                                    .next()
                                    .and_then(|s| s.split(':').nth(1))
                                    .unwrap_or("image/png");
                                let data = url.split(',').nth(1).unwrap_or("");
                                json!({
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": media_type,
                                        "data": data,
                                    },
                                })
                            }
                            Some("text") => {
                                let mut text = part
                                    .get("text")
                                    .and_then(|t| t.as_str())
                                    .unwrap_or("")
                                    .to_string();
                                if let Some(name) = &msg_name {
                                    text = format!("{name}: {text}");
                                }
                                // Empty text becomes a zero-width space.
                                if text.is_empty() {
                                    text = "\u{200b}".to_string();
                                }
                                json!({ "type": "text", "text": text })
                            }
                            _ => part.clone(),
                        }
                    })
                    .collect();
                if let Some(obj) = message.as_object_mut() {
                    obj.insert("content".into(), Value::Array(converted));
                }
            }
            _ => {}
        }

        if let Some(obj) = message.as_object_mut() {
            obj.remove("name");
            obj.remove("tool_calls");
            obj.remove("tool_call_id");
        }
    }

    // Images in assistant turns must move into the following user turn.
    let mut i = 0;
    while i < messages.len() {
        let has_image = role_of(&messages[i]) == "assistant"
            && messages[i]
                .get("content")
                .and_then(|c| c.as_array())
                .is_some_and(|parts| {
                    parts
                        .iter()
                        .any(|p| p.get("type").and_then(|t| t.as_str()) == Some("image"))
                });
        if !has_image {
            i += 1;
            continue;
        }
        let mut j = i + 1;
        while j < messages.len() && role_of(&messages[j]) != "user" {
            j += 1;
        }
        if j >= messages.len() {
            messages.insert(i + 1, json!({ "role": "user", "content": [] }));
            j = i + 1;
        }
        let images: Vec<Value> = messages[i]
            .get("content")
            .and_then(|c| c.as_array())
            .map(|parts| {
                parts
                    .iter()
                    .filter(|p| p.get("type").and_then(|t| t.as_str()) == Some("image"))
                    .cloned()
                    .collect()
            })
            .unwrap_or_default();
        if let Some(obj) = messages[i].as_object_mut() {
            let kept: Vec<Value> = obj
                .get("content")
                .and_then(|c| c.as_array())
                .map(|parts| {
                    parts
                        .iter()
                        .filter(|p| p.get("type").and_then(|t| t.as_str()) != Some("image"))
                        .cloned()
                        .collect()
                })
                .unwrap_or_default();
            obj.insert("content".into(), Value::Array(kept));
        }
        if let Some(dst) = messages[j]
            .get_mut("content")
            .and_then(|c| c.as_array_mut())
        {
            dst.extend(images);
        }
        i += 1;
    }

    if let Some(prefill) = prefill_string {
        messages.push(json!({
            "role": "assistant",
            "content": [{ "type": "text", "text": prefill.trim_end() }],
        }));
    }

    // Merge consecutive same-role messages.
    let mut merged: Vec<Value> = Vec::new();
    for message in messages.into_iter() {
        let same_as_last = merged
            .last()
            .is_some_and(|last| role_of(last) == role_of(&message));
        if same_as_last {
            let incoming = message
                .get("content")
                .and_then(|c| c.as_array())
                .cloned()
                .unwrap_or_default();
            if let Some(dst) = merged
                .last_mut()
                .and_then(|m| m.get_mut("content"))
                .and_then(|c| c.as_array_mut())
            {
                dst.extend(incoming);
            }
        } else {
            merged.push(message);
        }
    }

    if !use_tools {
        for message in merged.iter_mut() {
            let Some(parts) = message.get_mut("content").and_then(|c| c.as_array_mut()) else {
                continue;
            };
            for part in parts.iter_mut() {
                match part.get("type").and_then(|t| t.as_str()) {
                    Some("tool_use") => {
                        let input = part.get("input").cloned().unwrap_or(Value::Null);
                        if let Some(obj) = part.as_object_mut() {
                            obj.insert("type".into(), json!("text"));
                            obj.insert("text".into(), Value::String(input.to_string()));
                            obj.remove("id");
                            obj.remove("name");
                            obj.remove("input");
                        }
                    }
                    Some("tool_result") => {
                        let content = part.get("content").cloned().unwrap_or(Value::Null);
                        if let Some(obj) = part.as_object_mut() {
                            obj.insert("type".into(), json!("text"));
                            obj.insert("text".into(), content);
                            obj.remove("tool_use_id");
                            obj.remove("content");
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    (Value::Array(merged), Value::Array(system_prompt))
}

// ---------------------------------------------------------------------------
// AI21
// ---------------------------------------------------------------------------

/// Mirror of `convertAI21Messages`: squash the leading system messages into one
/// `system` turn, merge consecutive same-role turns with a blank line.
pub fn convert_ai21_messages(messages: &Value, names: &PromptNames, placeholder: &str) -> Value {
    let Some(input) = messages.as_array() else {
        return json!([]);
    };
    let mut messages: Vec<Value> = input.clone();

    let mut i = 0;
    let mut system_prompt = String::new();
    while i < messages.len() {
        if role_of(&messages[i]) != "system" {
            break;
        }
        apply_example_names(&mut messages[i], names, false);
        system_prompt.push_str(&js_str(messages[i].get("content")));
        system_prompt.push_str("\n\n");
        i += 1;
    }
    messages.drain(0..i);

    if messages.is_empty() {
        messages.push(json!({ "role": "user", "content": placeholder }));
    }

    if !system_prompt.is_empty() {
        messages.insert(
            0,
            json!({ "role": "system", "content": system_prompt.trim() }),
        );
    }

    for message in messages.iter_mut() {
        if let Some(name) = name_of(message).map(str::to_string) {
            if role_of(message) != "system" && !content_starts_with(message, &format!("{name}: ")) {
                prefix_content(message, &format!("{name}: "));
            }
            if let Some(obj) = message.as_object_mut() {
                obj.remove("name");
            }
        }
    }

    let mut merged: Vec<Value> = Vec::new();
    for message in messages.into_iter() {
        let same_as_last = merged
            .last()
            .is_some_and(|last| role_of(last) == role_of(&message));
        if same_as_last {
            let incoming = js_str(message.get("content"));
            if let Some(last) = merged.last_mut() {
                let existing = js_str(last.get("content"));
                if let Some(obj) = last.as_object_mut() {
                    obj.insert(
                        "content".into(),
                        Value::String(format!("{existing}\n\n{incoming}")),
                    );
                }
            }
        } else {
            merged.push(message);
        }
    }

    Value::Array(merged)
}

// ---------------------------------------------------------------------------
// Mistral
// ---------------------------------------------------------------------------

/// `crypto.createHash('sha512').update(id).digest('hex').slice(0, 9)`.
fn sanitize_tool_id(id: &str) -> String {
    use sha2::Digest;
    let digest = sha2::Sha512::digest(id.as_bytes());
    digest
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>()
        .chars()
        .take(9)
        .collect()
}

/// Mirror of `convertMistralMessages`.
pub fn convert_mistral_messages(
    messages: &Value,
    names: &PromptNames,
    enable_prefix: bool,
) -> Value {
    let Some(input) = messages.as_array() else {
        return json!([]);
    };
    let mut messages: Vec<Value> = input.clone();

    if enable_prefix {
        if let Some(last) = messages.last_mut() {
            if role_of(last) == "assistant" {
                if let Some(obj) = last.as_object_mut() {
                    obj.insert("prefix".into(), json!(true));
                }
            }
        }
    }

    for message in messages.iter_mut() {
        let mut new_calls: Option<Vec<Value>> = None;
        if let Some(calls) = message.get("tool_calls").and_then(|t| t.as_array()) {
            new_calls = Some(
                calls
                    .iter()
                    .map(|call| {
                        let mut call = call.clone();
                        if let Some(id) = call.get("id").and_then(|i| i.as_str()) {
                            let clean = sanitize_tool_id(id);
                            if let Some(obj) = call.as_object_mut() {
                                obj.insert("id".into(), Value::String(clean));
                            }
                        }
                        call
                    })
                    .collect(),
            );
        }
        if let Some(calls) = new_calls {
            if let Some(obj) = message.as_object_mut() {
                obj.insert("tool_calls".into(), Value::Array(calls));
            }
        }

        if role_of(message) == "tool" {
            if let Some(id) = message.get("tool_call_id").and_then(|i| i.as_str()) {
                let clean = sanitize_tool_id(id);
                if let Some(obj) = message.as_object_mut() {
                    obj.insert("tool_call_id".into(), Value::String(clean));
                }
            }
        }

        let role = role_of(message).to_string();
        let name = name_of(message).unwrap_or("").to_string();

        if role == "system" && name == "example_assistant" {
            if !names.char_name.is_empty()
                && !content_starts_with(message, &format!("{}: ", names.char_name))
                && !message
                    .get("content")
                    .and_then(|c| c.as_str())
                    .is_some_and(|c| names.starts_with_group_name(c))
            {
                prefix_content(message, &format!("{}: ", names.char_name));
            }
            if let Some(obj) = message.as_object_mut() {
                obj.remove("name");
            }
        }

        if role == "system" && name == "example_user" {
            if !names.user_name.is_empty()
                && !content_starts_with(message, &format!("{}: ", names.user_name))
            {
                prefix_content(message, &format!("{}: ", names.user_name));
            }
            if let Some(obj) = message.as_object_mut() {
                obj.remove("name");
            }
        }

        if !name.is_empty()
            && role != "system"
            && !content_starts_with(message, &format!("{name}: "))
        {
            prefix_content(message, &format!("{name}: "));
            if let Some(obj) = message.as_object_mut() {
                obj.remove("name");
            }
        }
    }

    // A user turn right after a tool turn is folded into the previous user turn.
    let mut rerun = true;
    while rerun {
        rerun = false;
        let mut i = 0;
        while i + 1 < messages.len() {
            if role_of(&messages[i]) == "tool" && role_of(&messages[i + 1]) == "user" {
                let last_user = messages[..i]
                    .iter()
                    .rposition(|m| role_of(m) == "user" && m.get("content").is_some());
                if let Some(last_user) = last_user {
                    let incoming = js_str(messages[i + 1].get("content"));
                    let existing = js_str(messages[last_user].get("content"));
                    if let Some(obj) = messages[last_user].as_object_mut() {
                        obj.insert(
                            "content".into(),
                            Value::String(format!("{existing}\n\n{incoming}")),
                        );
                    }
                    messages.remove(i + 1);
                    rerun = true;
                    continue;
                }
            }
            i += 1;
        }
    }

    // A system turn right after an assistant turn becomes a user turn.
    for i in 0..messages.len().saturating_sub(1) {
        if role_of(&messages[i]) == "assistant" && role_of(&messages[i + 1]) == "system" {
            if let Some(obj) = messages[i + 1].as_object_mut() {
                obj.insert("role".into(), json!("user"));
            }
        }
    }

    Value::Array(messages)
}

// ---------------------------------------------------------------------------
// xAI
// ---------------------------------------------------------------------------

/// Mirror of `convertXAIMessages`: name-prefix rules, then drop `name`.
pub fn convert_xai_messages(messages: &Value, names: &PromptNames) -> Value {
    let Some(input) = messages.as_array() else {
        return json!([]);
    };
    let mut messages: Vec<Value> = input.clone();

    for message in messages.iter_mut() {
        let role = role_of(message).to_string();
        let name = name_of(message).unwrap_or("").to_string();
        if name.is_empty() || role == "user" {
            continue;
        }

        let char_prefix_needed = !names.char_name.is_empty()
            && !content_starts_with(message, &format!("{}: ", names.char_name))
            && !message
                .get("content")
                .and_then(|c| c.as_str())
                .is_some_and(|c| names.starts_with_group_name(c));
        let user_prefix_needed = !names.user_name.is_empty()
            && !content_starts_with(message, &format!("{}: ", names.user_name));

        // An assistant turn and an `example_assistant` system turn prefix the
        // same way, which is why upstream has two identical arms.
        let is_char_turn = role == "assistant" || (role == "system" && name == "example_assistant");
        let prefix = if is_char_turn && char_prefix_needed {
            Some(names.char_name.clone())
        } else if role == "system" && name == "example_user" && user_prefix_needed {
            Some(names.user_name.clone())
        } else {
            None
        };

        if let Some(prefix) = prefix {
            prefix_content(message, &format!("{prefix}: "));
        }
        if let Some(obj) = message.as_object_mut() {
            obj.remove("name");
        }
    }

    Value::Array(messages)
}

// ---------------------------------------------------------------------------
// Cohere
// ---------------------------------------------------------------------------

/// Mirror of `convertCohereMessages`: no `name` support, tool calls borrow the
/// preceding assistant content as their primer.
pub fn convert_cohere_messages(messages: &Value, names: &PromptNames, placeholder: &str) -> Value {
    let Some(input) = messages.as_array() else {
        return json!([]);
    };
    let mut messages: Vec<Value> = input.clone();

    if messages.is_empty() {
        messages.push(json!({ "role": "user", "content": placeholder }));
    }

    let mut index = 0;
    while index < messages.len() {
        if messages[index]
            .get("tool_calls")
            .is_some_and(|t| t.is_array())
        {
            let previous_is_assistant = index > 0 && role_of(&messages[index - 1]) == "assistant";
            if previous_is_assistant {
                let borrowed = messages[index - 1]
                    .get("content")
                    .cloned()
                    .unwrap_or(Value::Null);
                if let Some(obj) = messages[index].as_object_mut() {
                    obj.insert("content".into(), borrowed);
                }
                messages.remove(index - 1);
                index = index.saturating_sub(1);
            } else {
                let names_joined = messages[index]
                    .get("tool_calls")
                    .and_then(|t| t.as_array())
                    .map(|calls| {
                        calls
                            .iter()
                            .map(|tc| {
                                tc.pointer("/function/name")
                                    .and_then(|n| n.as_str())
                                    .unwrap_or("")
                                    .to_string()
                            })
                            .collect::<Vec<_>>()
                            .join(", ")
                    })
                    .unwrap_or_default();
                if let Some(obj) = messages[index].as_object_mut() {
                    obj.insert(
                        "content".into(),
                        Value::String(format!("I'm going to call a tool for that: {names_joined}")),
                    );
                }
            }
        }

        let role = role_of(&messages[index]).to_string();
        if let Some(name) = name_of(&messages[index]).map(str::to_string) {
            if role == "system" && name == "example_assistant" {
                let already =
                    content_starts_with(&messages[index], &format!("{}: ", names.char_name))
                        || messages[index]
                            .get("content")
                            .and_then(|c| c.as_str())
                            .is_some_and(|c| names.starts_with_group_name(c));
                if !names.char_name.is_empty() && !already {
                    prefix_content(&mut messages[index], &format!("{}: ", names.char_name));
                }
            }
            if role == "system"
                && name == "example_user"
                && !names.user_name.is_empty()
                && !content_starts_with(&messages[index], &format!("{}: ", names.user_name))
            {
                prefix_content(&mut messages[index], &format!("{}: ", names.user_name));
            }
            if role != "system" && !content_starts_with(&messages[index], &format!("{name}: ")) {
                prefix_content(&mut messages[index], &format!("{name}: "));
            }
            if let Some(obj) = messages[index].as_object_mut() {
                obj.remove("name");
            }
        }

        index += 1;
    }

    Value::Array(messages)
}

// ---------------------------------------------------------------------------
// Gemini / Vertex AI
// ---------------------------------------------------------------------------

/// `GEMINI_MEDIA_RESOLUTION` — only `low` and `high` map; `auto` stays unset.
fn gemini_media_resolution(detail: Option<&str>) -> Option<&'static str> {
    match detail {
        Some("low") => Some("media_resolution_low"),
        Some("high") => Some("media_resolution_high"),
        _ => None,
    }
}

/// Whether the model rejects a trailing model turn (Gemini 3.6/3.7 flash and
/// 3.5 flash-lite).
fn gemini_no_prefill_model(model: &str) -> bool {
    model.contains("gemini-3.6-flash")
        || model.contains("gemini-3.7-flash")
        || model.contains("gemini-3.8-flash")
        || model.contains("gemini-3.9-flash")
        || model.contains("gemini-3.5-flash-lite")
}

/// Mirror of `convertGooglePrompt`.
///
/// Returns `(contents, system_instruction)`; the latter is
/// `{"parts":[{"text":...}]}` and is only sent when it has parts.
pub fn convert_google_prompt(
    messages: &Value,
    model: &str,
    use_sys_prompt: bool,
    names: &PromptNames,
    enable_thought_signatures: bool,
) -> (Value, Value) {
    let Some(input) = messages.as_array() else {
        return (json!([]), json!({ "parts": [] }));
    };
    let mut messages: Vec<Value> = input.clone();
    let mut sys_prompt: Vec<String> = Vec::new();

    if use_sys_prompt {
        while messages.len() > 1 && role_of(&messages[0]) == "system" {
            let first = &mut messages[0];
            if !names.user_name.is_empty()
                && name_of(first) == Some("example_user")
                && !content_starts_with(first, &format!("{}: ", names.user_name))
            {
                prefix_content(first, &format!("{}: ", names.user_name));
            }
            if !names.char_name.is_empty() && name_of(first) == Some("example_assistant") {
                let already = content_starts_with(first, &format!("{}: ", names.char_name))
                    || first
                        .get("content")
                        .and_then(|c| c.as_str())
                        .is_some_and(|c| names.starts_with_group_name(c));
                if !already {
                    prefix_content(first, &format!("{}: ", names.char_name));
                }
            }
            sys_prompt.push(js_str(messages[0].get("content")));
            messages.remove(0);
        }
    }

    let system_instruction = json!({
        "parts": sys_prompt.iter().map(|t| json!({ "text": t })).collect::<Vec<_>>()
    });

    let no_prefill = gemini_no_prefill_model(model);
    let is_gemini_3 = model.contains("gemini-3");
    let is_gemini_25 = model.contains("gemini-2.5");
    let is_image_model = model.contains("-image");
    // `toolNameMap` maps a tool_call_id to the function name, populated as
    // tool calls are walked.
    let mut tool_name_map: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();

    let mut contents: Vec<Value> = Vec::new();

    let message_count = messages.len();
    for (index, message) in messages.iter_mut().enumerate() {
        // Role mapping.
        let role = match role_of(message) {
            "system" | "tool" => "user".to_string(),
            "assistant" => {
                if no_prefill && index + 1 == message_count {
                    "user".to_string()
                } else {
                    "model".to_string()
                }
            }
            other => other.to_string(),
        };

        // Content → array of parts.
        if !message.get("content").is_some_and(|c| c.is_array()) {
            let has_tool_calls = message
                .get("tool_calls")
                .and_then(|t| t.as_array())
                .is_some_and(|a| !a.is_empty());
            let has_tool_call_id = message
                .get("tool_call_id")
                .and_then(|t| t.as_str())
                .is_some_and(|s| !s.is_empty());
            let part = if has_tool_calls {
                json!({
                    "type": "tool_calls",
                    "tool_calls": message.get("tool_calls").cloned().unwrap_or(Value::Null),
                })
            } else if has_tool_call_id {
                json!({
                    "type": "tool_call_id",
                    "tool_call_id": message.get("tool_call_id").cloned().unwrap_or(Value::Null),
                    "content": js_str(message.get("content")),
                })
            } else {
                json!({ "type": "text", "text": js_str(message.get("content")) })
            };
            if let Some(obj) = message.as_object_mut() {
                obj.insert("content".into(), json!([part]));
            }
        }

        // Name prefixing applies to text parts only.
        if let Some(name) = name_of(message).map(str::to_string) {
            if let Some(parts) = message.get_mut("content").and_then(|c| c.as_array_mut()) {
                for part in parts.iter_mut() {
                    if part.get("type").and_then(|t| t.as_str()) != Some("text") {
                        continue;
                    }
                    let text = part
                        .get("text")
                        .and_then(|t| t.as_str())
                        .unwrap_or("")
                        .to_string();
                    let rewritten = if name == "example_user" {
                        if !names.user_name.is_empty()
                            && !text.starts_with(&format!("{}: ", names.user_name))
                        {
                            Some(format!("{}: {text}", names.user_name))
                        } else {
                            None
                        }
                    } else if name == "example_assistant" {
                        if !names.char_name.is_empty()
                            && !text.starts_with(&format!("{}: ", names.char_name))
                            && !names.starts_with_group_name(&text)
                        {
                            Some(format!("{}: {text}", names.char_name))
                        } else {
                            None
                        }
                    } else if !text.starts_with(&format!("{name}: ")) {
                        Some(format!("{name}: {text}"))
                    } else {
                        None
                    };
                    if let Some(rewritten) = rewritten {
                        if let Some(obj) = part.as_object_mut() {
                            obj.insert("text".into(), Value::String(rewritten));
                        }
                    }
                }
            }
            if let Some(obj) = message.as_object_mut() {
                obj.remove("name");
            }
        }

        // Build Gemini parts.
        let mut parts: Vec<Value> = Vec::new();
        let content_parts = message
            .get("content")
            .and_then(|c| c.as_array())
            .cloned()
            .unwrap_or_default();

        for part in content_parts.iter() {
            let add_data_url_part =
                |url: &str, default_mime: &str, detail: Option<&str>, out: &mut Vec<Value>| {
                    if !url.starts_with("data:") {
                        return;
                    }
                    let mut split = url.splitn(2, ',');
                    let header = split.next().unwrap_or("");
                    let data = split.next().unwrap_or("");
                    let mime = header
                        .split(';')
                        .next()
                        .and_then(|h| h.strip_prefix("data:"))
                        .filter(|m| !m.is_empty())
                        .unwrap_or(default_mime);
                    let mut entry = json!({
                        "inlineData": { "mimeType": mime, "data": data }
                    });
                    if is_gemini_3 {
                        if let Some(level) = gemini_media_resolution(detail) {
                            if let Some(obj) = entry.as_object_mut() {
                                obj.insert("mediaResolution".into(), json!({ "level": level }));
                            }
                        }
                    }
                    out.push(entry);
                };

            match part.get("type").and_then(|t| t.as_str()) {
                Some("text") => parts.push(json!({
                    "text": part.get("text").cloned().unwrap_or(Value::Null)
                })),
                Some("tool_call_id") => {
                    let id = part
                        .get("tool_call_id")
                        .and_then(|t| t.as_str())
                        .unwrap_or("");
                    let name = tool_name_map
                        .get(id)
                        .cloned()
                        .unwrap_or_else(|| "unknown".to_string());
                    let content = part.get("content").cloned().unwrap_or(Value::Null);
                    parts.push(json!({
                        "functionResponse": {
                            "name": name,
                            "response": { "name": name, "content": content },
                        }
                    }));
                }
                Some("tool_calls") => {
                    if let Some(calls) = part.get("tool_calls").and_then(|t| t.as_array()) {
                        for call in calls {
                            let fn_name = call
                                .pointer("/function/name")
                                .and_then(|n| n.as_str())
                                .unwrap_or("")
                                .to_string();
                            let args = call
                                .pointer("/function/arguments")
                                .map(parse_tool_arguments)
                                .filter(|v| !v.is_null())
                                .unwrap_or_else(|| {
                                    call.pointer("/function/arguments")
                                        .cloned()
                                        .unwrap_or(Value::Null)
                                });
                            let mut entry = json!({
                                "functionCall": { "name": fn_name, "args": args }
                            });
                            if let Some(sig) = call.get("signature").filter(|s| !s.is_null()) {
                                if let Some(obj) = entry.as_object_mut() {
                                    obj.insert("thoughtSignature".into(), sig.clone());
                                }
                            }
                            parts.push(entry);
                            if let Some(id) = call.get("id").and_then(|i| i.as_str()) {
                                tool_name_map.insert(id.to_string(), fn_name);
                            }
                        }
                    }
                }
                Some("image_url") => {
                    let url = part
                        .pointer("/image_url/url")
                        .and_then(|u| u.as_str())
                        .unwrap_or("");
                    let detail = part.pointer("/image_url/detail").and_then(|d| d.as_str());
                    add_data_url_part(url, "image/png", detail, &mut parts);
                }
                Some("video_url") => {
                    let url = part
                        .pointer("/video_url/url")
                        .and_then(|u| u.as_str())
                        .unwrap_or("");
                    let detail = part.pointer("/video_url/detail").and_then(|d| d.as_str());
                    add_data_url_part(url, "video/mp4", detail, &mut parts);
                }
                Some("audio_url") => {
                    let url = part
                        .pointer("/audio_url/url")
                        .and_then(|u| u.as_str())
                        .unwrap_or("");
                    add_data_url_part(url, "audio/mpeg", None, &mut parts);
                }
                _ => {}
            }
        }

        // Thought signatures for Gemini 3 / 2.5.
        if is_gemini_3 || is_gemini_25 {
            const SKIP: &str = "skip_thought_signature_validator";
            let text_signature = message
                .get("signature")
                .and_then(|s| s.as_str())
                .map(str::to_string);
            for part in parts.iter_mut() {
                let Some(obj) = part.as_object_mut() else {
                    continue;
                };
                let teki = text_signature.is_some() && obj.get("text").is_some();
                if enable_thought_signatures && teki {
                    obj.insert(
                        "thoughtSignature".into(),
                        Value::String(text_signature.clone().unwrap_or_default()),
                    );
                } else if is_gemini_3 {
                    // Gemini 3 mandates the signature on function calls.
                    if obj.contains_key("functionCall") && !obj.contains_key("thoughtSignature") {
                        obj.insert("thoughtSignature".into(), json!(SKIP));
                    }
                    if is_image_model && role == "model" {
                        let has_text = obj.get("text").is_some();
                        let has_inline = obj.contains_key("inlineData");
                        if has_text || has_inline {
                            obj.insert("thoughtSignature".into(), json!(SKIP));
                        }
                    }
                }
            }
        }

        // Merge into the previous turn when the role repeats.
        let merges = index > 0
            && contents
                .last()
                .and_then(|c| c.get("role"))
                .and_then(|r| r.as_str())
                == Some(role.as_str());
        if merges {
            let Some(dst) = contents.last_mut() else {
                continue;
            };
            for part in parts.into_iter() {
                let is_text = part.get("text").is_some();
                if is_text {
                    let dst_parts = dst.get_mut("parts").and_then(|p| p.as_array_mut());
                    let mut appended = false;
                    if let Some(dst_parts) = dst_parts {
                        if let Some(text_part) =
                            dst_parts.iter_mut().find(|p| p.get("text").is_some())
                        {
                            let existing = text_part
                                .get("text")
                                .and_then(|t| t.as_str())
                                .unwrap_or("")
                                .to_string();
                            let incoming = part
                                .get("text")
                                .and_then(|t| t.as_str())
                                .unwrap_or("")
                                .to_string();
                            if let Some(obj) = text_part.as_object_mut() {
                                obj.insert(
                                    "text".into(),
                                    Value::String(format!("{existing}\n\n{incoming}")),
                                );
                            }
                            appended = true;
                        }
                    }
                    if !appended {
                        if let Some(dst_parts) = dst.get_mut("parts").and_then(|p| p.as_array_mut())
                        {
                            dst_parts.push(part);
                        }
                    }
                } else {
                    let keep = part.get("inlineData").is_some()
                        || part.get("functionCall").is_some()
                        || part.get("functionResponse").is_some()
                        || part.get("thoughtSignature").is_some()
                        || part.get("mediaResolution").is_some();
                    if keep {
                        if let Some(dst_parts) = dst.get_mut("parts").and_then(|p| p.as_array_mut())
                        {
                            dst_parts.push(part);
                        }
                    }
                }
            }
        } else {
            contents.push(json!({ "role": role, "parts": parts }));
        }
    }

    (Value::Array(contents), system_instruction)
}

// ---------------------------------------------------------------------------
// Reasoning budgets (src/prompt-converters.js)
// ---------------------------------------------------------------------------

/// `REASONING_EFFORT` values arrive as strings; `None` means absent.
pub fn claude_budget_tokens(
    max_tokens: f64,
    reasoning_effort: Option<&str>,
    stream: bool,
    is_adaptive_model: bool,
) -> Option<Value> {
    let effort = reasoning_effort.unwrap_or("");
    if is_adaptive_model {
        return match effort {
            "min" | "low" => Some(json!("low")),
            "medium" => Some(json!("medium")),
            "high" => Some(json!("high")),
            "max" => Some(json!("max")),
            _ => None,
        };
    }

    // Only `auto` means "no budget". An absent/unknown effort falls through
    // upstream's switch with `budgetTokens = 0`, which the clamp below raises
    // to the 1024 minimum — so it always produces a budget.
    if effort == "auto" {
        return None;
    }
    let mut budget = match effort {
        "min" => 1024.0,
        "low" => (max_tokens * 0.1).floor(),
        "medium" => (max_tokens * 0.25).floor(),
        "high" => (max_tokens * 0.5).floor(),
        "max" => (max_tokens * 0.95).floor(),
        _ => 0.0,
    };
    budget = budget.max(1024.0);
    if !stream {
        budget = budget.min(21333.0);
    }
    Some(json!(budget))
}

fn google_flash_budget(max_tokens: f64, effort: &str) -> Option<f64> {
    match effort {
        "auto" => return None,
        "min" => return Some(0.0),
        _ => {}
    }
    let budget = match effort {
        "low" => (max_tokens * 0.1).floor(),
        "medium" => (max_tokens * 0.25).floor(),
        "high" => (max_tokens * 0.5).floor(),
        "max" => max_tokens,
        _ => 0.0,
    };
    Some(budget.min(24576.0))
}

fn google_flash_lite_budget(max_tokens: f64, effort: &str) -> Option<f64> {
    match effort {
        "auto" => return None,
        "min" => return Some(0.0),
        _ => {}
    }
    let budget = match effort {
        "low" => (max_tokens * 0.1).floor(),
        "medium" => (max_tokens * 0.25).floor(),
        "high" => (max_tokens * 0.5).floor(),
        "max" => max_tokens,
        _ => 0.0,
    };
    Some(budget.clamp(512.0, 24576.0))
}

fn google_pro_budget(max_tokens: f64, effort: &str) -> Option<f64> {
    let budget = match effort {
        "auto" => return None,
        "min" => 128.0,
        "low" => (max_tokens * 0.1).floor(),
        "medium" => (max_tokens * 0.25).floor(),
        "high" => (max_tokens * 0.5).floor(),
        "max" => max_tokens,
        _ => 0.0,
    };
    Some(budget.clamp(128.0, 32768.0))
}

/// Mirror of `calculateGoogleBudgetTokens`. Returns either a numeric budget or
/// an effort level string for Gemini 3.
pub fn google_budget_tokens(
    max_tokens: f64,
    reasoning_effort: Option<&str>,
    model: &str,
) -> Option<Value> {
    let effort = reasoning_effort.unwrap_or("");

    let is_gemini3_pro = model.contains("gemini-3") && model.contains("-pro");
    let is_gemini3_flash = model.contains("gemini-3") && model.contains("-flash");

    if is_gemini3_pro {
        return match effort {
            "auto" => None,
            "min" | "low" | "medium" => Some(json!("low")),
            "high" | "max" => Some(json!("high")),
            _ => None,
        };
    }

    if is_gemini3_flash {
        let no_minimal = model.contains("gemini-3.7-flash")
            || model.contains("gemini-3.8-flash")
            || model.contains("gemini-3.9-flash");
        return match effort {
            "auto" => None,
            "min" => Some(json!(if no_minimal { "low" } else { "minimal" })),
            "low" => Some(json!("low")),
            "medium" => Some(json!("medium")),
            "high" | "max" => Some(json!("high")),
            _ => None,
        };
    }

    let numeric = if model.contains("flash-lite") {
        google_flash_lite_budget(max_tokens, effort)
    } else if model.contains("flash") {
        google_flash_budget(max_tokens, effort)
    } else if model.contains("pro") {
        google_pro_budget(max_tokens, effort)
    } else {
        None
    };

    numeric.map(|b| json!(b))
}
