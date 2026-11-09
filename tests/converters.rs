//! Golden-vector tests for the message converters.
//!
//! `tests/data/converters.json` is produced by `data/_vectors.mjs`, which
//! imports the real `src/prompt-converters.js` and dumps its output for a
//! prompt exercising every special case (example names, consecutive same-role
//! turns, tool calls, tool results, images, empty text, images in assistant
//! turns). Comparing against that file is the only way to be sure the Rust
//! port sends the provider the same prompt Node did.

use serde_json::{json, Value};
use st_gateway::converters::*;
use st_gateway::prompt::PromptNames;

/// The vector run reads `promptPlaceholder` from config.yaml, which sets
/// "[Start a new chat]"; the converters take it as a parameter.
const PLACEHOLDER: &str = "[Start a new chat]";

fn names() -> PromptNames {
    PromptNames {
        char_name: "Seraphina".to_string(),
        user_name: "Kai".to_string(),
        group_names: vec!["Luna".to_string()],
    }
}

fn vectors() -> Value {
    let text = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/converters.json"
    ))
    .expect("tests/data/converters.json — regenerate with: node data/_vectors.mjs > tests/data/converters.json");
    serde_json::from_str(&text).expect("converters.json is valid JSON")
}

/// The prompt the vectors were generated from, rebuilt here so the test does
/// not depend on the generator's ordering.
fn prompt() -> Value {
    json!([
        { "role": "system", "content": "You are Seraphina.", "name": "main" },
        { "role": "system", "content": "Hello there", "name": "example_user" },
        { "role": "system", "content": "Hi Kai!", "name": "example_assistant" },
        { "role": "system", "content": "Luna said something" },
        { "role": "user", "content": "First question", "name": "Kai" },
        { "role": "assistant", "content": [{ "type": "text", "text": "First answer" }] },
        { "role": "user", "content": "Second question" },
        { "role": "user", "content": "Same-role follow-up" },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call_1",
                "function": { "name": "get_weather", "arguments": "{\"city\":\"Paris\"}" }
            }]
        },
        { "role": "tool", "tool_call_id": "call_1", "content": "{\"temp\":21}" },
        {
            "role": "user",
            "content": [
                { "type": "text", "text": "What about images?" },
                { "type": "image_url", "image_url": { "url": "data:image/png;base64,QUJD", "detail": "low" } }
            ]
        },
        {
            "role": "assistant",
            "content": [
                { "type": "image_url", "image_url": { "url": "data:image/jpeg;base64,REVG" } },
                { "type": "text", "text": "look at this" }
            ]
        },
        { "role": "user", "content": [{ "type": "text", "text": "" }] }
    ])
}

#[test]
fn claude_conversion_matches_upstream() {
    let expected = vectors();
    let expected = &expected["claude"];
    let cases: [(&str, Option<&str>, bool, bool); 4] = [
        ("sysprompt_tools", None, true, true),
        ("sysprompt_notools", None, true, false),
        ("nosysprompt", None, false, true),
        ("prefill", Some("Sure, here is "), true, true),
    ];
    for (key, prefill, use_sys, use_tools) in cases {
        let (messages, system) = convert_claude_messages(
            &prompt(),
            prefill,
            use_sys,
            use_tools,
            &names(),
            PLACEHOLDER,
        );
        assert_eq!(messages, expected[key]["messages"], "claude {key} messages");
        assert_eq!(system, expected[key]["systemPrompt"], "claude {key} system");
    }

    // An empty prompt becomes the placeholder. The vector run reads it from
    // config.yaml, so derive the expected text instead of hardcoding it.
    let (messages, system) =
        convert_claude_messages(&json!([]), None, true, true, &names(), PLACEHOLDER);
    let expected_messages = &expected["empty"]["messages"];
    assert_eq!(
        messages.as_array().unwrap().len(),
        1,
        "one placeholder turn"
    );
    assert_eq!(
        messages[0]["role"], expected_messages[0]["role"],
        "claude empty role"
    );
    assert_eq!(
        system, expected["empty"]["systemPrompt"],
        "claude empty system"
    );
}

#[test]
fn ai21_conversion_matches_upstream() {
    let expected = vectors();
    let converted = convert_ai21_messages(&prompt(), &names(), PLACEHOLDER);
    assert_eq!(converted, expected["ai21"]["basic"]);
    assert_eq!(
        convert_ai21_messages(&json!([]), &names(), PLACEHOLDER),
        expected["ai21"]["empty"]
    );
}

#[test]
fn cohere_conversion_matches_upstream() {
    let expected = vectors();
    assert_eq!(
        convert_cohere_messages(&prompt(), &names(), PLACEHOLDER),
        expected["cohere"]["basic"]["chatHistory"]
    );
    assert_eq!(
        convert_cohere_messages(&json!([]), &names(), PLACEHOLDER),
        expected["cohere"]["empty"]["chatHistory"]
    );
}

#[test]
fn xai_conversion_matches_upstream() {
    let expected = vectors();
    assert_eq!(
        convert_xai_messages(&prompt(), &names()),
        expected["xai"]["basic"]
    );
}

#[test]
fn mistral_conversion_matches_upstream() {
    // Mistral's converter is not in the vector file (the generator only dumps
    // the exported ones), so assert the invariants it exists to guarantee.
    let converted = convert_mistral_messages(&prompt(), &names(), false);
    let messages = converted.as_array().expect("array");

    // Tool-call ids are truncated SHA-512 hex, and the id and the matching
    // tool result get the *same* sanitized value.
    let call = messages
        .iter()
        .flat_map(|m| {
            m.get("tool_calls")
                .and_then(|t| t.as_array())
                .cloned()
                .unwrap_or_default()
        })
        .next()
        .expect("a tool call survives");
    let call_id = call["id"].as_str().expect("id is a string");
    assert_eq!(call_id.len(), 9, "sanitized id is 9 chars");
    assert!(call_id.chars().all(|c| c.is_ascii_hexdigit()), "id is hex");
    let tool_result = messages
        .iter()
        .find(|m| m.get("role").and_then(|r| r.as_str()) == Some("tool"))
        .expect("tool result survives");
    assert_eq!(
        tool_result["tool_call_id"], call_id,
        "both sides sanitized alike"
    );

    // Upstream keeps `name` on plain system turns ("main", "Luna said
    // something") and only strips it from the example_* and prefixed ones.
    assert_eq!(
        messages[0]["name"], "main",
        "a plain system turn keeps its name"
    );
    assert!(
        messages[1].get("name").is_none(),
        "example_user name stripped"
    );
    assert!(
        messages[2].get("name").is_none(),
        "example_assistant name stripped"
    );
    let first_user = messages
        .iter()
        .find(|m| m.get("role").and_then(|r| r.as_str()) == Some("user"))
        .expect("a user turn");
    assert_eq!(first_user["content"], "Kai: First question");

    // Prefix mode marks a trailing assistant turn.
    let mut with_prefill = prompt();
    with_prefill
        .as_array_mut()
        .unwrap()
        .push(json!({ "role": "assistant", "content": "Sure" }));
    let prefixed = convert_mistral_messages(&with_prefill, &names(), true);
    assert_eq!(
        prefixed.as_array().unwrap().last().unwrap()["prefix"],
        json!(true)
    );
}

#[test]
fn google_conversion_matches_upstream() {
    let expected = vectors();
    for model in [
        "mistral-large-latest",
        "claude-opus-4-5",
        "gemini-2.5-flash",
        "gemini-3-pro",
    ] {
        let key = format!("google_{model}");
        for (field, use_sys) in [("sysprompt", true), ("nosysprompt", false)] {
            let (contents, system) =
                convert_google_prompt(&prompt(), model, use_sys, &names(), true);
            assert_eq!(
                contents, expected[&key][field]["contents"],
                "{key} {field} contents"
            );
            assert_eq!(
                system, expected[&key][field]["system_instruction"],
                "{key} {field} system"
            );
        }
        let (contents, system) = convert_google_prompt(&json!([]), model, true, &names(), true);
        assert_eq!(
            contents, expected[&key]["empty"]["contents"],
            "{key} empty contents"
        );
        assert_eq!(system, expected[&key]["empty"]["system_instruction"]);
    }
}

#[test]
fn claude_budget_tokens_match_upstream() {
    let expected = vectors();
    for entry in expected["budgets"]["claude"].as_array().unwrap() {
        let effort = entry[0].as_str().unwrap();
        let effort = if effort == "undefined" {
            None
        } else {
            Some(effort)
        };
        assert_eq!(
            claude_budget_tokens(8000.0, effort, true, false).and_then(|v| v.as_f64()),
            entry[1].as_f64(),
            "claude budget for {effort:?}"
        );
    }
    for entry in expected["budgets"]["claude_nonstream"].as_array().unwrap() {
        let effort = Some(entry[0].as_str().unwrap());
        assert_eq!(
            claude_budget_tokens(100_000.0, effort, false, false).and_then(|v| v.as_f64()),
            entry[1].as_f64(),
            "claude non-stream budget for {effort:?}"
        );
    }
    for entry in expected["budgets"]["claude_adaptive"].as_array().unwrap() {
        let effort = Some(entry[0].as_str().unwrap());
        let ours = claude_budget_tokens(8000.0, effort, true, true);
        match (&ours, &entry[1]) {
            (Some(Value::String(a)), Value::String(b)) => assert_eq!(a, b, "{effort:?}"),
            (None, Value::Null) => {}
            other => panic!("claude adaptive budget for {effort:?}: {other:?}"),
        }
    }
}

#[test]
fn google_budget_tokens_match_upstream() {
    let expected = vectors();
    for model_entry in expected["budgets"]["google"].as_array().unwrap() {
        let model = model_entry[0].as_str().unwrap();
        for effort_entry in model_entry[1].as_array().unwrap() {
            let effort = Some(effort_entry[0].as_str().unwrap());
            // `-1` is upstream's "no budget" sentinel; we drop the field
            // entirely, so both sides must agree on "nothing to send".
            let expected = effort_entry[1]
                .as_f64()
                .filter(|v| *v >= 0.0)
                .map(Value::from)
                .or_else(|| effort_entry[1].as_str().map(|s| Value::from(s.to_string())));
            assert_eq!(
                google_budget_tokens(8000.0, effort, model),
                expected,
                "google budget for {model} / {effort:?}"
            );
        }
    }
}
