//! Golden-vector tests for the provider request builders.
//!
//! `tests/data/providers.json` is the *exact* request the upstream Node
//! handler produced, captured with a `node-fetch` stub (see the throwaway
//! harness in this repo's history). Comparing URLs, headers and JSON bodies
//! field-by-field is the only cheap way to know a provider port is faithful —
//! a wrong header name or a dropped `top_k` looks fine locally and fails
//! against the real API.

use serde_json::{json, Value};
use st_gateway::chat_completions::PreparedRequest;
use st_gateway::config::ResolvedConfig;
use st_gateway::prompt::PromptNames;
use st_gateway::providers::*;

fn cfg() -> st_gateway::config::StConfig {
    ResolvedConfig::from_text(
        "dataRoot: ./data\npromptPlaceholder: \"[Start a new chat]\"\n",
        std::path::PathBuf::from("."),
    )
    .st
}

fn names() -> PromptNames {
    PromptNames {
        char_name: "Seraphina".to_string(),
        user_name: "Kai".to_string(),
        group_names: vec![],
    }
}

fn vectors() -> Value {
    let text = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/providers.json"
    ))
    .expect("tests/data/providers.json must exist");
    serde_json::from_str(&text).expect("providers.json is valid JSON")
}

/// `secrets.json` shaped like the capture harness wrote it.
fn secrets() -> Value {
    let keys = [
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
    ];
    let mut doc = serde_json::Map::new();
    for key in keys {
        doc.insert(
            format!("api_key_{key}"),
            json!([{ "id": "x", "value": format!("sk-{key}"), "active": true, "label": "" }]),
        );
    }
    Value::Object(doc)
}

/// The prompt every vector was captured with.
fn base(source: &str, extra: Value) -> Value {
    let mut body = json!({
        "chat_completion_source": source,
        "model": "test-model",
        "messages": [
            { "role": "system", "content": "You are Seraphina." },
            { "role": "user", "content": "Hello", "name": "Kai" },
            { "role": "assistant", "content": "Hi!" },
            { "role": "user", "content": "How are you?" }
        ],
        "temperature": 0.8,
        "top_p": 0.9,
        "top_k": 40,
        "max_tokens": 500,
        "stream": false,
        "presence_penalty": 0.1,
        "frequency_penalty": 0.2,
        "seed": 42,
        "stop": ["STOP"],
        "logprobs": 3,
        "char_name": "Seraphina",
        "user_name": "Kai",
        "group_names": []
    });
    if let (Some(target), Some(extra)) = (body.as_object_mut(), extra.as_object()) {
        for (k, v) in extra {
            target.insert(k.clone(), v.clone());
        }
    }
    body
}

/// Compare a built request against the captured one.
fn assert_matches(case: &str, built: &PreparedRequest, expected: &Value) {
    let expected = &expected["calls"][0];
    assert_eq!(built.url, expected["url"], "{case}: url");

    // Header names are compared case-insensitively; values must be exact.
    let lower = |value: &Value| -> Vec<(String, String)> {
        let mut out: Vec<(String, String)> = value
            .as_object()
            .map(|map| {
                map.iter()
                    .map(|(k, v)| (k.to_ascii_lowercase(), v.as_str().unwrap_or("").to_string()))
                    .collect()
            })
            .unwrap_or_default();
        out.sort();
        out
    };
    let mut ours: Vec<(String, String)> = built
        .headers
        .iter()
        .map(|(k, v)| (k.to_ascii_lowercase(), v.clone()))
        .collect();
    ours.sort();
    assert_eq!(ours, lower(&expected["headers"]), "{case}: headers");

    assert_eq!(
        canonical(&built.body),
        canonical(&expected["body"]),
        "{case}: body"
    );
}

/// JSON bodies are compared structurally: key order does not matter to a
/// provider, but a missing or extra key does.
fn canonical(value: &Value) -> Value {
    value.clone()
}

/// Build the case whose inputs are all "plain" (no extra fields).
fn build_plain(source: &str) -> PreparedRequest {
    let body = base(source, json!({}));
    let c = cfg();
    let n = names();
    let s = secrets();
    match source {
        "claude" => {
            build_claude(&body, &s, &c, &n, "[Start a new chat]")
                .expect("claude builds")
                .prepared
        }
        "ai21" => {
            build_ai21(&body, &s, &n, "[Start a new chat]")
                .expect("ai21 builds")
                .prepared
        }
        "cohere" => {
            build_cohere(&body, &s, &n, "[Start a new chat]")
                .expect("cohere builds")
                .prepared
        }
        other => {
            let flavour =
                OpenAiFlavour::parse(other).unwrap_or_else(|| panic!("{other} is a flavour"));
            build_openai_flavour(flavour, &body, &s, &c, &n, "[Start a new chat]")
                .expect("provider builds")
                .prepared
        }
    }
}

#[test]
fn simple_providers_match_upstream_requests() {
    let expected = vectors();
    for source in [
        "ai21",
        "mistralai",
        "cohere",
        "deepseek",
        "aimlapi",
        "xai",
        "chutes",
        "minimax",
        "electronhub",
    ] {
        assert_matches(source, &build_plain(source), &expected[source]);
    }
}

#[test]
fn claude_matches_upstream_request() {
    let expected = vectors();
    assert_matches("claude", &build_plain("claude"), &expected["claude"]);
}

#[tokio::test]
async fn makersuite_matches_upstream_request() {
    let expected = vectors();
    let body = base("makersuite", json!({}));
    let built = build_gemini(&body, &secrets(), &cfg(), &names(), false)
        .await
        .expect("gemini builds")
        .prepared;
    assert_matches("makersuite", &built, &expected["makersuite"]);
}

#[test]
fn azure_matches_upstream_request() {
    let expected = vectors();
    let body = base(
        "azure_openai",
        json!({
            "azure_base_url": "https://example.openai.azure.com",
            "azure_deployment_name": "my-deploy",
            "azure_api_version": "2024-10-21"
        }),
    );
    let built = build_openai_flavour(
        OpenAiFlavour::AzureOpenAi,
        &body,
        &secrets(),
        &cfg(),
        &names(),
        "[Start a new chat]",
    )
    .expect("azure builds")
    .prepared;
    assert_matches("azure_openai", &built, &expected["azure_openai"]);
}

#[test]
fn claude_thinking_matches_upstream_request() {
    let expected = vectors();
    let body = base(
        "claude",
        json!({
            "model": "claude-opus-4-5",
            "reasoning_effort": "medium",
            "include_reasoning": true,
            "use_sysprompt": true
        }),
    );
    let built = build_claude(&body, &secrets(), &cfg(), &names(), "[Start a new chat]")
        .expect("claude thinking builds")
        .prepared;
    assert_matches("claude_thinking", &built, &expected["claude_thinking"]);
}

#[tokio::test]
async fn makersuite_sysprompt_matches_upstream_request() {
    let expected = vectors();
    let body = base(
        "makersuite",
        json!({
            "model": "gemini-2.5-flash",
            "use_sysprompt": true,
            "enable_web_search": true,
            "json_schema": {
                "name": "out",
                "value": { "type": "object", "properties": { "a": { "type": "string" } } }
            }
        }),
    );
    let built = build_gemini(&body, &secrets(), &cfg(), &names(), false)
        .await
        .expect("gemini sysprompt builds")
        .prepared;
    assert_matches(
        "makersuite_sysprompt",
        &built,
        &expected["makersuite_sysprompt"],
    );
}

#[test]
fn json_schema_variants_match_upstream_requests() {
    let expected = vectors();
    let c = cfg();
    let n = names();
    let s = secrets();

    for (case, source, flavour) in [
        ("cohere_json", "cohere", None),
        ("deepseek_json", "deepseek", Some(OpenAiFlavour::DeepSeek)),
        ("xai_json", "xai", Some(OpenAiFlavour::Xai)),
    ] {
        let extra = match case {
            "cohere_json" => json!({
                "model": "command-r-plus-08-2024",
                "json_schema": { "name": "out", "value": { "type": "object" } }
            }),
            "xai_json" => json!({
                "json_schema": { "name": "out", "value": { "type": "object" } },
                "reasoning_effort": "high"
            }),
            _ => json!({ "json_schema": { "name": "out", "value": { "type": "object" } } }),
        };
        let body = base(source, extra);
        let built = match flavour {
            None => build_cohere(&body, &s, &n, "[Start a new chat]"),
            Some(flavour) => build_openai_flavour(flavour, &body, &s, &c, &n, "[Start a new chat]"),
        }
        .expect("builds")
        .prepared
        .clone();
        assert_matches(case, &built, &expected[case]);
    }
}

#[test]
fn claude_json_schema_matches_upstream_request() {
    let expected = vectors();
    let body = base(
        "claude",
        json!({
            "model": "claude-sonnet-4-5",
            "json_schema": { "name": "out", "description": "d", "value": { "type": "object" } }
        }),
    );
    let built = build_claude(&body, &secrets(), &cfg(), &names(), "[Start a new chat]")
        .expect("claude json builds")
        .prepared;
    assert_matches("claude_json", &built, &expected["claude_json"]);
}
