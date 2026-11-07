//! Golden tests for prompt post-processing against the REAL upstream
//! `src/prompt-converters.js`. Reference vectors come from
//! `tests/golden/generate_prompt.js` (run from the SillyTavern repo root with
//! node). Every vector is compared as exact JSON, so upstream quirks
//! (double prefixes on named messages, tool-squash rules, placeholder
//! insertion index) are locked in.

use st_gateway::prompt::{
    add_open_router_signatures, convert_text_completion_prompt, embed_open_router_media,
    post_process_prompt, PromptNames, PromptProcessingType,
};

const GOLDEN: &str = include_str!("golden/golden_prompt.json");

fn golden() -> serde_json::Value {
    serde_json::from_str(GOLDEN).expect("golden_prompt.json is valid JSON")
}

fn names() -> PromptNames {
    PromptNames {
        char_name: "Seraphina".into(),
        user_name: "User".into(),
        group_names: vec!["Alice".into(), "Bob".into()],
    }
}

/// The golden vectors are generated with the repo `config.yaml`, which sets
/// `promptPlaceholder: "[Start a new chat]"`.
const PLACEHOLDER: &str = "[Start a new chat]";

#[test]
fn post_process_prompt_matches_node_for_all_vectors() {
    let g = golden();
    let cases = g["postProcessPrompt"].as_array().unwrap();
    assert!(cases.len() >= 100, "expected full vector matrix");

    let mut checked = 0;
    for case in cases {
        let name = case["case"].as_str().unwrap();
        let type_name = case["type"].as_str().unwrap();
        let expected = &case["output"];
        if expected.get("__error").is_some() {
            continue; // upstream threw; nothing to compare
        }
        let mut messages = case["input"].clone();
        post_process_prompt(
            &mut messages,
            PromptProcessingType::parse(type_name),
            &names(),
            PLACEHOLDER,
        );
        assert_eq!(
            serde_json::to_string(&messages).unwrap(),
            serde_json::to_string(expected).unwrap(),
            "mismatch for case {name:?} / type {type_name:?}"
        );
        checked += 1;
    }
    assert!(checked >= 100, "only {checked} vectors compared");
}

#[test]
fn convert_text_completion_prompt_matches_node() {
    let g = golden();
    for case in g["convertTextCompletionPrompt"].as_array().unwrap() {
        let expected = case["output"].as_str().unwrap();
        assert_eq!(
            convert_text_completion_prompt(&case["input"]),
            expected,
            "mismatch for input {}",
            serde_json::to_string(&case["input"]).unwrap()
        );
    }
}

#[test]
fn embed_open_router_media_matches_node() {
    let g = golden();
    for case in g["embedOpenRouterMedia"].as_array().unwrap() {
        let audio = case["opts"]["audio"].as_bool().unwrap();
        let video = case["opts"]["video"].as_bool().unwrap();
        let mut messages = case["input"].clone();
        embed_open_router_media(&mut messages, audio, video);
        assert_eq!(
            serde_json::to_string(&messages).unwrap(),
            serde_json::to_string(&case["output"]).unwrap(),
            "mismatch for opts audio={audio} video={video}"
        );
    }
}

#[test]
fn add_open_router_signatures_matches_node() {
    let g = golden();
    for case in g["addOpenRouterSignatures"].as_array().unwrap() {
        let model = case["model"].as_str().unwrap();
        let mut messages = case["input"].clone();
        // Upstream default for gemini.thoughtSignatures is true.
        add_open_router_signatures(&mut messages, model, true);
        assert_eq!(
            serde_json::to_string(&messages).unwrap(),
            serde_json::to_string(&case["output"]).unwrap(),
            "mismatch for model {model}"
        );
    }
}
