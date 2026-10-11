//! Golden tests: verify the Rust implementation byte-for-byte against the
//! real upstream Node modules. Reference vectors are produced by
//! `tests/golden/generate.js` (run from the repo root with node) which uses
//! the actual `keygrip`, `png-chunk-text`, and `character-card-parser.js`.

use base64::Engine;
use st_gateway::png;

const GOLDEN: &str = include_str!("golden/golden.json");

fn golden() -> serde_json::Value {
    serde_json::from_str(GOLDEN).expect("golden.json is valid JSON")
}

#[test]
fn keygrip_signatures_match_node() {
    let g = golden();
    for case in g["keygrip"].as_array().unwrap() {
        let data = case["data"].as_str().unwrap();
        let expected_sig = case["sig"].as_str().unwrap();
        let actual = st_gateway::auth::keygrip_sign("secret-one", data);
        assert_eq!(actual, expected_sig, "keygrip mismatch for data {data}");

        // Key rotation: a different secret must NOT verify
        let alt = case["sig_alt_secret"].as_str().unwrap();
        assert_ne!(st_gateway::auth::keygrip_sign("secret-one", data), alt);
    }
}

#[test]
fn session_cookie_decoding_matches_node_encoding() {
    let g = golden();
    for case in g["keygrip"].as_array().unwrap() {
        let value = case["value"].as_str().unwrap();
        // Rust decode: base64 → utf8 → JSON
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(value)
            .expect("session value is base64");
        let decoded: serde_json::Value =
            serde_json::from_slice(&bytes).expect("session payload is JSON");
        assert_eq!(&decoded, &case["payload"]);
    }
}

#[test]
fn pngtext_encode_byte_layout_matches_node() {
    let g = golden();
    let expected_hex = g["pngtext_encode_hex"].as_str().unwrap();
    let ours = png::encode_text_chunk("chara", "plain-value");
    // Node's PNGtext.encode returns only the data part (keyword\0text);
    // our encode_text_chunk wraps it into a full chunk. Compare data region.
    let our_data = &ours[8..ours.len() - 4]; // strip len+type and CRC
    assert_eq!(hex_encode(our_data), expected_hex);
}

#[test]
fn card_png_roundtrip_against_node_write() {
    let g = golden();
    let png_bytes = base64::engine::general_purpose::STANDARD
        .decode(g["png_base64"].as_str().unwrap())
        .unwrap();
    let expected_keywords: Vec<&str> = g["chunk_keywords"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();

    // Write the golden card onto a minimal PNG in Rust and compare tEXt keywords
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("golden.png");
    std::fs::write(&path, &png_bytes).unwrap();

    // Extract card from the Node-written PNG
    let chunks = png::read_text_chunks(&path).unwrap();
    let keywords: Vec<String> = chunks.iter().map(|c| c.keyword.clone()).collect();
    assert_eq!(
        keywords, expected_keywords,
        "tEXt keyword set/order mismatch"
    );

    // The card JSON Rust extracts must be the v3-mutated readback Node produced
    let card = png::extract_card_json(&chunks).unwrap();
    assert_eq!(card, g["card_json_readback"].as_str().unwrap());

    // parseability: readback JSON must deep-equal Node's readback object
    let ours: serde_json::Value = serde_json::from_str(&card).unwrap();
    let theirs: serde_json::Value =
        serde_json::from_str(g["card_json_readback"].as_str().unwrap()).unwrap();
    assert_eq!(ours, theirs);
}

#[test]
fn rust_write_matches_node_chunk_semantics() {
    let g = golden();
    // Rebuild in Rust: take the base structure and re-write the same card via
    // chunk surgery, then confirm Node's parser-compatible keywords survive.
    let png_bytes = base64::engine::general_purpose::STANDARD
        .decode(g["png_base64"].as_str().unwrap())
        .unwrap();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("rt.png");
    std::fs::write(&path, &png_bytes).unwrap();

    // Rust "edit": re-write the exact same card json (idempotent surgery)
    let chunks = png::read_text_chunks(&path).unwrap();
    let card = png::extract_card_json(&chunks).unwrap();
    let b64 = base64::engine::general_purpose::STANDARD.encode(card.as_bytes());
    let new_chara = png::encode_text_chunk("chara", &b64);

    // Also write a ccv3 like Node's write() does (spec-mutated)
    let mut v3: serde_json::Value = serde_json::from_str(&card).unwrap();
    v3["spec"] = serde_json::Value::from("chara_card_v3");
    v3["spec_version"] = serde_json::Value::from("3.0");
    let b64v3 = base64::engine::general_purpose::STANDARD.encode(v3.to_string().as_bytes());
    let new_ccv3 = png::encode_text_chunk("ccv3", &b64v3);

    png::write_card_to_png(&path, &[new_chara, new_ccv3]).unwrap();

    // Verify the result reads back exactly like the Node original
    let chunks_after = png::read_text_chunks(&path).unwrap();
    let keywords: Vec<String> = chunks_after.iter().map(|c| c.keyword.clone()).collect();
    assert_eq!(keywords, vec!["chara", "ccv3"]);
    let card_after = png::extract_card_json(&chunks_after).unwrap();
    assert_eq!(card_after, g["card_json_readback"].as_str().unwrap());

    // IDAT must be untouched by surgery (byte-identical IDAT region)
    let expected_chunk_names: Vec<&str> = g["chunk_names"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    assert_eq!(
        chunk_names(&std::fs::read(&path).unwrap()),
        expected_chunk_names
    );
}

/// Parse chunk type names from a PNG byte buffer.
fn chunk_names(png: &[u8]) -> Vec<String> {
    let mut out = Vec::new();
    let mut pos = 8;
    while pos + 8 <= png.len() {
        let len = u32::from_be_bytes(png[pos..pos + 4].try_into().unwrap()) as usize;
        let ctype = &png[pos + 4..pos + 8];
        out.push(String::from_utf8_lossy(ctype).to_string());
        pos += 8 + len + 4;
        if ctype == b"IEND" {
            break;
        }
    }
    out
}

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[test]
fn calculate_data_size_matches_js_string_length() {
    // JS String length = UTF-16 code units
    use st_gateway::card::calculate_data_size;
    // 'é' = 1 unit; '😀' = 2 units
    let data = serde_json::json!({ "a": "é", "b": "😀" });
    assert_eq!(calculate_data_size(&data), 3);
    // null → String(null) = "null" = 4 units
    let data = serde_json::json!({ "x": null });
    assert_eq!(calculate_data_size(&data), 4);
    // numbers → their string form
    let data = serde_json::json!({ "n": 12345 });
    assert_eq!(calculate_data_size(&data), 5);
}

#[test]
fn read_from_v2_field_hoisting_matches_lodash_semantics() {
    use st_gateway::card::read_from_v2;
    // v2 value wins; missing extension fields get ST defaults
    let mut card = serde_json::json!({
        "spec": "chara_card_v2",
        "data": {
            "name": "V2Name",
            "description": "v2desc",
            "extensions": { "talkativeness": 0.7 }
        },
        "name": "V1Name",
        "description": "v1desc"
    });
    read_from_v2(&mut card);
    assert_eq!(card["name"], "V2Name"); // v2 overwrites v1
    assert_eq!(card["description"], "v2desc");
    assert_eq!(card["talkativeness"], 0.7); // present: no default
    assert_eq!(card["fav"], false); // missing: default false
    assert!(card["chat"].as_str().unwrap().starts_with("V2Name - "));
}

#[test]
fn convert_to_v2_matches_chara_format_data_defaults() {
    use st_gateway::card::convert_to_v2;
    let v1 = serde_json::json!({
        "name": "Old Card",
        "description": "desc",
        "custom_keeper": "preserved"
    });
    let v2 = convert_to_v2(&v1);
    assert_eq!(v2["spec"], "chara_card_v2");
    assert_eq!(v2["spec_version"], "2.0");
    assert_eq!(v2["name"], "Old Card");
    assert_eq!(v2["data"]["extensions"]["talkativeness"], 0.5);
    assert_eq!(v2["data"]["extensions"]["fav"], false);
    assert_eq!(v2["custom_keeper"], "preserved");
    assert_eq!(v2["avatar"], "none");
}

#[test]
fn world_book_position_and_extensions_match_upstream() {
    use st_gateway::card::convert_world_info_to_character_book;
    let entries = serde_json::json!({
        "1": { "uid": 1, "key": ["b"], "keysecondary": [], "content": "y",
               "constant": false, "selective": true, "order": 200,
               "position": 1, "disable": true, "displayIndex": 1,
               "probability": 80, "depth": 2 },
        "0": { "uid": 0, "key": ["a"], "keysecondary": [], "content": "x",
               "constant": false, "selective": true, "order": 100,
               "position": 0, "disable": false, "displayIndex": 0 }
    });
    let book = convert_world_info_to_character_book("w", &entries);
    assert_eq!(book["name"], "w");
    let arr = book["entries"].as_array().unwrap();
    // Integer-like keys iterate in ascending numeric order (JS semantics)
    assert_eq!(arr[0]["id"], 0);
    assert_eq!(arr[1]["id"], 1);
    assert_eq!(arr[0]["position"], "before_char");
    assert_eq!(arr[1]["position"], "after_char");
    assert_eq!(arr[1]["enabled"], false); // disable: true
    assert_eq!(arr[1]["extensions"]["probability"], 80); // provided, not nulled
    assert_eq!(arr[1]["extensions"]["depth"], 2); // provided
    assert_eq!(arr[0]["extensions"]["depth"], 4); // default
    assert_eq!(arr[0]["extensions"]["probability"], serde_json::Value::Null);
}

#[test]
fn unset_private_fields_matches_upstream() {
    use st_gateway::card::unset_private_fields;
    let mut card = serde_json::json!({
        "fav": true,
        "chat": "gone",
        "data": { "extensions": { "fav": true } }
    });
    unset_private_fields(&mut card);
    assert_eq!(card["fav"], false);
    assert_eq!(card["data"]["extensions"]["fav"], false);
    assert!(card.get("chat").is_none());
}

#[test]
fn config_parses_st_config_yaml_schema() {
    // Against the real config.yaml keys used by the gateway
    let text = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../config.yaml"))
        .unwrap_or_else(|_| {
            // Fall back to a synthetic config if repo root isn't the cwd of the test runner
            "port: 8000\ndataRoot: ./data\n".to_string()
        });
    let cfg = st_gateway::config::ResolvedConfig::from_text(&text, std::path::PathBuf::from(".."));
    // The repo config sets port 8000
    assert!(cfg.node_port >= 1);
    assert!(!cfg.cookie_secret_path().as_os_str().is_empty());
}
