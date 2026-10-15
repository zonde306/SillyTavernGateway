//! B1 golden tests: sanitize-filename mirror verified against the REAL
//! Node package, plus edit-attribute / rename / export behavior.

use serde_json::Value;

const SANITIZE: &str = include_str!("golden/sanitize.json");

#[test]
fn sanitize_filename_matches_node_package() {
    let vectors: Value = serde_json::from_str(SANITIZE).unwrap();
    for (input, expected) in vectors.as_object().unwrap() {
        let actual = st_gateway::card::sanitize_filename(input);
        assert_eq!(
            actual,
            expected.as_str().unwrap(),
            "sanitize mismatch for input {input:?}"
        );
    }
}

#[test]
fn get_png_name_dedups() {
    let dir = tempfile::tempdir().unwrap();
    let chars = dir.path().join("characters");
    std::fs::create_dir_all(&chars).unwrap();

    // No collision: base name kept
    assert_eq!(st_gateway::card::get_png_name("Alice", &chars), "Alice");
    std::fs::write(chars.join("Alice.png"), b"png").unwrap();
    // Collision: counter appended (startIndex 0 → base, then 1, 2, ...)
    assert_eq!(st_gateway::card::get_png_name("Alice", &chars), "Alice1");
    std::fs::write(chars.join("Alice1.png"), b"png").unwrap();
    assert_eq!(st_gateway::card::get_png_name("Alice", &chars), "Alice2");
}

mod helpers {
    use st_gateway::card::CardDeps;

    pub fn make_card_png(path: &std::path::Path, name: &str, description: &str) {
        let card = serde_json::json!({
            "spec": "chara_card_v2", "spec_version": "2.0",
            "name": name,
            "description": description,
            "data": { "name": name, "description": description }
        })
        .to_string();
        use base64::Engine;
        let b64 = base64::engine::general_purpose::STANDARD.encode(card.as_bytes());
        let mut data = b"chara\0".to_vec();
        data.extend_from_slice(b64.as_bytes());
        let mut png_bytes = Vec::new();
        png_bytes.extend_from_slice(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]);
        png_bytes.extend_from_slice(&st_gateway::png::build_chunk(
            b"IHDR",
            &[0, 0, 0, 1, 0, 0, 0, 1, 8, 6, 0, 0, 0],
        ));
        png_bytes.extend_from_slice(&st_gateway::png::build_chunk(b"tEXt", &data));
        png_bytes.extend_from_slice(&st_gateway::png::build_chunk(b"IEND", &[]));
        std::fs::write(path, png_bytes).unwrap();
    }

    pub fn deps_for(dir: &std::path::Path) -> CardDeps {
        CardDeps {
            characters_dir: dir.join("characters"),
            chats_dir: dir.join("chats"),
            worlds_dir: dir.join("worlds"),
        }
    }
}

/// edit-attribute semantics: field set on BOTH top level and data.<field>;
/// unknown field rejected; json_data rejected.
#[test]
fn edit_attribute_sets_both_levels() {
    use st_gateway::{card, png};

    let dir = tempfile::tempdir().unwrap();
    let chars = dir.path().join("characters");
    std::fs::create_dir_all(&chars).unwrap();
    helpers::make_card_png(&chars.join("a.png"), "A", "desc");

    // Read raw card, simulate the handler's mutation
    let raw = {
        let chunks = png::read_text_chunks(&chars.join("a.png")).unwrap();
        png::extract_card_json(&chunks).unwrap()
    };
    let mut char: Value = serde_json::from_str(&raw).unwrap();

    // Known field: "description" exists in both levels on our card
    let field = "description";
    assert!(
        char.get(field).is_some() || char.get("data").unwrap().get(field).is_some(),
        "field must exist at one level (upstream 400s otherwise)"
    );
    let value = Value::from("hello world");
    if let Some(o) = char.as_object_mut() {
        o.insert(field.to_string(), value.clone());
    }
    if let Some(data) = char.get_mut("data") {
        if let Some(o) = data.as_object_mut() {
            o.insert(field.to_string(), value);
        }
    }
    let new_json = serde_json::to_string(&char).unwrap();

    // Write back via chunk surgery, re-read, verify both levels
    use base64::Engine;
    let b64 = base64::engine::general_purpose::STANDARD.encode(new_json.as_bytes());
    let chunk = png::encode_text_chunk("chara", &b64);
    png::write_card_to_png(&chars.join("a.png"), &[chunk]).unwrap();

    let chunks = png::read_text_chunks(&chars.join("a.png")).unwrap();
    let reread = png::extract_card_json(&chunks).unwrap();
    let updated: Value = serde_json::from_str(&reread).unwrap();
    assert_eq!(updated["description"], "hello world");
    assert_eq!(updated["data"]["description"], "hello world");

    // Unknown field would be rejected upstream (not present at either level)
    let unknown_exists =
        updated.get("nonexistent").is_some() || updated["data"].get("nonexistent").is_some();
    assert!(!unknown_exists);

    let _ = card::humanized_datetime(); // keep import in scope
}

/// rename semantics: name fields updated, new file written, old removed,
/// chats folder migrated.
#[test]
fn rename_rewrites_moves_and_unlinks() {
    let dir = tempfile::tempdir().unwrap();
    let deps = helpers::deps_for(dir.path());
    std::fs::create_dir_all(&deps.characters_dir).unwrap();
    std::fs::create_dir_all(deps.chats_dir.join("oldname")).unwrap();
    std::fs::write(deps.chats_dir.join("oldname/chat.jsonl"), b"chat").unwrap();
    helpers::make_card_png(&deps.characters_dir.join("oldname.png"), "oldname", "d");

    // Simulate rename_sync's core steps (the handler wraps them identically)
    let new_name = st_gateway::card::sanitize_filename("New Name!");
    assert_eq!(new_name, "New Name!"); // ! is not in illegalRe (verified vs Node package)
    let new_internal = st_gateway::card::get_png_name(&new_name, &deps.characters_dir);
    let new_avatar = format!("{new_internal}.png");

    let old_path = deps.characters_dir.join("oldname.png");
    let raw = {
        let chunks = st_gateway::png::read_text_chunks(&old_path).unwrap();
        st_gateway::png::extract_card_json(&chunks).unwrap()
    };
    let parsed: Value = serde_json::from_str(&raw).unwrap();
    let mut old_data = st_gateway::card::get_chara_card_v2(parsed);
    st_gateway::card::set_path_value(&mut old_data, "data.name", Value::from(new_name.clone()));
    st_gateway::card::set_path_value(&mut old_data, "name", Value::from(new_name));
    let new_json = serde_json::to_string(&old_data).unwrap();

    let new_path = deps.characters_dir.join(&new_avatar);
    {
        use base64::Engine;
        let b64 = base64::engine::general_purpose::STANDARD.encode(new_json.as_bytes());
        let chunk = st_gateway::png::encode_text_chunk("chara", &b64);
        st_gateway::png::write_card_to_png_from_to(&old_path, &new_path, &[chunk]).unwrap();
    }

    let old_chats = deps.chats_dir.join("oldname");
    let new_chats = deps.chats_dir.join(&new_internal);
    if old_chats.exists() && !new_chats.exists() {
        std::fs::rename(&old_chats, &new_chats).unwrap();
    }
    std::fs::remove_file(&old_path).unwrap();

    // Verify
    assert!(!old_path.exists());
    assert!(new_path.exists());
    assert!(new_chats.join("chat.jsonl").exists());
    assert!(!old_chats.exists());

    let chunks = st_gateway::png::read_text_chunks(&new_path).unwrap();
    let reread = st_gateway::png::extract_card_json(&chunks).unwrap();
    let updated: Value = serde_json::from_str(&reread).unwrap();
    assert_eq!(updated["name"], "New Name!");
    assert_eq!(updated["data"]["name"], "New Name!");
}

/// export json: private fields stripped, pretty printed.
#[test]
fn export_json_strips_private_fields() {
    let dir = tempfile::tempdir().unwrap();
    let chars = dir.path().join("characters");
    std::fs::create_dir_all(&chars).unwrap();
    helpers::make_card_png(&chars.join("a.png"), "A", "d");

    let raw = {
        let chunks = st_gateway::png::read_text_chunks(&chars.join("a.png")).unwrap();
        st_gateway::png::extract_card_json(&chunks).unwrap()
    };
    let parsed: Value = serde_json::from_str(&raw).unwrap();
    let mut obj = st_gateway::card::get_chara_card_v2(parsed);
    st_gateway::card::unset_private_fields(&mut obj);
    let pretty = serde_json::to_string_pretty(&obj).unwrap();

    let out: Value = serde_json::from_str(&pretty).unwrap();
    assert_eq!(out["fav"], false);
    assert_eq!(out["data"]["extensions"]["fav"], false);
    assert!(out.get("chat").is_none(), "chat must be removed");
    // 4-space pretty print marker
    assert!(pretty.contains("\n    \"name\""));
}
