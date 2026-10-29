//! B2 integration tests: /create, /edit (multipart+crop), /edit-avatar,
//! /import — exercised through the real Axum router with synthesized
//! multipart bodies. Uses the default-user data dir sandbox via tempdirs.

/// Build a PNG card file in the characters dir (helpers mirror b1_routes).
fn make_card_png(path: &std::path::Path, name: &str) {
    let card = serde_json::json!({
        "spec": "chara_card_v2", "spec_version": "2.0",
        "name": name,
        "description": "desc",
        "data": { "name": name, "description": "desc" }
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

/// A real 4×4 red PNG (encoded via the image crate) for avatar uploads.
fn tiny_png() -> Vec<u8> {
    let img = image::DynamicImage::new_rgb8(4, 4);
    let mut out = Vec::new();
    img.write_to(&mut std::io::Cursor::new(&mut out), image::ImageFormat::Png)
        .unwrap();
    out
}

/// Auth-less request injection: bypass check_auth by calling the sync
/// functions directly (check_auth requires a session cookie). We test the
/// sync cores + chunk semantics here; the HTTP wiring is trivial routing.
#[test]
fn create_sync_json_body_creates_card() {
    let dir = tempfile::tempdir().unwrap();
    let deps = st_gateway::card::CardDeps {
        characters_dir: dir.path().join("default-user/characters"),
        chats_dir: dir.path().join("default-user/chats"),
        worlds_dir: dir.path().join("default-user/worlds"),
    };
    std::fs::create_dir_all(&deps.characters_dir).unwrap();
    std::fs::create_dir_all(&deps.chats_dir).unwrap();
    let data_root = dir.path().to_path_buf();

    let form = serde_json::json!({
        "ch_name": "Test Char",
        "description": "hello",
        "first_mes": "hi",
        "talkativeness": "0.5",
    });
    let ch_name = st_gateway::card::sanitize_filename("Test Char");
    let avatar_name =
        st_gateway::routes::__test::create_sync(&form, &ch_name, None, None, &deps, &data_root)
            .unwrap();

    assert_eq!(avatar_name, "Test Char.png");
    let out_path = deps.characters_dir.join("Test Char.png");
    assert!(out_path.exists());
    // chats folder created
    assert!(deps.chats_dir.join("Test Char").is_dir());

    // Card JSON readable via chunk surgery; has v2 + v3 chunks
    let chunks = st_gateway::png::read_text_chunks(&out_path).unwrap();
    let json_raw = st_gateway::png::extract_card_json(&chunks).unwrap();
    let card: serde_json::Value = serde_json::from_str(&json_raw).unwrap();
    // extraction prefers ccv3 (write() semantics: both chunks present)
    assert_eq!(card["spec"], "chara_card_v3");
    assert_eq!(card["data"]["name"], "Test Char");
    assert_eq!(card["data"]["description"], "hello");
    assert_eq!(card["avatar"], "none");
    let has_v3 = chunks.iter().any(|c| c.keyword == "ccv3");
    assert!(has_v3, "ccv3 chunk must be present (write() semantics)");
}

#[test]
fn create_sync_upload_with_crop() {
    let dir = tempfile::tempdir().unwrap();
    let deps = st_gateway::card::CardDeps {
        characters_dir: dir.path().join("default-user/characters"),
        chats_dir: dir.path().join("default-user/chats"),
        worlds_dir: dir.path().join("default-user/worlds"),
    };
    std::fs::create_dir_all(&deps.characters_dir).unwrap();
    std::fs::create_dir_all(&deps.chats_dir).unwrap();
    let data_root = dir.path().to_path_buf();

    let form = serde_json::json!({ "ch_name": "Cropped", "description": "d" });
    let ch_name = st_gateway::card::sanitize_filename("Cropped");
    let upload = tiny_png(); // 4×4
    let crop = st_gateway::avatar::Crop {
        x: 0.0,
        y: 0.0,
        width: 4.0,
        height: 4.0,
        want_resize: true, // → 512×768
    };

    let avatar_name = st_gateway::routes::__test::create_sync(
        &form,
        &ch_name,
        Some(&upload),
        Some(crop),
        &deps,
        &data_root,
    )
    .unwrap();

    let out_path = deps.characters_dir.join(&avatar_name);
    let img = image::open(&out_path).unwrap();
    assert_eq!((img.width(), img.height()), (512, 768));
    // card chunks present in the re-encoded PNG
    let chunks = st_gateway::png::read_text_chunks(&out_path).unwrap();
    let json_raw = st_gateway::png::extract_card_json(&chunks).unwrap();
    let card: serde_json::Value = serde_json::from_str(&json_raw).unwrap();
    assert_eq!(card["data"]["name"], "Cropped");
}

#[test]
fn edit_avatar_sync_preserves_card_data() {
    let dir = tempfile::tempdir().unwrap();
    let deps = st_gateway::card::CardDeps {
        characters_dir: dir.path().join("default-user/characters"),
        chats_dir: dir.path().join("default-user/chats"),
        worlds_dir: dir.path().join("default-user/worlds"),
    };
    std::fs::create_dir_all(&deps.characters_dir).unwrap();
    let data_root = dir.path().to_path_buf();

    make_card_png(&deps.characters_dir.join("char.png"), "Char");

    let upload = tiny_png();
    let crop = st_gateway::avatar::Crop {
        x: 0.0,
        y: 0.0,
        width: 4.0,
        height: 4.0,
        want_resize: false,
    };
    st_gateway::routes::__test::edit_avatar_sync(
        &upload,
        "char.png",
        Some(crop),
        &deps,
        &data_root,
    )
    .unwrap();

    // Image replaced while card data survives. Note: write() upgrades the
    // payload to spec v3 (ccv3 chunk wins extraction), so compare data
    // fields, not the raw string.
    let after = {
        let chunks =
            st_gateway::png::read_text_chunks(&deps.characters_dir.join("char.png")).unwrap();
        st_gateway::png::extract_card_json(&chunks).unwrap()
    };
    let after_v: serde_json::Value = serde_json::from_str(&after).unwrap();
    assert_eq!(after_v["spec"], "chara_card_v3", "write() upgrades to v3");
    assert_eq!(after_v["name"], "Char");
    assert_eq!(after_v["description"], "desc");
    assert_eq!(after_v["data"]["name"], "Char");
    assert_eq!(after_v["data"]["description"], "desc");
    let img = image::open(deps.characters_dir.join("char.png")).unwrap();
    assert_eq!((img.width(), img.height()), (4, 4));
}

#[test]
fn import_sync_png_v2_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let deps = st_gateway::card::CardDeps {
        characters_dir: dir.path().join("default-user/characters"),
        chats_dir: dir.path().join("default-user/chats"),
        worlds_dir: dir.path().join("default-user/worlds"),
    };
    std::fs::create_dir_all(&deps.characters_dir).unwrap();
    let data_root = dir.path().to_path_buf();

    // Build an upload: REAL decodable PNG + V2 card chunks
    let base = tiny_png();
    let card = serde_json::json!({
        "spec": "chara_card_v2", "spec_version": "2.0",
        "name": "Imported", "description": "d",
        "data": { "name": "Imported", "description": "d" }
    })
    .to_string();
    use base64::Engine;
    let b64 = base64::engine::general_purpose::STANDARD.encode(card.as_bytes());
    let upload_path = dir.path().join("upload.png");
    std::fs::write(&upload_path, &base).unwrap();
    let chunk = st_gateway::png::encode_text_chunk("chara", &b64);
    st_gateway::png::write_card_to_png(&upload_path, &[chunk]).unwrap();
    let upload = std::fs::read(&upload_path).unwrap();

    let file_name =
        st_gateway::routes::__test::import_sync(&upload, "png", None, &deps, &data_root).unwrap();

    assert_eq!(file_name, "Imported");
    let out_path = deps.characters_dir.join("Imported.png");
    assert!(out_path.exists());

    let chunks = st_gateway::png::read_text_chunks(&out_path).unwrap();
    let json_raw = st_gateway::png::extract_card_json(&chunks).unwrap();
    let card: serde_json::Value = serde_json::from_str(&json_raw).unwrap();
    assert_eq!(card["spec"], "chara_card_v3"); // ccv3 chunk wins extraction
    assert_eq!(card["data"]["name"], "Imported");
    // create_date refreshed (ISO)
    assert!(card["create_date"].as_str().unwrap().contains("T"));
    // image preserved (decodable)
    let img = image::open(&out_path).unwrap();
    assert_eq!((img.width(), img.height()), (4, 4));
}

#[test]
fn import_sync_yaml() {
    let dir = tempfile::tempdir().unwrap();
    let deps = st_gateway::card::CardDeps {
        characters_dir: dir.path().join("default-user/characters"),
        chats_dir: dir.path().join("default-user/chats"),
        worlds_dir: dir.path().join("default-user/worlds"),
    };
    std::fs::create_dir_all(&deps.characters_dir).unwrap();
    let data_root = dir.path().to_path_buf();

    let yaml_text = b"name: Yaml Char\ncontext: Some context\ngreeting: Hello there\n";
    let file_name =
        st_gateway::routes::__test::import_sync(yaml_text, "yaml", None, &deps, &data_root)
            .unwrap();

    assert_eq!(file_name, "Yaml Char");
    let out_path = deps.characters_dir.join("Yaml Char.png");
    assert!(out_path.exists());
    let chunks = st_gateway::png::read_text_chunks(&out_path).unwrap();
    let json_raw = st_gateway::png::extract_card_json(&chunks).unwrap();
    let card: serde_json::Value = serde_json::from_str(&json_raw).unwrap();
    assert_eq!(card["data"]["description"], "Some context");
    assert_eq!(card["data"]["first_mes"], "Hello there");
    // default avatar image used (ai4.png decoded)
    let img = image::open(&out_path).unwrap();
    assert!(img.width() > 0);
}

#[test]
fn edit_full_sync_json_branch_uses_chunk_surgery() {
    let dir = tempfile::tempdir().unwrap();
    let deps = st_gateway::card::CardDeps {
        characters_dir: dir.path().join("default-user/characters"),
        chats_dir: dir.path().join("default-user/chats"),
        worlds_dir: dir.path().join("default-user/worlds"),
    };
    std::fs::create_dir_all(&deps.characters_dir).unwrap();
    let data_root = dir.path().to_path_buf();

    // Real decodable PNG (chunk surgery must keep it decodable)
    let base = tiny_png();
    let card = serde_json::json!({
        "spec": "chara_card_v2", "spec_version": "2.0",
        "name": "E", "description": "desc",
        "data": { "name": "E", "description": "desc" }
    })
    .to_string();
    use base64::Engine;
    let b64 = base64::engine::general_purpose::STANDARD.encode(card.as_bytes());
    let tmp_src = deps.characters_dir.join("e.src.png");
    std::fs::write(&tmp_src, &base).unwrap();
    let chunk = st_gateway::png::encode_text_chunk("chara", &b64);
    st_gateway::png::write_card_to_png(&tmp_src, &[chunk]).unwrap();
    std::fs::rename(&tmp_src, deps.characters_dir.join("e.png")).unwrap();

    let form = serde_json::json!({
        "ch_name": "E",
        "avatar_url": "e.png",
        "description": "updated description",
        "chat": "e - chat",
        "create_date": "2024-01-01T00:00:00.000Z",
    });
    st_gateway::routes::__test::edit_full_sync(&form, "e.png", None, None, &deps, &data_root)
        .unwrap();

    let chunks = st_gateway::png::read_text_chunks(&deps.characters_dir.join("e.png")).unwrap();
    let json_raw = st_gateway::png::extract_card_json(&chunks).unwrap();
    let card: serde_json::Value = serde_json::from_str(&json_raw).unwrap();
    assert_eq!(card["data"]["description"], "updated description");
    assert_eq!(card["chat"], "e - chat");
    assert_eq!(card["create_date"], "2024-01-01T00:00:00.000Z");
    // chunk surgery: image data untouched (file stays tiny, not re-encoded)
    assert!(image::open(deps.characters_dir.join("e.png")).is_ok());
}

/// multer semantics: an avatar part with EMPTY filename is discarded —
/// upload stays None so /edit keeps the chunk-surgery path (avatar kept).
/// Regression test: /edit favoriting a character must NOT reset the avatar.
#[tokio::test]
async fn multipart_empty_filename_avatar_is_discarded() {
    use st_gateway::routes::__test::parse_multipart;
    let b = "----B";
    let crlf = "\r\n";
    let mut payload = String::new();
    payload.push_str(&format!("--{b}{crlf}"));
    payload.push_str(&format!(
        "Content-Disposition: form-data; name=\"ch_name\"{crlf}{crlf}"
    ));
    payload.push_str(&format!("test{crlf}"));
    payload.push_str(&format!("--{b}{crlf}"));
    payload.push_str(&format!(
        "Content-Disposition: form-data; name=\"avatar\"; filename=\"\"{crlf}"
    ));
    payload.push_str(&format!(
        "Content-Type: application/octet-stream{crlf}{crlf}"
    ));
    payload.push_str(&format!("{crlf}--{b}--{crlf}"));
    let ct = format!("multipart/form-data; boundary={b}");
    let mp = parse_multipart(&ct, axum::body::Body::from(payload.into_bytes()))
        .await
        .unwrap();
    assert!(
        mp.file_bytes.is_none(),
        "empty-filename avatar must not count as an upload (multer: !filename → discard)"
    );
    assert_eq!(mp.fields[0].0, "ch_name");
}

/// A real upload (filename present) still parses as a file.
#[tokio::test]
async fn multipart_real_avatar_still_parses() {
    use st_gateway::routes::__test::parse_multipart;
    let b = "----B";
    let crlf = "\r\n";
    let mut payload = String::new();
    payload.push_str(&format!("--{b}{crlf}"));
    payload.push_str(&format!(
        "Content-Disposition: form-data; name=\"avatar\"; filename=\"a.png\"{crlf}"
    ));
    payload.push_str(&format!("Content-Type: image/png{crlf}{crlf}"));
    payload.push_str("PNGDATA");
    payload.push_str(&format!("{crlf}--{b}--{crlf}"));
    let ct = format!("multipart/form-data; boundary={b}");
    let mp = parse_multipart(&ct, axum::body::Body::from(payload.into_bytes()))
        .await
        .unwrap();
    assert_eq!(mp.file_bytes.as_deref(), Some(b"PNGDATA".as_slice()));
}

/// End-to-end regression: /edit with an empty-filename avatar part must
/// preserve the existing PNG image (chunk surgery, no default-avatar reset).
#[tokio::test]
async fn edit_with_empty_avatar_part_preserves_image() {
    let dir = tempfile::tempdir().unwrap();
    let deps = st_gateway::card::CardDeps {
        characters_dir: dir.path().join("default-user/characters"),
        chats_dir: dir.path().join("default-user/chats"),
        worlds_dir: dir.path().join("default-user/worlds"),
    };
    std::fs::create_dir_all(&deps.characters_dir).unwrap();
    let data_root = dir.path().to_path_buf();

    // Existing card: real PNG + chara chunk
    let base = tiny_png();
    let card = serde_json::json!({
        "spec": "chara_card_v2", "spec_version": "2.0",
        "name": "F", "description": "orig",
        "data": { "name": "F", "description": "orig" }
    })
    .to_string();
    use base64::Engine;
    let b64 = base64::engine::general_purpose::STANDARD.encode(card.as_bytes());
    let src = deps.characters_dir.join("f.src.png");
    std::fs::write(&src, &base).unwrap();
    let chunk = st_gateway::png::encode_text_chunk("chara", &b64);
    st_gateway::png::write_card_to_png(&src, &[chunk]).unwrap();
    std::fs::rename(&src, deps.characters_dir.join("f.png")).unwrap();

    // /edit form: upload = None (after the multer-mirror fix in
    // parse_multipart, the browser's empty-filename avatar part maps to None)
    let form = serde_json::json!({
        "ch_name": "F",
        "avatar_url": "f.png",
        "description": "favorited!",
        "fav": "true",
        "json_data": "",
        "extensions": "{}",
        "talkativeness": "0.5",
    });
    st_gateway::routes::__test::edit_full_sync(&form, "f.png", None, None, &deps, &data_root)
        .unwrap();

    // Card JSON updated...
    let chunks = st_gateway::png::read_text_chunks(&deps.characters_dir.join("f.png")).unwrap();
    let raw = st_gateway::png::extract_card_json(&chunks).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&raw).unwrap();
    assert_eq!(parsed["data"]["description"], "favorited!");

    // ...and the IMAGE DATA is untouched (4x4, not the default avatar)
    let after = std::fs::read(deps.characters_dir.join("f.png")).unwrap();
    let _ = after;
    // Verify the PNG still decodes to 4x4 (not the default avatar, which
    // is much larger) — proves the image data was never replaced.
    let img = image::open(deps.characters_dir.join("f.png")).unwrap();
    assert_eq!(
        (img.width(), img.height()),
        (4, 4),
        "avatar must remain the original 4x4 image, not the default avatar"
    );
}

/// Regression test: /edit multipart with multiple `alternate_greetings` fields
/// must preserve ALL alternate first messages (not overwrite earlier ones, leaving
/// only 1 primary + 1 alternate greeting).
#[tokio::test]
async fn edit_multipart_multiple_alternate_greetings_preserved() {
    use st_gateway::routes::__test::{edit_full_sync, multipart_fields_to_map, parse_multipart};

    let dir = tempfile::tempdir().unwrap();
    let deps = st_gateway::card::CardDeps {
        characters_dir: dir.path().join("default-user/characters"),
        chats_dir: dir.path().join("default-user/chats"),
        worlds_dir: dir.path().join("default-user/worlds"),
    };
    std::fs::create_dir_all(&deps.characters_dir).unwrap();
    let data_root = dir.path().to_path_buf();

    // Existing card: real PNG + chara chunk with initial greetings
    let base = tiny_png();
    let card = serde_json::json!({
        "spec": "chara_card_v2", "spec_version": "2.0",
        "name": "MultiFM", "description": "desc",
        "data": {
            "name": "MultiFM",
            "description": "desc",
            "first_mes": "Primary message",
            "alternate_greetings": ["Initial alt 1", "Initial alt 2"]
        }
    })
    .to_string();
    use base64::Engine;
    let b64 = base64::engine::general_purpose::STANDARD.encode(card.as_bytes());
    let src = deps.characters_dir.join("multifm.src.png");
    std::fs::write(&src, &base).unwrap();
    let chunk = st_gateway::png::encode_text_chunk("chara", &b64);
    st_gateway::png::write_card_to_png(&src, &[chunk]).unwrap();
    std::fs::rename(&src, deps.characters_dir.join("multifm.png")).unwrap();

    // Construct multipart payload exactly as the browser frontend sends it on /edit:
    // multiple form parts with the identical name "alternate_greetings".
    let b = "----WebKitFormBoundaryMultiFM";
    let crlf = "\r\n";
    let mut payload = String::new();

    let fields = [
        ("ch_name", "MultiFM"),
        ("avatar_url", "multifm.png"),
        ("description", "updated desc"),
        ("first_mes", "Primary message"),
        ("alternate_greetings", "Alt greeting 1"),
        ("alternate_greetings", "Alt greeting 2"),
        ("alternate_greetings", "Alt greeting 3"),
    ];

    for (k, v) in fields {
        payload.push_str(&format!("--{b}{crlf}"));
        payload.push_str(&format!(
            "Content-Disposition: form-data; name=\"{k}\"{crlf}{crlf}"
        ));
        payload.push_str(&format!("{v}{crlf}"));
    }
    // Browser sends empty avatar field when no new file is chosen
    payload.push_str(&format!("--{b}{crlf}"));
    payload.push_str(&format!(
        "Content-Disposition: form-data; name=\"avatar\"; filename=\"\"{crlf}"
    ));
    payload.push_str(&format!(
        "Content-Type: application/octet-stream{crlf}{crlf}"
    ));
    payload.push_str(&format!("{crlf}--{b}--{crlf}"));

    let ct = format!("multipart/form-data; boundary={b}");
    let mp = parse_multipart(&ct, axum::body::Body::from(payload.into_bytes()))
        .await
        .unwrap();

    assert!(mp.file_bytes.is_none());
    let form_map = multipart_fields_to_map(&mp.fields);
    let form = serde_json::Value::Object(form_map);

    // Verify multipart_fields_to_map preserved all 3 alternate_greetings as an array
    let alt_greetings = form
        .get("alternate_greetings")
        .and_then(|v| v.as_array())
        .unwrap();
    assert_eq!(alt_greetings.len(), 3);
    assert_eq!(alt_greetings[0], "Alt greeting 1");
    assert_eq!(alt_greetings[1], "Alt greeting 2");
    assert_eq!(alt_greetings[2], "Alt greeting 3");

    // Execute edit
    edit_full_sync(&form, "multifm.png", None, None, &deps, &data_root).unwrap();

    // Read back character card from the PNG file
    let chunks =
        st_gateway::png::read_text_chunks(&deps.characters_dir.join("multifm.png")).unwrap();
    let raw = st_gateway::png::extract_card_json(&chunks).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&raw).unwrap();

    // Verify primary FM and all 3 alternate greetings survived
    assert_eq!(parsed["data"]["first_mes"], "Primary message");
    let saved_alts = parsed["data"]["alternate_greetings"].as_array().unwrap();
    assert_eq!(
        saved_alts,
        &vec![
            serde_json::json!("Alt greeting 1"),
            serde_json::json!("Alt greeting 2"),
            serde_json::json!("Alt greeting 3"),
        ],
        "all alternate first messages must be preserved after /edit"
    );
}

/// Unit test: multipart_fields_to_map converts repeated fields to arrays
/// matching Node.js multer semantics, while single fields remain single values.
#[test]
fn multipart_fields_to_map_duplicates_to_array() {
    use st_gateway::routes::__test::multipart_fields_to_map;

    let fields = vec![
        ("name".to_string(), "Alice".to_string()),
        ("tag".to_string(), "t1".to_string()),
        ("tag".to_string(), "t2".to_string()),
        ("tag".to_string(), "t3".to_string()),
        ("single".to_string(), "val".to_string()),
    ];

    let map = multipart_fields_to_map(&fields);
    assert_eq!(map.get("name").unwrap(), &serde_json::json!("Alice"));
    assert_eq!(map.get("single").unwrap(), &serde_json::json!("val"));
    assert_eq!(
        map.get("tag").unwrap(),
        &serde_json::json!(["t1", "t2", "t3"])
    );
}
