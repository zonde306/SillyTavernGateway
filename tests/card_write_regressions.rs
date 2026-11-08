//! Regression tests for the write paths that used to corrupt cards:
//!
//! 1. `/rename` must build the new card from the OLD card as source and the
//!    NEW path as destination. Passing the new (not yet existing) path as both
//!    made every rename fail with ENOENT.
//! 2. Concurrent `/edit` requests must not share a temp file. A fixed
//!    `<file>.png.tmp` let two writers interleave bytes and then race the
//!    rename — one request 500'd and the card was left with mixed content
//!    (which also ate `extra` first messages and broke `/get` + `/thumbnail`).

use st_gateway::card::CardDeps;

fn make_card_png(path: &std::path::Path, name: &str, alt: &[&str]) {
    let card = serde_json::json!({
        "spec": "chara_card_v2",
        "spec_version": "2.0",
        "name": name,
        "description": "desc",
        "first_mes": "hi",
        "data": {
            "name": name,
            "description": "desc",
            "first_mes": "hi",
            "alternate_greetings": alt,
        }
    })
    .to_string();
    use base64::Engine;
    let b64 = base64::engine::general_purpose::STANDARD.encode(card.as_bytes());
    let mut data = b"chara\0".to_vec();
    data.extend_from_slice(b64.as_bytes());

    let mut png = Vec::new();
    png.extend_from_slice(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]);
    png.extend_from_slice(&st_gateway::png::build_chunk(
        b"IHDR",
        &[0, 0, 0, 1, 0, 0, 0, 1, 8, 6, 0, 0, 0],
    ));
    png.extend_from_slice(&st_gateway::png::build_chunk(b"tEXt", &data));
    png.extend_from_slice(&st_gateway::png::build_chunk(b"IDAT", &[0, 0, 0]));
    png.extend_from_slice(&st_gateway::png::build_chunk(b"IEND", &[]));
    std::fs::write(path, png).unwrap();
}

fn deps_for(dir: &std::path::Path) -> CardDeps {
    let chars = dir.join("default-user/characters");
    std::fs::create_dir_all(&chars).unwrap();
    std::fs::create_dir_all(dir.join("default-user/chats")).unwrap();
    CardDeps {
        characters_dir: chars,
        chats_dir: dir.join("default-user/chats"),
        worlds_dir: dir.join("default-user/worlds"),
    }
}

#[test]
fn rename_produces_a_valid_card_and_removes_the_original() {
    let dir = tempfile::tempdir().unwrap();
    let deps = deps_for(dir.path());
    let old = deps.characters_dir.join("old.png");
    make_card_png(&old, "Old", &["g1", "g2"]);

    let new_avatar =
        st_gateway::routes::__test::rename_sync(&deps, dir.path(), "old.png", "New Name!")
            .expect("rename must succeed");

    assert_eq!(new_avatar, "New Name!.png");
    assert!(!old.exists(), "old card must be removed");
    let new_path = deps.characters_dir.join(&new_avatar);
    assert!(new_path.exists());

    // The renamed file must be a complete, parseable card (this is what broke
    // /characters/get into 404/zeroed stubs before).
    let chunks = st_gateway::png::read_text_chunks(&new_path).unwrap();
    let raw = st_gateway::png::extract_card_json(&chunks).unwrap();
    let card: serde_json::Value = serde_json::from_str(&raw).unwrap();
    assert_eq!(card["name"], "New Name!");
    assert_eq!(card["data"]["name"], "New Name!");
    // untouched payload survives the copy
    assert_eq!(card["data"]["alternate_greetings"][1], "g2");
}

#[test]
fn rename_collisions_do_not_overwrite_existing_cards() {
    let dir = tempfile::tempdir().unwrap();
    let deps = deps_for(dir.path());
    make_card_png(&deps.characters_dir.join("old.png"), "Old", &[]);
    make_card_png(&deps.characters_dir.join("Taken.png"), "Taken", &[]);

    let new_avatar =
        st_gateway::routes::__test::rename_sync(&deps, dir.path(), "old.png", "Taken").unwrap();
    assert_eq!(new_avatar, "Taken1.png");
    // the pre-existing card is untouched
    let chunks = st_gateway::png::read_text_chunks(&deps.characters_dir.join("Taken.png")).unwrap();
    let card: serde_json::Value =
        serde_json::from_str(&st_gateway::png::extract_card_json(&chunks).unwrap()).unwrap();
    assert_eq!(card["name"], "Taken");
}

#[test]
fn rename_of_a_missing_card_reports_the_path() {
    let dir = tempfile::tempdir().unwrap();
    let deps = deps_for(dir.path());
    let err =
        st_gateway::routes::__test::rename_sync(&deps, dir.path(), "gone.png", "New").unwrap_err();
    assert!(err.contains("gone.png"), "error must name the file: {err}");
}

#[test]
fn concurrent_edits_keep_the_card_intact() {
    let dir = tempfile::tempdir().unwrap();
    let deps = deps_for(dir.path());
    let path = deps.characters_dir.join("c.png");
    make_card_png(&path, "C", &["g1", "g2", "g3"]);
    let data_root = dir.path().to_path_buf();

    let handles: Vec<_> = (0..16)
        .map(|i| {
            let deps = deps.clone();
            let data_root = data_root.clone();
            std::thread::spawn(move || {
                let form = serde_json::json!({
                    "ch_name": "C",
                    "avatar_url": "c.png",
                    "description": format!("desc{i}"),
                    "first_mes": "hi",
                    "alternate_greetings": ["g1", "g2", "g3"],
                });
                st_gateway::routes::__test::edit_full_sync(
                    &form, "c.png", None, None, &deps, &data_root,
                )
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap().expect("every edit must succeed");
    }

    let chunks = st_gateway::png::read_text_chunks(&path).unwrap();
    let raw = st_gateway::png::extract_card_json(&chunks).unwrap();
    let card: serde_json::Value = serde_json::from_str(&raw).expect("card must still be valid");
    assert_eq!(card["data"]["alternate_greetings"][2], "g3");
    assert!(card["data"]["description"]
        .as_str()
        .unwrap()
        .starts_with("desc"));

    // no temp litter left behind
    let leftovers: Vec<String> = std::fs::read_dir(&deps.characters_dir)
        .unwrap()
        .flatten()
        .map(|e| e.file_name().to_string_lossy().to_string())
        .filter(|n| n.ends_with(".tmp"))
        .collect();
    assert!(leftovers.is_empty(), "leftover temp files: {leftovers:?}");
}

#[test]
fn edit_recreates_a_missing_card_instead_of_failing() {
    let dir = tempfile::tempdir().unwrap();
    let deps = deps_for(dir.path());
    // Provide the default avatar the fallback needs.
    let public = dir.path().join("public/img");
    std::fs::create_dir_all(&public).unwrap();
    let img = image::DynamicImage::new_rgb8(8, 8);
    img.write_to(
        &mut std::fs::File::create(public.join("ai4.png")).unwrap(),
        image::ImageFormat::Png,
    )
    .unwrap();

    std::env::set_current_dir(dir.path()).ok();
    let form = serde_json::json!({
        "ch_name": "Ghost",
        "avatar_url": "ghost.png",
        "description": "resurrected",
    });
    st_gateway::routes::__test::edit_full_sync(&form, "ghost.png", None, None, &deps, dir.path())
        .expect("a missing source card must fall back to the default avatar");

    let written = deps.characters_dir.join("ghost.png");
    assert!(written.exists(), "card must be recreated");
    let chunks = st_gateway::png::read_text_chunks(&written).unwrap();
    let card: serde_json::Value =
        serde_json::from_str(&st_gateway::png::extract_card_json(&chunks).unwrap()).unwrap();
    assert_eq!(card["description"], "resurrected");
}
