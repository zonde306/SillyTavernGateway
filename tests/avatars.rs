//! `/api/avatars/*` — persona avatar storage and upload safety.
//!
//! The upload path is where a "large image" turns into a process-killing
//! allocation, so these tests pin both the memory boundary and the upstream
//! behaviour that must survive it.

use st_gateway::avatar::{self, Crop};

fn png_of(w: u32, h: u32) -> Vec<u8> {
    let img = image::DynamicImage::ImageRgba8(image::RgbaImage::from_pixel(
        w,
        h,
        image::Rgba([12, 34, 56, 255]),
    ));
    let mut out = Vec::new();
    img.write_to(&mut std::io::Cursor::new(&mut out), image::ImageFormat::Png)
        .unwrap();
    out
}

fn decode_dims(bytes: &[u8]) -> (u32, u32) {
    let img = image::load_from_memory(bytes).unwrap();
    (img.width(), img.height())
}

#[test]
fn upload_stores_a_cropped_avatar_and_returns_the_filename() {
    let dir = tempfile::tempdir().unwrap();
    let user_dir = dir.path().join("default-user");
    let avatars = user_dir.join("User Avatars");
    std::fs::create_dir_all(&avatars).unwrap();

    let crop = Crop {
        x: 0.0,
        y: 0.0,
        width: 400.0,
        height: 400.0,
        want_resize: true,
    };
    let name = st_gateway::routes::__test::upload_avatar_sync(
        &avatars,
        &user_dir,
        &png_of(400, 400),
        Some(crop),
        "persona.png",
    )
    .unwrap();

    assert_eq!(name, "persona.png");
    let written = std::fs::read(avatars.join(&name)).unwrap();
    // applyAvatarCropResize: crop then cover to AVATAR_WIDTH x AVATAR_HEIGHT.
    assert_eq!(
        decode_dims(&written),
        (avatar::AVATAR_WIDTH, avatar::AVATAR_HEIGHT)
    );
}

#[test]
fn upload_without_a_name_uses_a_timestamped_filename() {
    let dir = tempfile::tempdir().unwrap();
    let user_dir = dir.path().join("default-user");
    let avatars = user_dir.join("User Avatars");
    std::fs::create_dir_all(&avatars).unwrap();

    let name = st_gateway::routes::__test::upload_avatar_sync(
        &avatars,
        &user_dir,
        &png_of(64, 64),
        None,
        "",
    )
    .unwrap();
    assert!(name.ends_with(".png"), "got {name}");
    assert!(name.trim_end_matches(".png").parse::<u128>().is_ok());
    assert!(avatars.join(&name).is_file());
}

#[test]
fn an_oversized_image_is_rejected_and_leaves_nothing_behind() {
    let dir = tempfile::tempdir().unwrap();
    let user_dir = dir.path().join("default-user");
    let avatars = user_dir.join("User Avatars");
    std::fs::create_dir_all(&avatars).unwrap();

    // 48MP: above MAX_SOURCE_PIXELS (40MP). Solid colour keeps the encoded
    // file tiny, so this proves the rejection is dimensional, not size-based.
    let result = st_gateway::routes::__test::upload_avatar_sync(
        &avatars,
        &user_dir,
        &png_of(8000, 6000),
        None,
        "big.png",
    );
    let err = result.unwrap_err();
    assert!(err.contains("too large"), "got {err}");
    assert!(
        std::fs::read_dir(&avatars).unwrap().next().is_none(),
        "a rejected upload must not write a file"
    );
}

#[test]
fn the_character_routes_refuse_an_oversized_avatar_too() {
    // /create and /edit reach the same decoder. Before the pixel ceiling they
    // would either OOM or silently substitute the default avatar.
    let oversized = png_of(8000, 6000);
    let default = png_of(8, 8);
    let err = avatar::prepare_upload_bytes(&oversized, None, &default).unwrap_err();
    assert!(err.contains("megapixel"), "got {err}");
}

#[test]
fn overwriting_removes_the_stale_persona_thumbnail() {
    let dir = tempfile::tempdir().unwrap();
    let user_dir = dir.path().join("default-user");
    let avatars = user_dir.join("User Avatars");
    let thumbs = user_dir.join("thumbnails/persona");
    std::fs::create_dir_all(&avatars).unwrap();
    std::fs::create_dir_all(&thumbs).unwrap();
    std::fs::write(thumbs.join("persona.png"), b"stale").unwrap();

    st_gateway::routes::__test::upload_avatar_sync(
        &avatars,
        &user_dir,
        &png_of(32, 32),
        None,
        "persona.png",
    )
    .unwrap();

    assert!(
        !thumbs.join("persona.png").exists(),
        "invalidateThumbnail('persona') must unlink the cached thumb"
    );
}

#[test]
fn listing_filters_to_images_and_orders_like_intl_collator() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    for name in ["b.png", "A.png", "notes.txt", "c.webp", "clip.mp4"] {
        std::fs::write(dir.join(name), b"x").unwrap();
    }
    std::fs::create_dir_all(dir.join("subfolder.png")).unwrap();

    let names = st_gateway::routes::__test::list_persona_avatars(dir);
    assert_eq!(names, vec!["A.png", "b.png", "c.webp"]);
}
