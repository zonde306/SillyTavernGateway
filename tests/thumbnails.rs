//! Thumbnail generation tests: cover semantics, animated skip, config
//! dimensions, negative cache, and concurrency limits.

use axum::body::Body;
use axum::http::Request;
use st_gateway::thumbnails::{generate_thumbnail, sanitize_alters, ThumbLimiter};
use st_gateway::{config, AppState};
use std::sync::Arc;

fn test_app_state(dir: &tempfile::TempDir, dims: Option<Vec<u32>>) -> Arc<AppState> {
    let avatar_dims = dims.unwrap_or_else(|| vec![96, 144]);
    let cfg_text = format!(
        "dataRoot: {}\nport: 8000\nthumbnails:\n  enabled: true\n  format: jpg\n  quality: 95\n  dimensions:\n    avatar:\n      - {}\n      - {}\ngateway:\n  routes:\n    \"thumbnail\": local\n",
        dir.path().join("data").to_string_lossy().replace('\\', "/"),
        avatar_dims[0],
        avatar_dims[1],
    );
    let cfg = config::ResolvedConfig::from_text(&cfg_text, dir.path().to_path_buf());
    Arc::new(AppState {
        thumb_limiter: Arc::new(ThumbLimiter::new(cfg.thumb_concurrency())),
        disk_cache: st_gateway::disk_cache::DiskCache::new(&cfg.data_root),
        cfg: Arc::new(cfg),
        node: st_gateway::forward::NodeClient::new(),
    })
}

/// A small opaque PNG built in-memory via the image crate.
fn make_png(width: u32, height: u32, color: [u8; 3]) -> Vec<u8> {
    let mut buf = Vec::new();
    let img = image::RgbaImage::from_fn(width, height, |_, _| {
        image::Rgba([color[0], color[1], color[2], 255])
    });
    image::DynamicImage::ImageRgba8(img)
        .write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
        .unwrap();
    buf
}

fn write_apng(path: &std::path::Path) {
    // PNG signature + IHDR + acTL chunk + IEND — isAnimatedApng matches on
    // 'acTL' within the first 200 bytes regardless of chunk validity.
    let mut b = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
    b.extend_from_slice(&st_gateway::png::build_chunk(
        b"acTL",
        &[0, 0, 0, 1, 0, 0, 0, 0],
    ));
    b.extend_from_slice(&st_gateway::png::build_chunk(b"IEND", &[]));
    std::fs::write(path, b).unwrap();
}

#[tokio::test]
async fn generates_avatar_cover_thumbnail() {
    let dir = tempfile::tempdir().unwrap();
    let state = test_app_state(&dir, None);
    let user = dir.path().join("data/default-user");
    std::fs::create_dir_all(user.join("characters")).unwrap();
    // 400x200 source (wide) → cover 96x144 crops sides
    std::fs::write(
        user.join("characters/a.png"),
        make_png(400, 200, [255, 0, 0]),
    )
    .unwrap();

    let ok = generate_thumbnail(&state, &user, "avatar", "a.png").unwrap();
    assert!(ok);

    let thumb_path = user.join("thumbnails/avatar/a.png");
    assert!(thumb_path.exists(), "thumbnail written");
    // thumb is JPEG (default format) despite .png extension — mirror of
    // upstream, which also writes JPEG bytes under the original name.
    let bytes = std::fs::read(&thumb_path).unwrap();
    assert_eq!(&bytes[..2], &[0xFF, 0xD8], "JPEG magic");
    let decoded = image::load_from_memory(&bytes).unwrap();
    assert_eq!((decoded.width(), decoded.height()), (96, 144));
}

#[tokio::test]
async fn custom_dimensions_from_config() {
    let dir = tempfile::tempdir().unwrap();
    let state = test_app_state(&dir, Some(vec![64, 64]));
    let user = dir.path().join("data/default-user");
    std::fs::create_dir_all(user.join("characters")).unwrap();
    std::fs::write(
        user.join("characters/s.png"),
        make_png(100, 100, [0, 255, 0]),
    )
    .unwrap();

    assert!(generate_thumbnail(&state, &user, "avatar", "s.png").unwrap());
    let bytes = std::fs::read(user.join("thumbnails/avatar/s.png")).unwrap();
    let decoded = image::load_from_memory(&bytes).unwrap();
    assert_eq!((decoded.width(), decoded.height()), (64, 64));
}

#[tokio::test]
async fn apng_is_skipped_no_thumbnail() {
    let dir = tempfile::tempdir().unwrap();
    let state = test_app_state(&dir, None);
    let user = dir.path().join("data/default-user");
    std::fs::create_dir_all(user.join("characters")).unwrap();
    write_apng(&user.join("characters/anim.png"));

    let ok = generate_thumbnail(&state, &user, "avatar", "anim.png").unwrap();
    assert!(!ok, "animated → no thumb");
    assert!(!user.join("thumbnails/avatar/anim.png").exists());
}

#[tokio::test]
async fn bg_area_preserving_scale() {
    let dir = tempfile::tempdir().unwrap();
    let state = test_app_state(&dir, None);
    let user = dir.path().join("data/default-user");
    std::fs::create_dir_all(user.join("backgrounds")).unwrap();
    // 1600x900 → target area 160*90=14400, ratio 16:9 → w=160, h=90 exactly
    std::fs::write(
        user.join("backgrounds/bg.png"),
        make_png(1600, 900, [0, 0, 255]),
    )
    .unwrap();

    assert!(generate_thumbnail(&state, &user, "bg", "bg.png").unwrap());
    let bytes = std::fs::read(user.join("thumbnails/bg/bg.png")).unwrap();
    let decoded = image::load_from_memory(&bytes).unwrap();
    assert_eq!((decoded.width(), decoded.height()), (160, 90));
}

#[tokio::test]
async fn missing_original_is_ok_false() {
    let dir = tempfile::tempdir().unwrap();
    let state = test_app_state(&dir, None);
    let user = dir.path().join("data/default-user");
    std::fs::create_dir_all(user.join("characters")).unwrap();
    let r = generate_thumbnail(&state, &user, "avatar", "ghost.png").unwrap();
    assert!(!r);
}

#[test]
fn sanitize_rejects_path_traversal_and_reserved() {
    assert!(sanitize_alters("../etc/passwd"));
    assert!(sanitize_alters("a/b.png"));
    assert!(sanitize_alters("a\\b.png"));
    assert!(sanitize_alters("CON.png"));
    assert!(sanitize_alters(""));
    assert!(!sanitize_alters("seraphina.png"));
    assert!(!sanitize_alters("My Card v2.png"));
}

#[tokio::test]
async fn negative_cache_expires() {
    let limiter = ThumbLimiter::new(2);
    assert!(!limiter.is_negatively_cached("k").await);
    limiter.mark_failed("k").await;
    assert!(limiter.is_negatively_cached("k").await);
    // Expiry path: no way to wait 60s in a test; force-expire by direct map edit
    {
        let mut map = limiter.failures.lock().await;
        map.insert("k".into(), 0); // already expired
    }
    assert!(!limiter.is_negatively_cached("k").await);
}

#[tokio::test]
async fn concurrency_permits_derive_from_batch_bytes() {
    // Test the config math via ResolvedConfig
    let cfg = config::ResolvedConfig::from_text(
        "gateway:\n  batch_bytes: 8388608\n", // 8 MiB → 1 permit
        std::path::PathBuf::from("."),
    );
    assert_eq!(cfg.thumb_concurrency(), 1);

    let cfg = config::ResolvedConfig::from_text(
        "gateway:\n  batch_bytes: 268435456\n", // 256 MiB → 32 → clamped 16
        std::path::PathBuf::from("."),
    );
    assert_eq!(cfg.thumb_concurrency(), 16);

    let cfg = config::ResolvedConfig::from_text("", std::path::PathBuf::from("."));
    assert_eq!(cfg.thumb_concurrency(), 32.min(16)); // 256MiB default → 16
}

#[tokio::test]
async fn route_serves_generated_thumbnail() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(dir.path().join("data")).unwrap();
    let secret = "test-secret";
    std::fs::write(dir.path().join("data/cookie-secret.txt"), secret).unwrap();
    let state = test_app_state(&dir, None);
    let user = dir.path().join("data/default-user");
    std::fs::create_dir_all(user.join("characters")).unwrap();
    std::fs::write(
        user.join("characters/r.png"),
        make_png(300, 300, [10, 20, 30]),
    )
    .unwrap();

    // Session cookie for auth
    let payload = serde_json::json!({"handle": "default-user", "csrfToken": "t"});
    let value = base64::engine::general_purpose::STANDARD.encode(payload.to_string());
    use base64::Engine;
    // keygrip signs "<full-cookie-name>=<value>" (cookie-session semantics)
    let name = st_gateway::auth::cookie_session_name();
    let data = format!("{}={}", name, value);
    let sig = st_gateway::auth::keygrip_sign(secret, &data);
    let cookie = format!(
        "{}={}; {}.sig={}",
        st_gateway::auth::cookie_session_name(),
        value,
        st_gateway::auth::cookie_session_name(),
        sig
    );

    let app = st_gateway::routes::build_router(state.clone());
    let req = Request::builder()
        .method("GET")
        .uri("/thumbnail?type=avatar&file=r.png")
        .header("cookie", &cookie)
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), 200);
    let ct = resp
        .headers()
        .get("content-type")
        .unwrap()
        .to_str()
        .unwrap();
    // Express sendFile sets content-type by extension (.png), even though
    // the bytes are JPEG — same behavior mirrors upstream.
    assert_eq!(ct, "image/png");
    let bytes = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
        .await
        .unwrap();
    // JPEG magic
    assert_eq!(&bytes[..2], &[0xFF, 0xD8]);
    // And it was persisted
    assert!(user.join("thumbnails/avatar/r.png").exists());
}

// tower ServiceExt for oneshot
use tower::ServiceExt;
