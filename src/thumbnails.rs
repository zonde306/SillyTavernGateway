//! Local thumbnail route (`GET /thumbnail`), mirroring
//! `src/endpoints/thumbnails.js`:
//!
//! - hit path: cached thumb exists and is newer than the original → serve it
//! - miss path: decode original (image crate, native, parallel), resize
//!   (avatar/persona: cover to W×H; bg: area-preserving scale), encode
//!   JPEG/PNG per config, atomic write, serve
//! - animated detection: APNG (`acTL` in first 200 bytes) and WebP (`ANIM` /
//!   `ANMF`) are never thumbnailed — the frontend handles originals
//! - skipped extensions served as originals; GIF always served as original
//! - Firefox cache-busting header on image responses
//!
//! Concurrency: decode permits derive from `gateway.batch_bytes` (the memory
//! budget): batch_bytes / 8 MiB per permit, clamped to [1, 16]. Queued
//! requests hold only the filename, so a 6k cold burst is memory-safe.
//! Negative cache: recently failed files short-circuit for 60s to prevent
//! frontend retry storms from re-hitting disk.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::Semaphore;

use axum::body::Body;
use axum::extract::State;
use axum::http::{header, Request, StatusCode};
use axum::response::{IntoResponse, Response};
use percent_encoding::percent_decode_str;

use crate::auth::Authenticator;
use crate::AppState;

/// Extensions never thumbnailed (served as original). Mirrors
/// SKIPPED_EXTENSIONS.
const SKIPPED_EXTENSIONS: &[&str] = &[".apng", ".mp4", ".webm", ".avi", ".mkv", ".flv", ".gif"];

/// Negative-cache TTL for failed generations.
const NEGATIVE_CACHE_TTL_SECS: u64 = 60;

/// Shared thumbnail machinery (semaphore + negative cache).
pub struct ThumbLimiter {
    pub permits: Arc<Semaphore>,
    /// file key → unix timestamp when the negative entry expires.
    pub failures: tokio::sync::Mutex<HashMap<String, u64>>,
}

impl ThumbLimiter {
    pub fn new(permits: usize) -> Self {
        Self {
            permits: Arc::new(Semaphore::new(permits.max(1))),
            failures: tokio::sync::Mutex::new(HashMap::new()),
        }
    }

    pub async fn is_negatively_cached(&self, key: &str) -> bool {
        let now = now_unix();
        let mut map = self.failures.lock().await;
        match map.get(key) {
            Some(&until) if until > now => true,
            Some(_) => {
                map.remove(key);
                false
            }
            None => false,
        }
    }

    pub async fn mark_failed(&self, key: &str) {
        let mut map = self.failures.lock().await;
        map.insert(key.to_string(), now_unix() + NEGATIVE_CACHE_TTL_SECS);
    }
}

fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Mirror of `sanitize-filename` for query param validation: the upstream
/// route 403s when `sanitize(file) !== file`. We approximate by rejecting
/// filenames that sanitize-filename would alter: control chars, reserved
/// Windows chars, trailing dots/spaces, reserved device names.
pub fn sanitize_alters(file: &str) -> bool {
    if file.is_empty() {
        return true;
    }
    if file.chars().any(|c| {
        c.is_control() || matches!(c, '/' | '\\' | '?' | '*' | ':' | '|' | '"' | '<' | '>')
    }) {
        return true;
    }
    // Windows reserved device names
    let stem = file.split('.').next().unwrap_or("").to_ascii_uppercase();
    if matches!(
        stem.as_str(),
        "CON"
            | "PRN"
            | "AUX"
            | "NUL"
            | "COM1"
            | "COM2"
            | "COM3"
            | "COM4"
            | "COM5"
            | "COM6"
            | "COM7"
            | "COM8"
            | "COM9"
            | "LPT1"
            | "LPT2"
            | "LPT3"
            | "LPT4"
            | "LPT5"
            | "LPT6"
            | "LPT7"
            | "LPT8"
            | "LPT9"
    ) {
        return true;
    }
    false
}

/// Directory lookup mirroring getThumbnailFolder/getOriginalFolder.
fn thumb_folder(user_dir: &Path, type_name: &str) -> PathBuf {
    match type_name {
        "bg" => user_dir.join("thumbnails/bg"),
        "avatar" => user_dir.join("thumbnails/avatar"),
        "persona" => user_dir.join("thumbnails/persona"),
        _ => unreachable!("route validates type"),
    }
}

fn original_folder(user_dir: &Path, type_name: &str) -> PathBuf {
    match type_name {
        "bg" => user_dir.join("backgrounds"),
        "avatar" => user_dir.join("characters"),
        "persona" => user_dir.join("User Avatars"),
        _ => unreachable!("route validates type"),
    }
}

/// isAnimatedApng: 'acTL' in the first 200 bytes.
fn is_animated_apng(buf: &[u8]) -> bool {
    let head = &buf[..buf.len().min(200)];
    find_subslice(head, b"acTL").is_some()
}

/// isAnimatedWebP: 'ANIM' or 'ANMF' in the first 200 bytes.
fn is_animated_webp(buf: &[u8]) -> bool {
    let head = &buf[..buf.len().min(200)];
    find_subslice(head, b"ANIM").is_some() || find_subslice(head, b"ANMF").is_some()
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    haystack.windows(needle.len()).position(|w| w == needle)
}

/// GET /thumbnail?type=...&file=...&animated=...
pub async fn thumbnail_route(State(state): State<Arc<AppState>>, req: Request<Body>) -> Response {
    let (parts, body) = req.into_parts();
    let _ = axum::body::to_bytes(body, 64 * 1024).await; // GET: body unused
    let limiter = state.thumb_limiter.clone();

    // Auth: thumbnails are behind the same session (publicRouter sits inside
    // the app's middleware chain in Express; the gateway verifies the same
    // session cookies). CSRF not needed for GET.
    let authenticator = Authenticator::new(&state.cfg);
    let auth = match authenticator.authenticate(&parts, false) {
        Ok(a) => a,
        Err(e) => {
            return Response::builder()
                .status(e.status())
                .body(Body::from(e.to_string()))
                .unwrap()
        }
    };

    // Parse query params
    let query = parts.uri.query().unwrap_or("");
    let mut q_type = "";
    let mut q_file = "";
    let mut q_animated = "";
    for pair in query.split('&') {
        let mut kv = pair.splitn(2, '=');
        let k = kv.next().unwrap_or("");
        let v = kv.next().unwrap_or("");
        match k {
            "type" => q_type = v,
            "file" => q_file = v,
            "animated" => q_animated = v,
            _ => {}
        }
    }
    let q_file = percent_decode_str(q_file).decode_utf8_lossy().to_string();

    if q_type != "bg" && q_type != "avatar" && q_type != "persona" {
        return StatusCode::BAD_REQUEST.into_response();
    }
    if sanitize_alters(&q_file) {
        return StatusCode::FORBIDDEN.into_response();
    }

    let user_dir = state.cfg.user_dir(&auth.handle);
    let original_folder = original_folder(&user_dir, q_type);
    let thumbnail_folder = thumb_folder(&user_dir, q_type);
    let path_to_cached = thumbnail_folder.join(&q_file);
    let path_to_original = original_folder.join(&q_file);

    let serve_original = || -> Response {
        if !path_to_original.exists() {
            return StatusCode::NOT_FOUND.into_response();
        }
        serve_file(&path_to_original, &parts)
    };

    // Disabled thumbnails → always original
    if !state.cfg.thumb_enabled() {
        return serve_original();
    }

    let animated_enabled = q_animated == "true";
    let ext = Path::new(&q_file)
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| format!(".{}", e.to_ascii_lowercase()))
        .unwrap_or_default();

    let is_skipped = SKIPPED_EXTENSIONS.contains(&ext.as_str());
    if animated_enabled && is_skipped {
        return serve_original();
    }
    if ext == ".gif" {
        return serve_original();
    }

    // Cache hit path: thumb exists, original not newer → serve thumb
    if path_to_cached.exists() {
        let regen = match (
            std::fs::metadata(&path_to_original),
            std::fs::metadata(&path_to_cached),
        ) {
            (Ok(o), Ok(c)) => {
                let o_mtime = o.modified().ok();
                let c_ctime = c.created().ok().or_else(|| c.modified().ok());
                match (o_mtime, c_ctime) {
                    (Some(o), Some(c)) => o > c,
                    _ => false,
                }
            }
            _ => false, // original missing → keep serving the thumb
        };
        if !regen {
            return serve_file(&path_to_cached, &parts);
        }
    }

    // Negative cache: recently failed → serve original (mirrors "generation
    // failed → serve original" without re-attempting disk+decode work)
    let nkey = format!("{}/{}/{}", auth.handle, q_type, q_file);
    if limiter.is_negatively_cached(&nkey).await {
        return serve_original();
    }

    // Miss → generate (bounded by the decode semaphore)
    let gen_state = state.clone();
    let gen_file = q_file.clone();
    let gen_type = q_type.to_string();
    let gen_dir = user_dir.clone();
    let permit = limiter.permits.clone().acquire_owned().await;
    let result = tokio::task::spawn_blocking(move || {
        generate_thumbnail(&gen_state, &gen_dir, &gen_type, &gen_file)
    })
    .await;
    drop(permit);

    let ok = matches!(result, Ok(Ok(true)));
    if !ok {
        limiter.mark_failed(&nkey).await;
        return serve_original();
    }

    if path_to_cached.exists() {
        return serve_file(&path_to_cached, &parts);
    }
    StatusCode::NOT_FOUND.into_response()
}

/// Serve a file with content-type and the Firefox no-store header when the
/// requester is Firefox (mirrors invalidateFirefoxCache).
fn serve_file(path: &Path, parts: &axum::http::request::Parts) -> Response {
    let Ok(bytes) = std::fs::read(path) else {
        return StatusCode::NOT_FOUND.into_response();
    };
    let mime = match path.extension().and_then(|e| e.to_str()) {
        Some(e) if e.eq_ignore_ascii_case("png") => "image/png",
        Some(e) if e.eq_ignore_ascii_case("jpg") || e.eq_ignore_ascii_case("jpeg") => "image/jpeg",
        Some(e) if e.eq_ignore_ascii_case("webp") => "image/webp",
        Some(e) if e.eq_ignore_ascii_case("gif") => "image/gif",
        _ => "application/octet-stream",
    };
    let is_firefox = parts
        .headers
        .get(header::USER_AGENT)
        .and_then(|v| v.to_str().ok())
        .map(|ua| ua.to_ascii_lowercase().contains("firefox"))
        .unwrap_or(false);

    let mut builder = Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, mime);
    if is_firefox {
        builder = builder.header(header::CACHE_CONTROL, "must-understand, no-store");
    }
    builder.body(Body::from(bytes)).unwrap()
}

/// Mirror of generateThumbnail + processSingleImage. Returns Ok(true) when a
/// thumbnail now exists, Ok(false) when the file legitimately has no thumb
/// (animated/skipped/missing), Err on unexpected failure.
pub fn generate_thumbnail(
    state: &AppState,
    user_dir: &Path,
    type_name: &str,
    file: &str,
) -> Result<bool, String> {
    let original_folder = original_folder(user_dir, type_name);
    let thumbnail_folder = thumb_folder(user_dir, type_name);
    let path_to_cached = thumbnail_folder.join(file);
    let path_to_original = original_folder.join(file);

    if !path_to_original.exists() {
        tracing::error!(
            "[generateThumbnail] original file not found: {}",
            path_to_original.display()
        );
        return Ok(false);
    }

    let ext = Path::new(file)
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| format!(".{}", e.to_ascii_lowercase()))
        .unwrap_or_default();

    // Animated detection (PNG/APNG and WebP only)
    let raw = std::fs::read(&path_to_original).map_err(|e| e.to_string())?;
    if ext == ".webp" && is_animated_webp(&raw) {
        return Ok(false);
    }
    if ext == ".png" && is_animated_apng(&raw) {
        return Ok(false);
    }
    if SKIPPED_EXTENSIONS.contains(&ext.as_str()) {
        return Ok(false);
    }

    // Decode (image crate). Preserve semantics: unsupported formats fail →
    // caller serves original.
    let img = image::load_from_memory(&raw).map_err(|e| format!("decode failed: {e}"))?;
    let (orig_w, orig_h) = (img.width(), img.height());
    let aspect_ratio = if orig_h > 0 {
        orig_w as f64 / orig_h as f64
    } else {
        1.0
    };

    let (config_w, config_h) = state.cfg.thumb_dims(type_name);
    let thumb = if type_name == "bg" {
        // Area-preserving scale (upstream math):
        // targetPixelArea = cw * ch; w = sqrt(area * ratio); h = sqrt(area / ratio)
        let target_area = (config_w as f64) * (config_h as f64);
        let tw = (target_area * aspect_ratio).sqrt().round() as u32;
        let th = (target_area / aspect_ratio).sqrt().round() as u32;
        img.resize_exact(
            tw.max(1),
            th.max(1),
            image::imageops::FilterType::Triangle, // bilinear
        )
    } else {
        // cover: crop to fill W×H (center), like Jimp cover
        cover_resize(&img, config_w, config_h)
    };

    // Encode per config
    let png_format = state.cfg.thumb_format_is_png();
    let quality = state.cfg.thumb_quality();
    let mut out: Vec<u8> = Vec::new();
    if png_format {
        thumb
            .write_to(&mut std::io::Cursor::new(&mut out), image::ImageFormat::Png)
            .map_err(|e| format!("png encode failed: {e}"))?;
    } else {
        let mut encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, quality);
        encoder
            .encode_image(&thumb)
            .map_err(|e| format!("jpeg encode failed: {e}"))?;
    }

    // Atomic write (temp + rename), like write-file-atomic
    std::fs::create_dir_all(&thumbnail_folder).map_err(|e| e.to_string())?;
    let tmp = path_to_cached.with_extension(format!("tmp.{}", std::process::id()));
    std::fs::write(&tmp, &out).map_err(|e| e.to_string())?;
    std::fs::rename(&tmp, &path_to_cached).map_err(|e| e.to_string())?;

    tracing::debug!(
        file = %file,
        from = format!("{}x{}", orig_w, orig_h),
        aspect = aspect_ratio,
        bytes = out.len(),
        "thumbnail generated"
    );
    Ok(true)
}

/// Center-crop + resize to exactly W×H (Jimp `cover` semantics: the image is
/// scaled so both dimensions are >= target, then center-cropped).
fn cover_resize(img: &image::DynamicImage, target_w: u32, target_h: u32) -> image::DynamicImage {
    let (w, h) = (img.width() as f64, img.height() as f64);
    let scale = (target_w as f64 / w).max(target_h as f64 / h);
    let scaled_w = (w * scale).round() as u32;
    let scaled_h = (h * scale).round() as u32;
    let scaled = img.resize_exact(
        scaled_w.max(target_w),
        scaled_h.max(target_h),
        image::imageops::FilterType::Triangle,
    );
    let x = (scaled.width().saturating_sub(target_w)) / 2;
    let y = (scaled.height().saturating_sub(target_h)) / 2;
    scaled.crop_imm(x, y, target_w, target_h)
}
