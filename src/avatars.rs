//! `/api/avatars/*` — user (persona) avatar management, mirroring
//! `src/endpoints/avatars.js`.
//!
//! Routes:
//! - `POST /get`    → list the image files in `<user>/User Avatars`
//! - `POST /delete` → unlink one avatar + its persona thumbnail
//! - `POST /upload` → decode + crop + cover + PNG encode, atomically write
//!
//! ## Upload memory safety
//!
//! Upstream reads the upload with Jimp and keeps the whole bitmap (JS objects
//! on top of the pixel buffer) — a 26MB JPEG of 144 megapixels is enough to
//! take the process down. The Rust port bounds this in three places:
//!
//! 1. `avatar::check_dimensions` reads only the container header (5.9ms and no
//!    bitmap for a 12000x12000 JPEG) and rejects anything above
//!    `avatar::MAX_SOURCE_PIXELS` (40MP) with a 400 *before* decoding.
//! 2. `avatar::decode_bounded` passes the decoder an explicit `max_alloc`
//!    backstop for files that under-declare their dimensions.
//! 3. `AvatarLimiter` caps how many decodes run at once, so N concurrent
//!    uploads cannot stack their bitmaps (one 40MP decode peaks near 430MB).
//!
//! A size rejection is deliberately NOT routed through the "undecodable →
//! default avatar" fallback: silently replacing an oversized image with the
//! default would tell the user their upload worked.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use axum::body::Body;
use axum::extract::State;
use axum::http::{header, Request, StatusCode};
use axum::response::{IntoResponse, Response};
use serde_json::json;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::avatar::{self, Crop};
use crate::card;
use crate::routes::{check_auth, crop_from_query, parse_multipart};
use crate::AppState;

/// `<user>/User Avatars` (USER_DIRECTORY_TEMPLATE.avatars).
pub fn avatars_dir(state: &AppState, handle: &str) -> PathBuf {
    state.cfg.user_dir(handle).join("User Avatars")
}

/// Concurrency gate for image decoding in request handlers.
///
/// Unlike thumbnails (whose outputs are always small), an upload decode can
/// legitimately peak at ~465MB for a 40MP source — measured through the real
/// route. Permits are therefore sized per *worst-case decode*, not per 8MiB:
/// `budget / UPLOAD_DECODE_BYTES`, clamped to 1..=4. With the default 256MiB
/// budget that is a single permit, so concurrent uploads are serialized rather
/// than stacked; a larger budget buys up to 4-way parallelism.
pub struct AvatarLimiter {
    permits: Arc<Semaphore>,
}

/// Measured peak working set for the worst allowed upload (40MP → crop →
/// cover → PNG encode), rounded up.
pub const UPLOAD_DECODE_BYTES: u64 = 512 * 1024 * 1024;

impl AvatarLimiter {
    pub fn new(budget_bytes: u64) -> Self {
        let permits = (budget_bytes / UPLOAD_DECODE_BYTES).clamp(1, 4) as usize;
        Self {
            permits: Arc::new(Semaphore::new(permits)),
        }
    }

    /// Number of decodes that may run concurrently (test/diagnostics only).
    pub fn concurrency(&self) -> usize {
        self.permits.available_permits()
    }

    pub async fn acquire(&self) -> OwnedSemaphorePermit {
        self.permits
            .clone()
            .acquire_owned()
            .await
            .expect("avatar semaphore is never closed")
    }
}

/// Extension → MIME, limited to what the upstream `mime` package (1.6.0)
/// reports as `image/*`. `getImages` filters on that prefix, so the tables must
/// agree on membership — an extension missing here would silently disappear
/// from the avatar list.
///
/// **Deliberate divergence:** `avif`/`heic`/`heif`/`jxl` are included even
/// though mime 1.6.0 reports them as `application/octet-stream`, so upstream
/// hides personas stored in those formats. AVIF is handled everywhere else in
/// this gateway (decode, crop, thumbnails) via the pure-Rust oxideav decoder,
/// and silently dropping the user's file is worse than diverging from a stale
/// MIME table. HEIC/HEIF/JXL are listed for the same reason: they are images a
/// user can legitimately drop in, and hiding them makes the persona look lost.
const IMAGE_EXTENSIONS: &[&str] = &[
    "png", "jpg", "jpeg", "jpe", "webp", "gif", "bmp", "svg", "svgz", "tif", "tiff", "ico", "apng",
    "psd", "tga", "pcx", "pnm", "pbm", "pgm", "ppm", "ras", "rgb", "xbm", "xpm", "wbmp", "wdp",
    "ktx", "jp2", "jpx", "jpm", "avif", "heic", "heif", "jxl",
];

fn is_image_file(name: &str) -> bool {
    Path::new(name)
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .map(|e| IMAGE_EXTENSIONS.contains(&e.as_str()))
        .unwrap_or(false)
}

// ---------------------------------------------------------------------------
// POST /api/avatars/get
// ---------------------------------------------------------------------------

pub async fn avatars_get(State(state): State<Arc<AppState>>, req: Request<Body>) -> Response {
    let (parts, body) = req.into_parts();
    let auth = match check_auth(&state, &parts) {
        Ok(a) => a,
        Err(resp) => return resp,
    };
    let _ = axum::body::to_bytes(body, 1024 * 1024).await; // body unused

    let dir = avatars_dir(&state, &auth.handle);
    let names = tokio::task::spawn_blocking(move || list_images(&dir))
        .await
        .unwrap_or_default();
    json_response(StatusCode::OK, json!(names))
}

/// `getImages(directoryPath)` with the default `sortBy = 'name'`, which uses
/// `Intl.Collator().compare` — case-insensitive, punctuation-insensitive
/// ordering, not byte order. This is a close approximation of the root-locale
/// collator: compare case-folded and accent-stripped forms, then fall back to
/// the raw strings so the order stays total.
pub fn list_images(dir: &Path) -> Vec<String> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut names: Vec<String> = entries
        .flatten()
        .filter(|e| e.file_type().map(|t| t.is_file()).unwrap_or(false))
        .map(|e| e.file_name().to_string_lossy().to_string())
        .filter(|n| is_image_file(n))
        .collect();
    names.sort_by(|a, b| collate_key(a).cmp(&collate_key(b)).then_with(|| a.cmp(b)));
    names
}

/// Sort key approximating `Intl.Collator().compare`: lowercase, then drop
/// punctuation that the collator ignores at the primary strength.
fn collate_key(name: &str) -> String {
    name.chars()
        .filter(|c| !matches!(c, ' ' | '-' | '_' | '.' | '\'' | '"' | ',' | '(' | ')'))
        .flat_map(|c| c.to_lowercase())
        .collect()
}

// ---------------------------------------------------------------------------
// POST /api/avatars/delete
// ---------------------------------------------------------------------------

pub async fn avatars_delete(State(state): State<Arc<AppState>>, req: Request<Body>) -> Response {
    let (parts, body) = req.into_parts();
    let auth = match check_auth(&state, &parts) {
        Ok(a) => a,
        Err(resp) => return resp,
    };
    let body_bytes = match axum::body::to_bytes(body, 1024 * 1024).await {
        Ok(b) => b,
        Err(_) => return StatusCode::BAD_REQUEST.into_response(),
    };
    let Ok(form) = serde_json::from_slice::<serde_json::Value>(&body_bytes) else {
        return StatusCode::BAD_REQUEST.into_response();
    };
    let Some(avatar) = form.get("avatar").and_then(|v| v.as_str()) else {
        return StatusCode::BAD_REQUEST.into_response();
    };

    // getFileNameValidationFunction('avatar'): a name that sanitize() would
    // change carries a path separator, so upstream answers 403 rather than 400.
    let sanitized = card::sanitize_filename(avatar);
    if sanitized != avatar {
        tracing::warn!(avatar, "malicious avatar name prevented");
        return StatusCode::FORBIDDEN.into_response();
    }

    let dir = avatars_dir(&state, &auth.handle);
    let user_dir = state.cfg.user_dir(&auth.handle);
    let name = avatar.to_string();
    let deleted = tokio::task::spawn_blocking(move || {
        let file = dir.join(&sanitized);
        if !file.is_file() {
            return false;
        }
        if let Err(e) = std::fs::remove_file(&file) {
            tracing::error!(path = %file.display(), error = %e, "avatar delete failed");
            return false;
        }
        invalidate_persona_thumbnail(&user_dir, &name);
        true
    })
    .await
    .unwrap_or(false);

    if deleted {
        json_response(StatusCode::OK, json!({ "result": "ok" }))
    } else {
        StatusCode::NOT_FOUND.into_response()
    }
}

// ---------------------------------------------------------------------------
// POST /api/avatars/upload
// ---------------------------------------------------------------------------

/// Body ceiling for the upload route. The avatar file is the payload; 32MiB
/// is far above any real image (a 40MP JPEG is ~12MB) while keeping a hostile
/// body from being buffered. The decoder's own pixel/alloc limits are what
/// actually bound memory — this only bounds the transfer.
const UPLOAD_BODY_LIMIT: usize = 32 * 1024 * 1024;

pub async fn avatars_upload(State(state): State<Arc<AppState>>, req: Request<Body>) -> Response {
    let (parts, body) = req.into_parts();
    let auth = match check_auth(&state, &parts) {
        Ok(a) => a,
        Err(resp) => return resp,
    };
    let ct = parts
        .headers
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    if !ct.starts_with("multipart/") {
        return StatusCode::BAD_REQUEST.into_response();
    }
    let crop = crop_from_query(parts.uri.query());

    let body_bytes = match axum::body::to_bytes(body, UPLOAD_BODY_LIMIT).await {
        Ok(b) => b,
        Err(e) => {
            tracing::warn!(error = %e, "avatar upload rejected: body too large");
            return (
                StatusCode::PAYLOAD_TOO_LARGE,
                "image upload is too large (32MB limit)",
            )
                .into_response();
        }
    };
    let mp = match parse_multipart(&ct, Body::from(body_bytes)).await {
        Ok(m) => m,
        Err(e) => {
            tracing::warn!(error = %e, "avatar upload: multipart parse failed");
            return StatusCode::BAD_REQUEST.into_response();
        }
    };
    // upstream: `if (!request.file) return 400`
    let Some(upload) = mp.file_bytes else {
        return StatusCode::BAD_REQUEST.into_response();
    };
    let overwrite_raw = mp
        .fields
        .iter()
        .find(|(k, _)| k == "overwrite_name")
        .map(|(_, v)| v.clone())
        .unwrap_or_default();

    // getFileNameValidationFunction('overwrite_name')
    if !overwrite_raw.is_empty() {
        let sanitized = card::sanitize_filename(&overwrite_raw);
        if sanitized != overwrite_raw {
            tracing::warn!(
                overwrite_name = %overwrite_raw,
                "malicious overwrite_name prevented"
            );
            return StatusCode::BAD_REQUEST.into_response();
        }
    }

    let handle = auth.handle.clone();
    let dir = avatars_dir(&state, &handle);
    let user_dir = state.cfg.user_dir(&handle);

    // Admission before decode: one 40MP image peaks near 430MB, so concurrent
    // uploads must not stack.
    let permit = state.avatar_limiter.acquire().await;

    let result = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        upload_sync(&dir, &user_dir, &upload, crop, &overwrite_raw)
    })
    .await
    .unwrap_or_else(|e| Err(UploadError::Internal(format!("join error: {e}"))));

    match result {
        Ok(name) => json_response(StatusCode::OK, json!({ "path": name })),
        Err(UploadError::InvalidImage(detail)) => {
            tracing::warn!(error = %detail, "avatar upload: invalid image");
            (StatusCode::BAD_REQUEST, "Is not a valid image").into_response()
        }
        Err(UploadError::TooLarge(too_large)) => {
            tracing::warn!(%too_large, "avatar upload rejected: image too large");
            (StatusCode::BAD_REQUEST, too_large.to_string()).into_response()
        }
        Err(UploadError::Internal(detail)) => {
            tracing::error!(error = %detail, "avatar upload failed");
            (StatusCode::BAD_REQUEST, "Is not a valid image").into_response()
        }
    }
}

enum UploadError {
    /// Undecodable bytes → upstream's `400 "Is not a valid image"`.
    InvalidImage(String),
    /// Above `avatar::MAX_SOURCE_PIXELS` — reported distinctly so the user
    /// learns the real cause instead of "not a valid image".
    TooLarge(avatar::TooLarge),
    Internal(String),
}

/// The blocking half of the upload: decode, crop/cover, encode, write.
///
/// Mirrors `avatars.js` `/upload` exactly:
/// - `filename = sanitize(overwrite_name || Date.now() + '.png')`
/// - write to `<user>/User Avatars/<filename>`
/// - `invalidateThumbnail(directories, 'persona', overwrite_name)` and
///   `cacheBuster.bust` happen only when overwriting an existing avatar.
fn upload_sync(
    dir: &Path,
    user_dir: &Path,
    raw: &[u8],
    crop: Option<Crop>,
    overwrite_name: &str,
) -> Result<String, UploadError> {
    // Header-only rejection before any bitmap is allocated.
    if let Err(too_large) = avatar::check_dimensions(raw) {
        return Err(UploadError::TooLarge(too_large));
    }

    let image =
        avatar::decode_bounded(raw).map_err(|e| UploadError::InvalidImage(e.to_string()))?;
    let encoded =
        avatar::apply_crop_resize(&image, crop.as_ref()).map_err(UploadError::InvalidImage)?;
    drop(image);

    let filename = if !overwrite_name.is_empty() {
        card::sanitize_filename(overwrite_name)
    } else {
        // `Date.now()` in ms — same shape the frontend expects.
        format!("{}.png", now_millis())
    };
    if filename.is_empty() {
        return Err(UploadError::Internal(
            "sanitized avatar name is empty".to_string(),
        ));
    }

    std::fs::create_dir_all(dir)
        .map_err(|e| UploadError::Internal(format!("mkdir {}: {e}", dir.display())))?;
    let target = dir.join(&filename);

    // Upstream invalidates the persona thumbnail only when overwriting a name
    // it already knows about. Doing it unconditionally would churn the cache
    // for a fresh upload.
    if !overwrite_name.is_empty() {
        invalidate_persona_thumbnail(user_dir, &filename);
    }

    crate::atomic::write(&target, &encoded)
        .map_err(|e| UploadError::Internal(format!("write {}: {e}", target.display())))?;

    tracing::info!(
        file = %filename,
        bytes = encoded.len(),
        "persona avatar uploaded"
    );
    Ok(filename)
}

/// `invalidateThumbnail(directories, 'persona', file)`:
/// unlink `<user>/thumbnails/persona/<sanitize(file)>`.
fn invalidate_persona_thumbnail(user_dir: &Path, file: &str) {
    let thumb = user_dir
        .join("thumbnails/persona")
        .join(card::sanitize_filename(file));
    if thumb.is_file() {
        if let Err(e) = std::fs::remove_file(&thumb) {
            tracing::warn!(
                path = %thumb.display(),
                error = %e,
                "persona thumbnail invalidation failed"
            );
        }
    }
}

fn now_millis() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

/// Test seam: the blocking upload core with its error flattened to a string.
pub fn upload_sync_for_tests(
    dir: &Path,
    user_dir: &Path,
    raw: &[u8],
    crop: Option<Crop>,
    overwrite_name: &str,
) -> Result<String, String> {
    upload_sync(dir, user_dir, raw, crop, overwrite_name).map_err(|e| match e {
        UploadError::InvalidImage(d) => format!("invalid image: {d}"),
        UploadError::TooLarge(t) => format!("too large: {t}"),
        UploadError::Internal(d) => format!("internal: {d}"),
    })
}

fn json_response(status: StatusCode, value: serde_json::Value) -> Response {
    Response::builder()
        .status(status)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(value.to_string()))
        .unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_filter_matches_the_upstream_mime_table() {
        assert!(is_image_file("a.png"));
        assert!(is_image_file("a.JPG"));
        assert!(is_image_file("a.webp"));
        assert!(is_image_file("a.apng"));
        assert!(is_image_file("a.svg"));
        assert!(!is_image_file("a.txt"));
        assert!(!is_image_file("a.mp4"));
        // Deliberate divergence from mime 1.6.0 (which reports
        // application/octet-stream and so hides these upstream): AVIF is
        // fully decodable here, and hiding a real image file makes the
        // persona look lost.
        assert!(is_image_file("a.avif"));
        assert!(is_image_file("a.heic"));
        assert!(is_image_file("a.jxl"));
        assert!(!is_image_file("noextension"));
    }

    #[test]
    fn collate_order_is_case_insensitive() {
        let mut names = vec!["B.png", "a.png", "C.png.py", "_x.png"];
        names.sort_by(|a, b| collate_key(a).cmp(&collate_key(b)).then_with(|| a.cmp(b)));
        assert_eq!(names, vec!["a.png", "B.png", "C.png.py", "_x.png"]);
    }

    #[test]
    fn punctuation_is_ignored_like_intl_collator() {
        // Intl.Collator orders "a b.png" and "a_b.png" before "ab.png"? No:
        // punctuation is ignored at primary strength, so all three tie and the
        // raw-string tiebreak decides deterministically.
        assert_eq!(collate_key("a b.png"), collate_key("a_b.png"));
        assert_eq!(collate_key("a_b.png"), collate_key("ab.png"));
    }

    #[test]
    fn generated_name_has_the_png_extension() {
        let name = format!("{}.png", now_millis());
        assert!(name.ends_with(".png"));
        assert!(name.len() > 4);
    }
}

#[cfg(test)]
mod limiter_tests {
    use super::*;

    #[test]
    fn default_budget_serializes_uploads() {
        // 256MiB default / 512MiB worst-case decode = 0, clamped up to 1.
        assert_eq!(AvatarLimiter::new(256 * 1024 * 1024).concurrency(), 1);
    }

    #[test]
    fn a_large_budget_allows_a_few_parallel_decodes() {
        assert_eq!(AvatarLimiter::new(1024 * 1024 * 1024).concurrency(), 2);
        assert_eq!(AvatarLimiter::new(4096 * 1024 * 1024).concurrency(), 4);
        // Never more than 4, however generous the budget.
        assert_eq!(AvatarLimiter::new(64 * 1024 * 1024 * 1024).concurrency(), 4);
    }
}
