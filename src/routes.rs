//! Router composition: auth middleware, local migrated routes, and
//! transparent forwarding fallback.

use std::sync::Arc;

use axum::body::Body;
use axum::extract::State;
use axum::http::{Request, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use serde_json::json;

use crate::auth::{AuthError, Authenticator};
use crate::avatar;
use crate::disk_cache;
use crate::{card, forward, png, AppState};

/// Methods that require CSRF validation (csrf-sync: everything except
/// GET/HEAD/OPTIONS).
fn requires_csrf(method: &axum::http::Method) -> bool {
    !matches!(
        *method,
        axum::http::Method::GET | axum::http::Method::HEAD | axum::http::Method::OPTIONS
    )
}

/// Extract auth from request parts; on failure produce the error response.
fn check_auth(
    state: &AppState,
    parts: &axum::http::request::Parts,
) -> Result<crate::auth::AuthContext, Response> {
    let authenticator = Authenticator::new(&state.cfg);
    authenticator
        .authenticate(parts, requires_csrf(&parts.method))
        .map_err(|e: AuthError| {
            // Auth failures MUST be visible: log cookie-name suffix and whether
            // session/sig cookies were present to make mismatch diagnosis trivial.
            let has_session = parts
                .headers
                .get(axum::http::header::COOKIE)
                .and_then(|v| v.to_str().ok())
                .map(|c| c.contains(&authenticator.session_cookie))
                .unwrap_or(false);
            tracing::warn!(
                path = %parts.uri.path(),
                error = %e,
                expected_cookie = %authenticator.session_cookie,
                session_cookie_present = has_session,
                "auth rejected"
            );
            Response::builder()
                .status(e.status())
                .body(Body::from(e.to_string()))
                .unwrap()
        })
}

pub fn build_router(state: Arc<AppState>) -> axum::Router {
    let thumbnail_local = state.cfg.route_mode("/thumbnail") == crate::config::RouteMode::Local;
    let mut router = axum::Router::new();
    if thumbnail_local {
        router = router.route(
            "/thumbnail",
            axum::routing::get(crate::thumbnails::thumbnail_route),
        );
    }
    router
        .route("/api/characters/all", post(characters_all))
        .route("/api/characters/get", post(characters_get))
        .route("/api/characters/edit", post(characters_edit_full))
        .route(
            "/api/characters/edit-attribute",
            post(characters_edit_attribute),
        )
        .route("/api/characters/rename", post(characters_rename))
        .route("/api/characters/export", post(characters_export))
        .route("/api/characters/create", post(characters_create))
        .route("/api/characters/edit-avatar", post(characters_edit_avatar))
        .route("/api/characters/import", post(characters_import))
        .route("/csrf-token", get(csrf_token_forward))
        .fallback(fallback)
        .with_state(state)
}

async fn fallback(State(state): State<Arc<AppState>>, req: Request<Body>) -> Response {
    forward::proxy(state, req).await
}

async fn csrf_token_forward(State(state): State<Arc<AppState>>, req: Request<Body>) -> Response {
    forward::proxy(state, req).await
}

fn user_dirs(state: &AppState, handle: &str) -> card::CardDeps {
    let user = state.cfg.user_dir(handle);
    card::CardDeps {
        characters_dir: user.join("characters"),
        chats_dir: user.join("chats"),
        worlds_dir: user.join("worlds"),
    }
}

/// POST /api/characters/all — local implementation.
async fn characters_all(State(state): State<Arc<AppState>>, req: Request<Body>) -> Response {
    let (parts, body) = req.into_parts();
    let auth = match check_auth(&state, &parts) {
        Ok(a) => a,
        Err(resp) => return resp,
    };
    let _ = axum::body::to_bytes(body, 1024 * 1024).await; // body unused

    let deps = user_dirs(&state, &auth.handle);
    // lazyLoadCharacters mirror: shallow list payload (upstream
    // performance.lazyLoadCharacters, frontend unshallows via /characters/get)
    let shallow = state
        .cfg
        .st
        .performance
        .as_ref()
        .and_then(|p| p.lazyLoadCharacters)
        .unwrap_or(false);
    let names: Vec<String> = match std::fs::read_dir(&deps.characters_dir) {
        Ok(rd) => rd
            .flatten()
            .filter(|e| e.path().extension().map(|x| x == "png").unwrap_or(false))
            .filter_map(|e| e.file_name().into_string().ok())
            .collect(),
        Err(_) => Vec::new(),
    };
    // JS sort order: none applied upstream (fs.readdirSync order). Keep as-is.

    // Streamed JSON array response. Files are batched by total file size:
    // a batch accumulates entries until adding the next would exceed
    // batch_bytes (a single oversized file is allowed to run alone). Within a
    // batch, PNGs parse concurrently; results are emitted in readdir order so
    // output stays identical to the upstream serial path. Peak memory ≈
    // batch_bytes of raw file IO (tEXt only is retained; IDAT is seek-skipped
    // and never buffered).
    let batch_bytes = state
        .cfg
        .gateway
        .batch_bytes
        .unwrap_or(DEFAULT_BATCH_BYTES)
        .max(1);
    let (sender, body_stream) =
        tokio::sync::mpsc::channel::<Result<bytes::Bytes, std::io::Error>>(16);
    let data_root = state.cfg.data_root.clone();
    let handle = auth.handle.clone();
    tokio::spawn(async move {
        let _ = sender.send(Ok(bytes::Bytes::from_static(b"[\n"))).await;
        let mut first = true;

        // Size-scan once (cheap stat per file) to drive size-based batching.
        let entries: Vec<(String, u64)> = names
            .iter()
            .filter_map(|n| {
                let size = std::fs::metadata(deps.characters_dir.join(n)).ok()?.len();
                Some((n.clone(), size))
            })
            .collect();

        for batch in plan_batches(&entries, batch_bytes) {
            process_batch(
                &batch,
                &deps,
                batch_bytes,
                shallow,
                &data_root,
                &handle,
                &mut first,
                &sender,
            )
            .await;
        }
        let _ = sender.send(Ok(bytes::Bytes::from_static(b"\n]"))).await;
    });

    let stream = tokio_stream_from_channel(body_stream);
    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "application/json")
        .body(Body::from_stream(stream))
        .unwrap()
}

fn tokio_stream_from_channel(
    rx: tokio::sync::mpsc::Receiver<Result<bytes::Bytes, std::io::Error>>,
) -> impl futures_util::Stream<Item = Result<bytes::Bytes, std::io::Error>> {
    tokio_stream::wrappers::ReceiverStream::new(rx)
}

/// Default per-batch total file size: 256 MiB.
const DEFAULT_BATCH_BYTES: u64 = 256 * 1024 * 1024;

/// Batch entry: file name + file size.
type BatchEntry = (String, u64);

/// Split entries into batches whose summed file sizes stay within
/// `batch_bytes`. Rules:
/// - accumulate in order until the next entry would overflow, then close the
///   batch;
/// - a single file larger than the limit is allowed as a one-entry batch
///   (never rejected);
/// - a zero-size file always joins the current batch.
fn plan_batches(entries: &[BatchEntry], batch_bytes: u64) -> Vec<Vec<BatchEntry>> {
    let mut batches = Vec::new();
    let mut current: Vec<BatchEntry> = Vec::new();
    let mut current_sum: u64 = 0;
    for entry in entries {
        let fits = current_sum + entry.1 <= batch_bytes;
        let current_empty = current.is_empty();
        if fits || current_empty {
            current_sum += entry.1;
            current.push(entry.clone());
        } else {
            batches.push(std::mem::take(&mut current));
            current_sum = entry.1;
            current.push(entry.clone());
        }
    }
    if !current.is_empty() {
        batches.push(current);
    }
    batches
}

/// Stub emitted for files that fail to process (mirrors the upstream catch
/// branch of `processCharacter`).
fn zeroed_stub() -> serde_json::Value {
    json!({ "date_added": 0, "date_last_chat": 0, "chat_size": 0 })
}

/// Cache-aware single-card processing, mirroring the upstream
/// readCharacterData → processCharacter pipeline:
///
/// 1. key = `<charactersDir>/<item>-<mtimeMs>` (identical to Node's
///    getCacheKey, so both processes share entries)
/// 2. disk hit → parse cached JSON, skip PNG chunk scanning entirely
/// 3. miss → full PNG parse (card JSON extracted), write back to disk
///
/// Note: unlike Node's memoryCache, cached value is the raw card JSON
/// string; the avatar/json_data/stats fields are re-applied per request
/// (they depend on the item name and current chat state).
fn process_character_cached(
    item: &str,
    deps: &card::CardDeps,
    cache: &disk_cache::DiskCache,
    data_root: &std::path::Path,
    handle: &str,
) -> serde_json::Value {
    let file_path = deps.characters_dir.join(item);

    // Derive the cache key from the file's current mtime.
    let key = match std::fs::metadata(&file_path) {
        Ok(meta) => {
            let (secs, nanos) = disk_cache::mtime_secs_nanos(&meta);
            disk_cache::cache_key(
                data_root,
                handle,
                item,
                &disk_cache::format_mtime_ms(secs, nanos),
            )
        }
        Err(_) => {
            // File gone: fall through to the normal error path
            return card::process_character(item, deps);
        }
    };

    // 1) disk hit: raw card JSON from cache → rebuild the response object
    if let Some(cached_json) = cache.get(&key) {
        match rebuild_response_from_cached(item, &cached_json, deps, &file_path) {
            Ok(v) => return v,
            Err(e) => {
                tracing::warn!(file = %item, error = %e, "cached entry unusable; reparsing");
            }
        }
    }

    // 2) miss: full PNG parse
    let value = card::process_character(item, deps);

    // 3) write back only on success (stubs are never cached, matching Node
    //    which caches `result` unconditionally — but our stub lacks imgData,
    //    so caching it would poison the entry)
    if value.get("name").is_some() {
        if let Some(raw) = value.get("json_data").and_then(|v| v.as_str()) {
            if let Err(e) = cache.put(&key, raw) {
                tracing::warn!(file = %item, error = %e, "disk cache write failed");
            }
        }
    }
    value
}

/// Rebuild the `/all` `/get` response object from a cached card JSON string,
/// applying the per-request fields (avatar/json_data/stats) exactly like
/// process_character_inner does after parsing.
fn rebuild_response_from_cached(
    item: &str,
    cached_json: &str,
    deps: &card::CardDeps,
    file_path: &std::path::Path,
) -> Result<serde_json::Value, String> {
    let parsed: serde_json::Value =
        serde_json::from_str(cached_json).map_err(|e| format!("cached JSON invalid: {e}"))?;
    let value = card::process_character_from_card(item, parsed, deps, file_path)?;
    // json_data must be the verbatim original card JSON — the cached string
    // IS that string (cache stores only successful parses' raw JSON).
    Ok(card::with_json_data(value, cached_json))
}

/// Process one planned batch: parse all files concurrently, emit results in
/// batch order (upstream readdir order preserved).
///
/// Cache: before parsing, each file is looked up in the shared node-persist-
/// compatible DiskCache (key = path-mtimeMs, same as Node); hits skip PNG
/// parsing entirely. Misses parse, then write the raw card JSON back so Node
/// can also reuse the entry.
///
/// Failure policy:
/// - if the whole batch fails (every entry errors), the batch is split in
///   half and each half retried recursively (down to single files);
/// - a single file that fails is reported as a zeroed stub and NOT retried.
async fn process_batch(
    batch: &[BatchEntry],
    deps: &card::CardDeps,
    batch_bytes: u64,
    shallow: bool,
    data_root: &std::path::Path,
    handle: &str,
    first: &mut bool,
    sender: &tokio::sync::mpsc::Sender<Result<bytes::Bytes, std::io::Error>>,
) {
    let cache = disk_cache::DiskCache::new(data_root);

    // Run the batch concurrently.
    let tasks: Vec<_> = batch
        .iter()
        .cloned()
        .map(|(name, _size)| {
            let deps2 = deps.clone();
            let data_root = data_root.to_path_buf();
            let handle = handle.to_string();
            let cache2 = cache.clone();
            tokio::task::spawn_blocking(move || {
                process_character_cached(&name, &deps2, &cache2, &data_root, &handle)
            })
        })
        .collect();

    let mut results: Vec<(String, Result<serde_json::Value, ()>)> = Vec::with_capacity(tasks.len());
    for (entry, task) in batch.iter().zip(tasks) {
        let name = entry.0.clone();
        let value = match task.await {
            Ok(v) => Ok(v),
            Err(join_err) => {
                tracing::error!(file = %name, error = %join_err, "worker panicked");
                Err(())
            }
        };
        results.push((name, value));
    }

    let total = results.len();
    let failed: Vec<&String> = results
        .iter()
        .filter(|(_, r)| r.is_err())
        .map(|(n, _)| n)
        .collect();

    // Whole-batch failure: split in half and retry each half recursively.
    if total > 1 && failed.len() == total {
        tracing::warn!(
            batch_size = total,
            batch_bytes,
            "entire batch failed; splitting and retrying halves"
        );
        let mid = batch.len() / 2;
        let (left, right) = batch.split_at(mid);
        if !left.is_empty() {
            Box::pin(process_batch(
                left,
                deps,
                batch_bytes,
                shallow,
                data_root,
                handle,
                first,
                sender,
            ))
            .await;
        }
        if !right.is_empty() {
            Box::pin(process_batch(
                right,
                deps,
                batch_bytes,
                shallow,
                data_root,
                handle,
                first,
                sender,
            ))
            .await;
        }
        return;
    }

    // Emit in batch order; single-file failures become stubs without retry.
    for (name, result) in results {
        let value = match result {
            Ok(v) => v,
            Err(()) => {
                tracing::warn!(file = %name, "single-file failure; emitting stub without retry");
                zeroed_stub()
            }
        };

        // Upstream filters !char.name
        let has_name = value.get("name").map(|n| !n.is_null()).unwrap_or(false);
        if !has_name {
            continue;
        }
        // lazyLoadCharacters mirror: shallow list payload (upstream toShallow)
        let value = if shallow {
            card::to_shallow(&value)
        } else {
            value
        };
        let serialized = serde_json::to_string(&value).unwrap_or_default();
        let chunk = if *first {
            *first = false;
            serialized
        } else {
            format!(",\n{serialized}")
        };
        let _ = sender.send(Ok(bytes::Bytes::from(chunk))).await;
    }
}

/// POST /api/characters/get — local implementation.
async fn characters_get(State(state): State<Arc<AppState>>, req: Request<Body>) -> Response {
    let (parts, body) = req.into_parts();
    let auth = match check_auth(&state, &parts) {
        Ok(a) => a,
        Err(resp) => return resp,
    };
    let body_bytes = match axum::body::to_bytes(body, 1024 * 1024).await {
        Ok(b) => b,
        Err(_) => return StatusCode::BAD_REQUEST.into_response(),
    };
    let parsed: serde_json::Value = match serde_json::from_slice(&body_bytes) {
        Ok(v) => v,
        Err(_) => return StatusCode::BAD_REQUEST.into_response(),
    };
    let avatar_url = match parsed.get("avatar_url").and_then(|v| v.as_str()) {
        Some(s) => s.to_string(),
        None => return StatusCode::BAD_REQUEST.into_response(),
    };

    // validateAvatarUrlMiddleware semantics are enforced here as a safety net.
    if avatar_url != sanitize_like(&avatar_url)
        || avatar_url.contains("..")
        || avatar_url.contains('/')
        || avatar_url.contains('\\')
    {
        return StatusCode::FORBIDDEN.into_response();
    }

    let deps = user_dirs(&state, &auth.handle);
    let file_path = deps.characters_dir.join(&avatar_url);
    if !file_path.exists() {
        return StatusCode::NOT_FOUND.into_response();
    }

    // Cache-aware single-card read (same shared cache as /all and Node)
    let data_root = state.cfg.data_root.clone();
    let handle = auth.handle.clone();
    let value = tokio::task::spawn_blocking(move || {
        let cache = disk_cache::DiskCache::new(&data_root);
        process_character_cached(&avatar_url, &deps, &cache, &data_root, &handle)
    })
    .await
    .unwrap_or_else(|_| json!({ "date_added": 0, "date_last_chat": 0, "chat_size": 0 }));

    (
        [(axum::http::header::CONTENT_TYPE, "application/json")],
        serde_json::to_string(&value).unwrap_or_default(),
    )
        .into_response()
}

/// Minimal sanitize-filename-like check for avatar file names (defense in
/// depth on the local path; the real validation mirror lives with edit).
fn sanitize_like(name: &str) -> String {
    // Reject control chars and reserved Windows characters the way
    // sanitize-filename replaces them; if replacement would change the name,
    // the comparison in the caller fails.
    name.chars()
        .map(|c| {
            if c.is_control() || matches!(c, '/' | '\\' | '?' | '*' | ':' | '|' | '"' | '<' | '>') {
                '\u{FFFD}'
            } else {
                c
            }
        })
        .collect()
}

/// Mirrors /edit flow: charaFormatData(request.body) on the *existing* card
/// JSON (json_data), write back via chunk surgery.
/// Read a card's raw JSON via chunk surgery (shared by B1 handlers).
fn read_card_raw(file_path: &std::path::Path) -> Result<String, String> {
    let chunks = png::read_text_chunks(file_path).map_err(|e| e.to_string())?;
    png::extract_card_json(&chunks)
}

/// Write raw card JSON back via chunk surgery (shared by
/// B1 handlers; mirrors write() semantics).
fn write_card_raw(
    file_path: &std::path::Path,
    card_json: &str,
    _data_root: &std::path::Path,
) -> Result<(), String> {
    use base64::Engine;
    let b64 = base64::engine::general_purpose::STANDARD.encode(card_json.as_bytes());
    let new_chunk = png::encode_text_chunk("chara", &b64);
    png::write_card_to_png(file_path, &[new_chunk]).map_err(|e| e.to_string())?;
    // (cache prune removed: node-persist never prunes on write; pruning 6k
    // entries costs ~18s sync IO and hung every create/edit)
    Ok(())
}

/// POST /api/characters/edit-attribute — set a single field on both the top
/// level and `data.<field>`, write back via chunk surgery (no pixel work).
async fn characters_edit_attribute(
    State(state): State<Arc<AppState>>,
    req: Request<Body>,
) -> Response {
    let (parts, body) = req.into_parts();
    let auth = match check_auth(&state, &parts) {
        Ok(a) => a,
        Err(resp) => return resp,
    };
    let body_bytes = match axum::body::to_bytes(body, 256 * 1024 * 1024).await {
        Ok(b) => b,
        Err(_) => return StatusCode::BAD_REQUEST.into_response(),
    };
    let form: serde_json::Value = match serde_json::from_slice(&body_bytes) {
        Ok(v) => v,
        Err(_) => {
            return (StatusCode::BAD_REQUEST, "Error: no response body detected").into_response()
        }
    };

    let ch_name = form.get("ch_name").and_then(|v| v.as_str()).unwrap_or("");
    if ch_name.is_empty() || ch_name == "." {
        return (StatusCode::BAD_REQUEST, "Error: invalid name.").into_response();
    }
    if form.get("field").and_then(|v| v.as_str()) == Some("json_data") {
        return (
            StatusCode::BAD_REQUEST,
            "Error: cannot edit json_data field.",
        )
            .into_response();
    }

    let avatar_url = match form.get("avatar_url").and_then(|v| v.as_str()) {
        Some(s) => s.to_string(),
        None => return StatusCode::BAD_REQUEST.into_response(),
    };
    let deps = user_dirs(&state, &auth.handle);
    let avatar_path = deps.characters_dir.join(&avatar_url);
    let data_root = state.cfg.data_root.clone();

    let result = tokio::task::spawn_blocking(move || {
        edit_attribute_sync(&form, &deps, &avatar_path, &avatar_url, &data_root)
    })
    .await
    .unwrap_or_else(|e| Err(format!("join error: {e}")));

    match result {
        Ok(()) => StatusCode::OK.into_response(),
        Err(e) => {
            tracing::error!("An error occurred, character edit invalidated: {e}");
            StatusCode::INTERNAL_SERVER_ERROR.into_response()
        }
    }
}

fn edit_attribute_sync(
    form: &serde_json::Value,
    deps: &card::CardDeps,
    avatar_path: &std::path::Path,
    avatar_url: &str,
    data_root: &std::path::Path,
) -> Result<(), String> {
    let raw = read_card_raw(avatar_path)?;
    let mut char: serde_json::Value =
        serde_json::from_str(&raw).map_err(|e| format!("card JSON invalid: {e}"))?;

    let field = form
        .get("field")
        .and_then(|v| v.as_str())
        .ok_or("missing field")?;
    // Check existence on either the top level or data.<field> (upstream
    // 400s when neither exists)
    let top = char.get(field).is_some();
    let nested = char.get("data").and_then(|d| d.get(field)).is_some();
    if !top && !nested {
        return Err("Error: invalid field.".to_string());
    }
    let value = form
        .get("value")
        .cloned()
        .unwrap_or(serde_json::Value::Null);
    if let Some(o) = char.as_object_mut() {
        o.insert(field.to_string(), value.clone());
    }
    if let Some(data) = char.get_mut("data") {
        if let Some(o) = data.as_object_mut() {
            o.insert(field.to_string(), value);
        }
    }

    let new_json = serde_json::to_string(&char).map_err(|e| e.to_string())?;
    let target = avatar_url.trim_end_matches(".png");
    let target_path = deps.characters_dir.join(format!("{target}.png"));
    write_card_raw(&target_path, &new_json, data_root)
}

/// POST /api/characters/rename — sanitize the new name, rewrite name fields,
/// chunk-surgery to the new file, migrate the chats folder, unlink the old.
async fn characters_rename(State(state): State<Arc<AppState>>, req: Request<Body>) -> Response {
    let (parts, body) = req.into_parts();
    let auth = match check_auth(&state, &parts) {
        Ok(a) => a,
        Err(resp) => return resp,
    };
    let body_bytes = match axum::body::to_bytes(body, 1024 * 1024).await {
        Ok(b) => b,
        Err(_) => return StatusCode::BAD_REQUEST.into_response(),
    };
    let form: serde_json::Value = match serde_json::from_slice(&body_bytes) {
        Ok(v) => v,
        Err(_) => return StatusCode::BAD_REQUEST.into_response(),
    };
    let old_avatar = match form.get("avatar_url").and_then(|v| v.as_str()) {
        Some(s) => s.to_string(),
        None => return StatusCode::BAD_REQUEST.into_response(),
    };
    let new_name_raw = match form.get("new_name").and_then(|v| v.as_str()) {
        Some(s) => s.to_string(),
        None => return StatusCode::BAD_REQUEST.into_response(),
    };

    let deps = user_dirs(&state, &auth.handle);
    let data_root = state.cfg.data_root.clone();

    let result = tokio::task::spawn_blocking(move || {
        rename_sync(&deps, &data_root, &old_avatar, &new_name_raw)
    })
    .await
    .unwrap_or_else(|e| Err(format!("join error: {e}")));

    match result {
        Ok(new_avatar) => (
            [(axum::http::header::CONTENT_TYPE, "application/json")],
            serde_json::json!({ "avatar": new_avatar }).to_string(),
        )
            .into_response(),
        Err(e) => {
            tracing::error!("rename failed: {e}");
            StatusCode::INTERNAL_SERVER_ERROR.into_response()
        }
    }
}

fn rename_sync(
    deps: &card::CardDeps,
    data_root: &std::path::Path,
    old_avatar: &str,
    new_name_raw: &str,
) -> Result<String, String> {
    let new_name = card::sanitize_filename(new_name_raw);
    let old_internal = old_avatar.trim_end_matches(".png");
    let new_internal = card::get_png_name(&new_name, &deps.characters_dir);
    let new_avatar = format!("{new_internal}.png");

    let old_avatar_path = deps.characters_dir.join(old_avatar);
    let new_avatar_path = deps.characters_dir.join(&new_avatar);

    // Read old card, rewrite name fields
    let raw = read_card_raw(&old_avatar_path)?;
    let parsed: serde_json::Value =
        serde_json::from_str(&raw).map_err(|e| format!("card JSON invalid: {e}"))?;
    let mut old_data = card::get_chara_card_v2(parsed);
    card::set_path_value(
        &mut old_data,
        "data.name",
        serde_json::Value::from(new_name.clone()),
    );
    card::set_path_value(&mut old_data, "name", serde_json::Value::from(new_name));
    let new_json = serde_json::to_string(&old_data).map_err(|e| e.to_string())?;

    // Write to the NEW location via chunk surgery (upstream writes with the
    // new internal name before removing the old file)
    write_card_raw(&new_avatar_path, &new_json, data_root)?;

    // Migrate the chats folder (same-dir rename; cpSync+rmSync upstream, but
    // same-filesystem rename is atomic and equivalent)
    let old_chats = deps.chats_dir.join(old_internal);
    let new_chats = deps.chats_dir.join(&new_internal);
    if old_chats.exists() && !new_chats.exists() {
        std::fs::rename(&old_chats, &new_chats).map_err(|e| {
            format!(
                "chats migration failed ({} -> {}): {e}",
                old_chats.display(),
                new_chats.display()
            )
        })?;
    }

    std::fs::remove_file(&old_avatar_path).map_err(|e| e.to_string())?;
    Ok(new_avatar)
}

/// POST /api/characters/export — png (chunk surgery to strip private fields)
/// and json (normalized + unset private fields, pretty 4-space) branches.
async fn characters_export(State(state): State<Arc<AppState>>, req: Request<Body>) -> Response {
    let (parts, body) = req.into_parts();
    let auth = match check_auth(&state, &parts) {
        Ok(a) => a,
        Err(resp) => return resp,
    };
    let body_bytes = match axum::body::to_bytes(body, 1024 * 1024).await {
        Ok(b) => b,
        Err(_) => return StatusCode::BAD_REQUEST.into_response(),
    };
    let form: serde_json::Value = match serde_json::from_slice(&body_bytes) {
        Ok(v) => v,
        Err(_) => return StatusCode::BAD_REQUEST.into_response(),
    };
    let format_owned = form
        .get("format")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let avatar_url = match form.get("avatar_url").and_then(|v| v.as_str()) {
        Some(s) => s.to_string(),
        None => return StatusCode::BAD_REQUEST.into_response(),
    };
    if format_owned != "png" && format_owned != "json" {
        return StatusCode::BAD_REQUEST.into_response();
    }

    let deps = user_dirs(&state, &auth.handle);
    // Upstream sanitizes the whole avatar_url; mirror by rejecting if altered
    if card::sanitize_filename(&avatar_url) != avatar_url {
        return StatusCode::FORBIDDEN.into_response();
    }
    let filename = deps.characters_dir.join(&avatar_url);
    if !filename.exists() {
        return StatusCode::NOT_FOUND.into_response();
    }

    let basename = filename
        .file_name()
        .map(|f| f.to_string_lossy().to_string())
        .unwrap_or_default();
    let data_root = state.cfg.data_root.clone();
    let result =
        tokio::task::spawn_blocking(move || export_sync(&filename, &format_owned, &data_root))
            .await
            .unwrap_or_else(|e| Err(format!("join error: {e}")));

    match result {
        Ok(Exported::Png(bytes)) => {
            let encoded: String = percent_encoding::utf8_percent_encode(
                &basename,
                percent_encoding::NON_ALPHANUMERIC,
            )
            .to_string();
            Response::builder()
                .status(StatusCode::OK)
                .header("content-type", "image/png")
                .header(
                    "content-disposition",
                    format!("attachment; filename=\"{encoded}\""),
                )
                .body(Body::from(bytes))
                .unwrap()
        }
        Ok(Exported::Json(pretty)) => (
            [(axum::http::header::CONTENT_TYPE, "application/json")],
            pretty,
        )
            .into_response(),
        Err(e) => {
            tracing::error!("Character export failed: {e}");
            StatusCode::INTERNAL_SERVER_ERROR.into_response()
        }
    }
}

enum Exported {
    Png(Vec<u8>),
    Json(String),
}

fn export_sync(
    filename: &std::path::Path,
    format: &str,
    _data_root: &std::path::Path,
) -> Result<Exported, String> {
    match format {
        "png" => {
            let raw = read_card_raw(filename)?;
            // mutateJsonString(json, unsetPrivateFields): parse, strip, dump
            let parsed: serde_json::Value =
                serde_json::from_str(&raw).map_err(|e| format!("card JSON invalid: {e}"))?;
            let mut mutated = parsed;
            card::unset_private_fields(&mut mutated);
            let mutated_json = serde_json::to_string(&mutated).map_err(|e| e.to_string())?;
            // write(rawBuffer, mutatedData): full chunk replacement
            use base64::Engine;
            let b64 = base64::engine::general_purpose::STANDARD.encode(mutated_json.as_bytes());
            let new_chunk = png::encode_text_chunk("chara", &b64);
            // Surgery to a temp copy so the original stays untouched
            // (upstream writes the mutated buffer straight to the response).
            let tmp_out = filename.with_extension("export.tmp");
            png::write_card_to_png_from_to(filename, &tmp_out, &[new_chunk])
                .map_err(|e| e.to_string())?;
            let bytes = std::fs::read(&tmp_out).map_err(|e| e.to_string())?;
            let _ = std::fs::remove_file(&tmp_out);
            Ok(Exported::Png(bytes))
        }
        "json" => {
            let raw = read_card_raw(filename)?;
            let parsed: serde_json::Value =
                serde_json::from_str(&raw).map_err(|_| "invalid".to_string())?;
            let mut obj = card::get_chara_card_v2(parsed);
            card::unset_private_fields(&mut obj);
            // JSON.stringify(obj, null, 4): 4-space pretty
            let pretty = serde_json::to_string_pretty(&obj).map_err(|e| e.to_string())?;
            Ok(Exported::Json(pretty))
        }
        _ => Err("unsupported format".to_string()),
    }
}

// =====================================================================
// B2: /import, /create, /edit (multipart+crop), /edit-avatar
// =====================================================================

/// One parsed multipart part set (multer `.single('avatar')` semantics:
/// fields into body, file into request.file).
pub struct MultipartUpload {
    pub fields: Vec<(String, String)>,
    pub file_bytes: Option<Vec<u8>>,
    pub file_name: Option<String>,
}

/// Decode a multipart/form-data body. Mirrors multer + multerMonkeyPatch
/// (file field named 'avatar'; latin1 filename decode applied by callers if
/// needed for naming).
async fn parse_multipart(content_type: &str, body: Body) -> Result<MultipartUpload, String> {
    let stream = body.into_data_stream();
    let boundary =
        multer::parse_boundary(content_type).map_err(|e| format!("multipart boundary: {e}"))?;
    let mut m = multer::Multipart::new(stream, boundary);
    let mut out = MultipartUpload {
        fields: Vec::new(),
        file_bytes: None,
        file_name: None,
    };
    while let Some(field) = m
        .next_field()
        .await
        .map_err(|e| format!("multipart: {e}"))?
    {
        let name = field.name().unwrap_or("").to_string();
        if name == "avatar" {
            // multer line 167: `if (!filename) return fileStream.resume()` —
            // a file part with EMPTY/absent filename is discarded, leaving
            // request.file undefined (JSON branch upstream). Mirror exactly,
            // otherwise every /edit without a new avatar would replace the
            // character image with the default avatar.
            let has_filename = field
                .file_name()
                .map(|f| !f.is_empty())
                .unwrap_or(false);
            if has_filename {
                out.file_name = field.file_name().map(|s| s.to_string());
                let bytes = field
                    .bytes()
                    .await
                    .map_err(|e| format!("multipart read: {e}"))?;
                out.file_bytes = Some(bytes.to_vec());
            } else {
                // drain and ignore: not an upload
                let _ = field.bytes().await;
            }
        } else {
            let text = field
                .text()
                .await
                .map_err(|e| format!("multipart field: {e}"))?;
            out.fields.push((name, text));
        }
    }
    Ok(out)
}

/// Mirror of charaFormatData (the /create + /edit entry point): json_data
/// base, all fields set, world info embedded. Returns the card JSON string.
fn chara_format_data_full(data: &serde_json::Value, deps: &card::CardDeps) -> String {
    // tryParse(data.json_data) || {}
    let existing: serde_json::Value = data
        .get("json_data")
        .and_then(|v| v.as_str())
        .and_then(|s| serde_json::from_str(s).ok())
        .unwrap_or_else(|| json!({}));
    let get = |k: &str| data.get(k).cloned().unwrap_or(serde_json::Value::Null);
    let input = card::FormatDataInput {
        ch_name: get("ch_name"),
        description: get("description"),
        personality: get("personality"),
        scenario: get("scenario"),
        first_mes: get("first_mes"),
        mes_example: get("mes_example"),
        creator_notes: get("creator_notes"),
        talkativeness: get("talkativeness"),
        fav: get("fav"),
        creator: get("creator"),
        tags: get("tags"),
        depth_prompt_prompt: get("depth_prompt_prompt"),
        depth_prompt_depth: get("depth_prompt_depth"),
        depth_prompt_role: get("depth_prompt_role"),
        system_prompt: get("system_prompt"),
        post_history_instructions: get("post_history_instructions"),
        character_version: get("character_version"),
        alternate_greetings: get("alternate_greetings"),
        world: get("world"),
        extensions: get("extensions"),
    };
    let mut char = card::chara_format_data_from_parts(&input, existing);

    // World info embedding (mirrors readWorldInfoFile in the JS flow)
    if let Some(world) = data
        .get("world")
        .and_then(|w| w.as_str())
        .filter(|s| !s.is_empty())
    {
        let sanitized: String = world
            .chars()
            .filter(|c| {
                !c.is_control()
                    && !matches!(c, '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|')
            })
            .collect();
        let wi_path = deps.worlds_dir.join(format!("{sanitized}.json"));
        if let Ok(text) = std::fs::read_to_string(&wi_path) {
            if let Ok(world_info) = serde_json::from_str::<serde_json::Value>(&text) {
                if let Some(original) = world_info.get("originalData") {
                    card::set_path_value(&mut char, "data.character_book", original.clone());
                } else if let Some(entries) = world_info.get("entries") {
                    let book = card::convert_world_info_to_character_book(world, entries);
                    card::set_path_value(&mut char, "data.character_book", book);
                }
            }
        }
    }
    serde_json::to_string(&char).unwrap_or_default()
}

/// Chunk bytes for a freshly built card: chara (v2) + ccv3 (v3) tEXt, as
/// character-card-parser.js#write does for all writeCharacterData paths.
fn card_chunks(card_json: &str) -> Vec<Vec<u8>> {
    use base64::Engine;
    let b64 = base64::engine::general_purpose::STANDARD.encode(card_json.as_bytes());
    let mut chunks = vec![png::encode_text_chunk("chara", &b64)];
    if let Ok(mut v3) = serde_json::from_str::<serde_json::Value>(card_json) {
        if let Some(o) = v3.as_object_mut() {
            o.insert("spec".into(), json!("chara_card_v3"));
            o.insert("spec_version".into(), json!("3.0"));
        }
        let v3_json = serde_json::to_string(&v3).unwrap_or_default();
        let v3_b64 = base64::engine::general_purpose::STANDARD.encode(v3_json.as_bytes());
        chunks.push(png::encode_text_chunk("ccv3", &v3_b64));
    }
    chunks
}

/// Splice card chunks into image bytes and write characters/<output>.png.
/// `image_bytes` may be a re-encoded PNG (crop path) or raw passthrough
/// (undecodable input); both are full PNG byte buffers.
fn write_card_bytes(
    image_bytes: Vec<u8>,
    card_json: &str,
    output_file: &str,
    deps: &card::CardDeps,
) -> Result<(), String> {
    let chunks = card_chunks(card_json);
    let out_path = deps.characters_dir.join(format!("{output_file}.png"));
    // Materialize the image bytes, then run the in-place chunk splice
    // (write_card_to_png reads src, writes src.png.tmp, renames back to src).
    std::fs::write(&out_path, &image_bytes).map_err(|e| format!("image write: {e}"))?;
    png::write_card_to_png(&out_path, &chunks).map_err(|e| e.to_string())?;
    Ok(())
    // (no prune: see write_card_raw comment)
}

/// writeCharacterData(DEFAULT_AVATAR_PATH, ...) — splice into a copy of
/// public/img/ai4.png.
fn write_default_avatar_card(
    data_root: &std::path::Path,
    card_json: &str,
    output_file: &str,
    deps: &card::CardDeps,
) -> Result<(), String> {
    // Node: DEFAULT_AVATAR_PATH = './public/img/ai4.png' relative to cwd
    // (SillyTavern root). Try cwd-relative, data_root-relative, and walking
    // up from data_root (covers test cwd = src-rs and nonstandard layouts).
    let mut candidates = vec![
        std::path::PathBuf::from("./public/img/ai4.png"),
        data_root.join("../public/img/ai4.png"),
    ];
    // Walk up from data_root AND from cwd (tests run in src-rs; the gateway
    // runs in the ST root, so cwd-relative hits immediately in production).
    for root in [Some(data_root.to_path_buf()), std::env::current_dir().ok()] {
        let mut cur = root;
        while let Some(dir) = cur {
            candidates.push(dir.join("public/img/ai4.png"));
            cur = dir.parent().map(|p| p.to_path_buf());
        }
    }
    let mut raw = None;
    for c in &candidates {
        if let Ok(bytes) = std::fs::read(c) {
            raw = Some(bytes);
            break;
        }
    }
    let raw = raw.ok_or_else(|| "default avatar missing: ./public/img/ai4.png".to_string())?;
    // upstream: default avatar is read via getInputImage's buffer branch →
    // parseImageBuffer → decode+crop(None)=re-encode. Mirror with decode+encode.
    let prepared = avatar::prepare_upload_bytes(&raw, None, &raw)?;
    write_card_bytes(prepared, card_json, output_file, deps)
}

/// Crop query parameter parser (`tryParse(request.query.crop)`).
fn parse_crop_query(query: Option<&str>) -> Option<avatar::Crop> {
    let v = query?;
    let decoded = percent_decode(v);
    let parsed: serde_json::Value = serde_json::from_str(&decoded).ok()?;
    avatar::Crop::from_value(&parsed)
}

/// Extract a `crop` query value from a raw query string.
fn crop_from_query(query: Option<&str>) -> Option<avatar::Crop> {
    query
        .and_then(|q| q.split('&').find_map(|kv| kv.strip_prefix("crop=")))
        .and_then(|v| parse_crop_query(Some(v)))
}

/// POST /api/characters/create — multipart or JSON, optional crop query,
/// optional avatar upload; falls back to the default avatar image.
async fn characters_create(State(state): State<Arc<AppState>>, req: Request<Body>) -> Response {
    let (parts, body) = req.into_parts();
    let auth = match check_auth(&state, &parts) {
        Ok(a) => a,
        Err(resp) => return resp,
    };
    let ct = parts
        .headers
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    let crop = crop_from_query(parts.uri.query());

    let body_bytes = match axum::body::to_bytes(body, 512 * 1024 * 1024).await {
        Ok(b) => b,
        Err(_) => return StatusCode::BAD_REQUEST.into_response(),
    };

    let (form, upload): (serde_json::Value, Option<Vec<u8>>) = if ct.starts_with("multipart/") {
        let mp = match parse_multipart(&ct, axum::body::Body::from(body_bytes)).await {
            Ok(m) => m,
            Err(_) => return StatusCode::BAD_REQUEST.into_response(),
        };
        let mut obj = serde_json::Map::new();
        for (k, v) in &mp.fields {
            obj.insert(k.clone(), json!(v));
        }
        (serde_json::Value::Object(obj), mp.file_bytes)
    } else {
        match serde_json::from_slice::<serde_json::Value>(&body_bytes) {
            Ok(v) => (v, None),
            Err(_) => return StatusCode::BAD_REQUEST.into_response(),
        }
    };

    // getFileNameValidationFunction('file_name'): forbiddenRegExp
    if let Some(fn_name) = form.get("file_name").and_then(|v| v.as_str()) {
        if fn_name.contains('/') || fn_name.contains('\\') || fn_name.contains('\0') {
            return StatusCode::BAD_REQUEST.into_response();
        }
    }

    let ch_name = match form.get("ch_name").and_then(|v| v.as_str()) {
        Some(s) => card::sanitize_filename(s),
        None => return StatusCode::BAD_REQUEST.into_response(),
    };

    let deps = user_dirs(&state, &auth.handle);
    let data_root = state.cfg.data_root.clone();

    let result = tokio::task::spawn_blocking(move || {
        create_sync(&form, &ch_name, upload.as_deref(), crop, &deps, &data_root)
    })
    .await
    .unwrap_or_else(|e| Err(format!("join error: {e}")));

    match result {
        Ok(avatar_name) => (
            [(axum::http::header::CONTENT_TYPE, "application/json")],
            avatar_name,
        )
            .into_response(),
        Err(e) => {
            tracing::error!("create failed: {e}");
            StatusCode::INTERNAL_SERVER_ERROR.into_response()
        }
    }
}

fn create_sync(
    form: &serde_json::Value,
    ch_name: &str,
    upload: Option<&[u8]>,
    crop: Option<avatar::Crop>,
    deps: &card::CardDeps,
    data_root: &std::path::Path,
) -> Result<String, String> {
    let card_json = chara_format_data_full(form, deps);
    let internal_name = match form.get("file_name").and_then(|v| v.as_str()) {
        Some(f) if !f.is_empty() => f.to_string(),
        _ => card::get_png_name(ch_name, &deps.characters_dir),
    };
    let avatar_name = format!("{internal_name}.png");

    // upstream mkdirSync(chatsPath) unconditionally
    let chats_path = deps.chats_dir.join(&internal_name);
    if !chats_path.exists() {
        std::fs::create_dir_all(&chats_path).map_err(|e| e.to_string())?;
    }

    match upload {
        None => {
            write_default_avatar_card(data_root, &card_json, &internal_name, deps)?;
        }
        Some(bytes) => {
            // upstream: undecodable upload → DEFAULT_AVATAR_PATH fallback
            let default_raw = read_default_avatar_bytes(data_root)?;
            let prepared = avatar::prepare_upload_bytes(bytes, crop.as_ref(), &default_raw)?;
            write_card_bytes(prepared, &card_json, &internal_name, deps)?;
        }
    }
    // NOTE: no cache prune here. node-persist never prunes on write; the
    // mtime-keyed lookup naturally ignores stale entries, and Node writes
    // fresh entries after its own writes. Pruning 6k entries costs ~18s of
    // sync file IO and made every create/edit hang.
    Ok(avatar_name)
}

/// Read DEFAULT_AVATAR_PATH (public/img/ai4.png) with the same path search
/// as write_default_avatar_card.
fn read_default_avatar_bytes(data_root: &std::path::Path) -> Result<Vec<u8>, String> {
    let mut candidates = vec![
        std::path::PathBuf::from("./public/img/ai4.png"),
        data_root.join("../public/img/ai4.png"),
    ];
    for root in [Some(data_root.to_path_buf()), std::env::current_dir().ok()] {
        let mut cur = root;
        while let Some(dir) = cur {
            candidates.push(dir.join("public/img/ai4.png"));
            cur = dir.parent().map(|p| p.to_path_buf());
        }
    }
    for c in &candidates {
        if let Ok(bytes) = std::fs::read(c) {
            return Ok(bytes);
        }
    }
    Err("default avatar missing: ./public/img/ai4.png".to_string())
}

/// POST /api/characters/edit — full handler: JSON (chunk surgery) and
/// multipart+crop (decode/crop/encode) branches, replacing the forward-on-
/// upload placeholder from B1.
async fn characters_edit_full(State(state): State<Arc<AppState>>, req: Request<Body>) -> Response {
    let (parts, body) = req.into_parts();
    let auth = match check_auth(&state, &parts) {
        Ok(a) => a,
        Err(resp) => return resp,
    };
    let ct = parts
        .headers
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    let crop = crop_from_query(parts.uri.query());

    let body_bytes = match axum::body::to_bytes(body, 512 * 1024 * 1024).await {
        Ok(b) => b,
        Err(_) => {
            return (StatusCode::BAD_REQUEST, "Error: no response body detected").into_response()
        }
    };

    let (form, upload): (serde_json::Value, Option<Vec<u8>>) = if ct.starts_with("multipart/") {
        let mp = match parse_multipart(&ct, axum::body::Body::from(body_bytes)).await {
            Ok(m) => m,
            Err(_) => return StatusCode::BAD_REQUEST.into_response(),
        };
        let mut obj = serde_json::Map::new();
        for (k, v) in &mp.fields {
            obj.insert(k.clone(), json!(v));
        }
        (serde_json::Value::Object(obj), mp.file_bytes)
    } else {
        match serde_json::from_slice::<serde_json::Value>(&body_bytes) {
            Ok(v) => (v, None),
            Err(_) => {
                return (StatusCode::BAD_REQUEST, "Error: no response body detected")
                    .into_response()
            }
        }
    };

    let ch_name = form.get("ch_name").and_then(|v| v.as_str()).unwrap_or("");
    if ch_name.is_empty() || ch_name == "." {
        return (StatusCode::BAD_REQUEST, "Error: invalid name.").into_response();
    }
    let avatar_url = match form.get("avatar_url").and_then(|v| v.as_str()) {
        Some(s) => s.to_string(),
        None => return StatusCode::BAD_REQUEST.into_response(),
    };

    let deps = user_dirs(&state, &auth.handle);
    let data_root = state.cfg.data_root.clone();

    let result = tokio::task::spawn_blocking(move || {
        edit_full_sync(
            &form,
            &avatar_url,
            upload.as_deref(),
            crop,
            &deps,
            &data_root,
        )
    })
    .await
    .unwrap_or_else(|e| Err(format!("join error: {e}")));

    match result {
        Ok(()) => StatusCode::OK.into_response(),
        Err(e) => {
            tracing::error!("An error occurred, character edit invalidated: {e}");
            StatusCode::INTERNAL_SERVER_ERROR.into_response()
        }
    }
}

fn edit_full_sync(
    form: &serde_json::Value,
    avatar_url: &str,
    upload: Option<&[u8]>,
    crop: Option<avatar::Crop>,
    deps: &card::CardDeps,
    data_root: &std::path::Path,
) -> Result<(), String> {
    let card_json = chara_format_data_full(form, deps);
    // char.chat / char.create_date overrides (upstream /edit)
    let mut char_v: serde_json::Value =
        serde_json::from_str(&card_json).map_err(|e| e.to_string())?;
    if let Some(chat) = form.get("chat") {
        card::set_path_value(&mut char_v, "chat", chat.clone());
    }
    if let Some(cd) = form.get("create_date") {
        card::set_path_value(&mut char_v, "create_date", cd.clone());
    }
    let card_json = serde_json::to_string(&char_v).map_err(|e| e.to_string())?;
    let target_file = avatar_url.replace(".png", "");

    match upload {
        None => {
            let avatar_path = deps.characters_dir.join(avatar_url);
            write_card_raw(&avatar_path, &card_json, data_root)
        }
        Some(bytes) => {
            invalidate_avatar_thumbnail(deps, avatar_url);
            let default_raw = read_default_avatar_bytes(data_root)?;
            let prepared = avatar::prepare_upload_bytes(bytes, crop.as_ref(), &default_raw)?;
            write_card_bytes(prepared, &card_json, &target_file, deps)?;
            // (cache prune removed: node-persist never prunes on write; pruning 6k
            // entries costs ~18s sync IO and hung every create/edit)
            Ok(())
        }
    }
}

/// POST /api/characters/edit-avatar — replace the image, keep card data.
async fn characters_edit_avatar(
    State(state): State<Arc<AppState>>,
    req: Request<Body>,
) -> Response {
    let (parts, body) = req.into_parts();
    let auth = match check_auth(&state, &parts) {
        Ok(a) => a,
        Err(resp) => return resp,
    };
    let ct = parts
        .headers
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    if !ct.starts_with("multipart/") {
        return (StatusCode::BAD_REQUEST, "Error: no file uploaded").into_response();
    }
    let crop = crop_from_query(parts.uri.query());

    let body_bytes = match axum::body::to_bytes(body, 512 * 1024 * 1024).await {
        Ok(b) => b,
        Err(_) => return StatusCode::BAD_REQUEST.into_response(),
    };
    let mp = match parse_multipart(&ct, axum::body::Body::from(body_bytes)).await {
        Ok(m) => m,
        Err(e) => {
            tracing::warn!("edit-avatar: multipart parse failed: {e}");
            return StatusCode::BAD_REQUEST.into_response();
        }
    };
    let fields = mp.fields.clone();
    let upload = match mp.file_bytes {
        Some(b) => b,
        None => return (StatusCode::BAD_REQUEST, "Error: no file uploaded").into_response(),
    };
    let avatar_url = match fields
        .iter()
        .find(|(k, _)| k == "avatar_url")
        .map(|(_, v)| v.as_str())
    {
        Some(s) => s.to_string(),
        None => {
            return (
                StatusCode::BAD_REQUEST,
                "Error: no avatar_url in request body",
            )
                .into_response()
        }
    };

    let deps = user_dirs(&state, &auth.handle);
    let data_root = state.cfg.data_root.clone();

    let result = tokio::task::spawn_blocking(move || {
        edit_avatar_sync(&upload, &avatar_url, crop, &deps, &data_root)
    })
    .await
    .unwrap_or_else(|e| Err(format!("join error: {e}")));

    match result {
        Ok(()) => StatusCode::OK.into_response(),
        Err(e) => {
            tracing::error!("An error occurred while editing avatar: {e}");
            StatusCode::INTERNAL_SERVER_ERROR.into_response()
        }
    }
}

fn edit_avatar_sync(
    upload: &[u8],
    avatar_url: &str,
    crop: Option<avatar::Crop>,
    deps: &card::CardDeps,
    data_root: &std::path::Path,
) -> Result<(), String> {
    let character_path = deps.characters_dir.join(avatar_url);
    if !character_path.exists() {
        return Err("character file does not exist".into());
    }
    // readCharacterData: existing card JSON preserved verbatim
    let data = read_card_raw(&character_path)?;

    invalidate_avatar_thumbnail(deps, avatar_url);
    let default_raw = read_default_avatar_bytes(data_root)?;
    let prepared = avatar::prepare_upload_bytes(upload, crop.as_ref(), &default_raw)?;
    let file_name = avatar_url.replace(".png", "");
    write_card_bytes(prepared, &data, &file_name, deps)?;
    // (cache prune removed: node-persist never prunes on write; pruning 6k
    // entries costs ~18s sync IO and hung every create/edit)
    Ok(())
}

/// invalidateThumbnail(directories, 'avatar', file): unlink
/// <user>/thumbnails/avatar/<file>.
fn invalidate_avatar_thumbnail(deps: &card::CardDeps, avatar_url: &str) {
    let safe = card::sanitize_filename(avatar_url);
    let thumb = deps
        .characters_dir
        .parent()
        .map(|p| p.join("thumbnails/avatar").join(safe));
    if let Some(t) = thumb {
        if t.is_file() {
            let _ = std::fs::remove_file(t);
        }
    }
}

/// POST /api/characters/import — png/json/yaml formats (charx/byaf remain
/// forwarded: zip/binary parsers, rare).
async fn characters_import(State(state): State<Arc<AppState>>, req: Request<Body>) -> Response {
    let (parts, body) = req.into_parts();
    let auth = match check_auth(&state, &parts) {
        Ok(a) => a,
        Err(resp) => return resp,
    };
    let ct = parts
        .headers
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    if !ct.starts_with("multipart/") {
        return StatusCode::BAD_REQUEST.into_response();
    }
    let body_bytes = match axum::body::to_bytes(body, 512 * 1024 * 1024).await {
        Ok(b) => b,
        Err(_) => return StatusCode::BAD_REQUEST.into_response(),
    };
    let mp = match parse_multipart(&ct, axum::body::Body::from(body_bytes)).await {
        Ok(m) => m,
        Err(e) => {
            tracing::warn!("import: multipart parse failed: {e}");
            return StatusCode::BAD_REQUEST.into_response();
        }
    };
    let fields = mp.fields.clone();
    let upload = match mp.file_bytes {
        Some(b) => b,
        None => return StatusCode::BAD_REQUEST.into_response(),
    };
    let field = |name: &str| {
        fields
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.as_str())
    };
    let format = field("file_type").unwrap_or("").to_string();
    let preserved = field("preserved_name")
        .filter(|s| !s.is_empty())
        .and_then(|s| {
            std::path::Path::new(s)
                .file_stem()
                .map(|f| f.to_string_lossy().to_string())
        })
        .filter(|s| !s.is_empty());

    if !matches!(format.as_str(), "png" | "json" | "yaml" | "yml") {
        // unsupported format → upstream throws → catch sends { error: true };
        // charx/byaf stay forwarded at the router level (route_mode).
        return (
            [(axum::http::header::CONTENT_TYPE, "application/json")],
            json!({ "error": true }).to_string(),
        )
            .into_response();
    }

    let deps = user_dirs(&state, &auth.handle);
    let deps2 = deps.clone();
    let data_root = state.cfg.data_root.clone();

    let preserved_for_invalidate = preserved.clone();
    let result = tokio::task::spawn_blocking(move || {
        import_sync(&upload, &format, preserved.as_deref(), &deps2, &data_root)
    })
    .await
    .unwrap_or_else(|e| Err(format!("join error: {e}")));

    match result {
        Ok(file_name) if !file_name.is_empty() => {
            if let Some(p) = &preserved_for_invalidate {
                invalidate_avatar_thumbnail(&deps, &format!("{p}.png"));
            }
            (
                [(axum::http::header::CONTENT_TYPE, "application/json")],
                json!({ "file_name": file_name }).to_string(),
            )
                .into_response()
        }
        Ok(_) => {
            tracing::warn!("Failed to import character");
            StatusCode::BAD_REQUEST.into_response()
        }
        Err(e) => {
            tracing::error!("import failed: {e}");
            (
                [(axum::http::header::CONTENT_TYPE, "application/json")],
                json!({ "error": true }).to_string(),
            )
                .into_response()
        }
    }
}

fn import_sync(
    upload: &[u8],
    format: &str,
    preserved: Option<&str>,
    deps: &card::CardDeps,
    data_root: &std::path::Path,
) -> Result<String, String> {
    let now_iso = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true);
    match format {
        "png" => {
            let raw = read_card_bytes(upload)?;
            let mut json_data: serde_json::Value =
                serde_json::from_str(&raw).map_err(|e| format!("card JSON invalid: {e}"))?;

            if let Some(name) = json_data
                .get("data")
                .and_then(|d| d.get("name"))
                .and_then(|n| n.as_str())
            {
                let s = card::sanitize_filename(name);
                card::set_path_value(&mut json_data, "data.name", json!(s));
            }
            let top_name = json_data
                .get("data")
                .and_then(|d| d.get("name"))
                .and_then(|n| n.as_str())
                .or_else(|| json_data.get("name").and_then(|n| n.as_str()))
                .unwrap_or("");
            let name = card::sanitize_filename(top_name);
            card::set_path_value(&mut json_data, "name", json!(name));
            let png_name = preserved
                .map(|s| s.to_string())
                .unwrap_or_else(|| card::get_png_name(&name, &deps.characters_dir));

            if json_data.get("spec").map(|s| !s.is_null()).unwrap_or(false) {
                // V2/V3 card
                import_risu_sprites(json_data.get("data"), deps);
                card::unset_private_fields(&mut json_data);
                card::read_from_v2(&mut json_data);
                card::set_path_value(&mut json_data, "create_date", json!(now_iso));
                let char_json = serde_json::to_string(&json_data).map_err(|e| e.to_string())?;
                let prepared = avatar::prepare_file_bytes(upload, None)?;
                write_card_bytes(prepared, &char_json, &png_name, deps)?;
                // (cache prune removed: node-persist never prunes on write; pruning 6k
                // entries costs ~18s sync IO and hung every create/edit)
                Ok(png_name)
            } else if json_data.get("name").map(|s| !s.is_null()).unwrap_or(false) {
                // V1 card
                if let Some(notes) = json_data.get("creator_notes").and_then(|n| n.as_str()) {
                    let cleaned = notes.replace("Creator's notes go here.", "");
                    card::set_path_value(&mut json_data, "creator_notes", json!(cleaned));
                }
                let char = card::convert_to_v2(&v1_to_card(&json_data, &now_iso));
                let char_json = serde_json::to_string(&char).map_err(|e| e.to_string())?;
                let prepared = avatar::prepare_file_bytes(upload, None)?;
                write_card_bytes(prepared, &char_json, &png_name, deps)?;
                // (cache prune removed: node-persist never prunes on write; pruning 6k
                // entries costs ~18s sync IO and hung every create/edit)
                Ok(png_name)
            } else {
                Ok(String::new())
            }
        }
        "json" => {
            let text = String::from_utf8_lossy(upload).to_string();
            let mut json_data: serde_json::Value =
                serde_json::from_str(&text).map_err(|e| format!("json invalid: {e}"))?;
            if json_data.get("spec").map(|s| !s.is_null()).unwrap_or(false) {
                import_risu_sprites(json_data.get("data"), deps);
                card::unset_private_fields(&mut json_data);
                if let Some(name) = json_data
                    .get("data")
                    .and_then(|d| d.get("name"))
                    .and_then(|n| n.as_str())
                {
                    let s = card::sanitize_filename(name);
                    card::set_path_value(&mut json_data, "data.name", json!(s));
                }
                let top_name = json_data
                    .get("data")
                    .and_then(|d| d.get("name"))
                    .and_then(|n| n.as_str())
                    .or_else(|| json_data.get("name").and_then(|n| n.as_str()))
                    .unwrap_or("");
                let name = card::sanitize_filename(top_name);
                card::set_path_value(&mut json_data, "name", json!(name));
                card::read_from_v2(&mut json_data);
                card::set_path_value(&mut json_data, "create_date", json!(now_iso));
                let png_name = preserved
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| card::get_png_name(&name, &deps.characters_dir));
                let char_json = serde_json::to_string(&json_data).map_err(|e| e.to_string())?;
                write_default_avatar_card(data_root, &char_json, &png_name, deps)?;
                // (cache prune removed: node-persist never prunes on write; pruning 6k
                // entries costs ~18s sync IO and hung every create/edit)
                Ok(png_name)
            } else if json_data.get("name").map(|s| !s.is_null()).unwrap_or(false) {
                let name = card::sanitize_filename(
                    json_data.get("name").and_then(|n| n.as_str()).unwrap_or(""),
                );
                if let Some(notes) = json_data.get("creator_notes").and_then(|n| n.as_str()) {
                    let cleaned = notes.replace("Creator's notes go here.", "");
                    card::set_path_value(&mut json_data, "creator_notes", json!(cleaned));
                }
                card::set_path_value(&mut json_data, "name", json!(name));
                let png_name = preserved
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| card::get_png_name(&name, &deps.characters_dir));
                let char = card::convert_to_v2(&v1_to_card(&json_data, &now_iso));
                let char_json = serde_json::to_string(&char).map_err(|e| e.to_string())?;
                write_default_avatar_card(data_root, &char_json, &png_name, deps)?;
                // (cache prune removed: node-persist never prunes on write; pruning 6k
                // entries costs ~18s sync IO and hung every create/edit)
                Ok(png_name)
            } else if json_data
                .get("char_name")
                .map(|s| !s.is_null())
                .unwrap_or(false)
            {
                // Pygmalion gradio json
                let name = card::sanitize_filename(
                    json_data
                        .get("char_name")
                        .and_then(|n| n.as_str())
                        .unwrap_or(""),
                );
                if let Some(notes) = json_data.get("creator_notes").and_then(|n| n.as_str()) {
                    let cleaned = notes.replace("Creator's notes go here.", "");
                    card::set_path_value(&mut json_data, "creator_notes", json!(cleaned));
                }
                let png_name = preserved
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| card::get_png_name(&name, &deps.characters_dir));
                let mut char = serde_json::json!({});
                card::set_path_value(
                    &mut char,
                    "name",
                    json_data.get("char_name").cloned().unwrap_or(json!("")),
                );
                card::set_path_value(
                    &mut char,
                    "description",
                    json!(json_data
                        .get("char_persona")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")),
                );
                card::set_path_value(
                    &mut char,
                    "creatorcomment",
                    json!(json_data
                        .get("creatorcomment")
                        .or_else(|| json_data.get("creator_notes"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("")),
                );
                card::set_path_value(&mut char, "personality", json!(""));
                card::set_path_value(
                    &mut char,
                    "first_mes",
                    json!(json_data
                        .get("char_greeting")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")),
                );
                card::set_path_value(&mut char, "avatar", json!("none"));
                card::set_path_value(
                    &mut char,
                    "chat",
                    json!(format!(
                        "{} - {}",
                        json_data.get("name").and_then(|n| n.as_str()).unwrap_or(""),
                        card::humanized_datetime()
                    )),
                );
                card::set_path_value(
                    &mut char,
                    "mes_example",
                    json!(json_data
                        .get("example_dialogue")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")),
                );
                card::set_path_value(
                    &mut char,
                    "scenario",
                    json!(json_data
                        .get("world_scenario")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")),
                );
                card::set_path_value(&mut char, "create_date", json!(now_iso));
                card::set_path_value(&mut char, "talkativeness", json!(0.5));
                card::set_path_value(&mut char, "creator", json!(""));
                card::set_path_value(&mut char, "tags", json!(""));
                let char = card::convert_to_v2(&char);
                let char_json = serde_json::to_string(&char).map_err(|e| e.to_string())?;
                write_default_avatar_card(data_root, &char_json, &png_name, deps)?;
                // (cache prune removed: node-persist never prunes on write; pruning 6k
                // entries costs ~18s sync IO and hung every create/edit)
                Ok(png_name)
            } else {
                Ok(String::new())
            }
        }
        "yaml" | "yml" => {
            let text = String::from_utf8_lossy(upload).to_string();
            let yaml_data: serde_json::Value =
                serde_yaml::from_str(&text).map_err(|e| format!("yaml invalid: {e}"))?;
            let name = card::sanitize_filename(
                yaml_data.get("name").and_then(|n| n.as_str()).unwrap_or(""),
            );
            let png_name = preserved
                .map(|s| s.to_string())
                .unwrap_or_else(|| card::get_png_name(&name, &deps.characters_dir));
            let mut char = serde_json::json!({});
            card::set_path_value(&mut char, "name", json!(name));
            card::set_path_value(
                &mut char,
                "description",
                json!(yaml_data
                    .get("context")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")),
            );
            card::set_path_value(
                &mut char,
                "first_mes",
                json!(yaml_data
                    .get("greeting")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")),
            );
            card::set_path_value(&mut char, "create_date", json!(now_iso));
            card::set_path_value(
                &mut char,
                "chat",
                json!(format!(
                    "{} - {}",
                    yaml_data.get("name").and_then(|n| n.as_str()).unwrap_or(""),
                    card::humanized_datetime()
                )),
            );
            card::set_path_value(&mut char, "personality", json!(""));
            card::set_path_value(&mut char, "creatorcomment", json!(""));
            card::set_path_value(&mut char, "avatar", json!("none"));
            card::set_path_value(&mut char, "mes_example", json!(""));
            card::set_path_value(&mut char, "scenario", json!(""));
            card::set_path_value(&mut char, "talkativeness", json!(0.5));
            card::set_path_value(&mut char, "creator", json!(""));
            card::set_path_value(&mut char, "tags", json!(""));
            let char = card::convert_to_v2(&char);
            let char_json = serde_json::to_string(&char).map_err(|e| e.to_string())?;
            write_default_avatar_card(data_root, &char_json, &png_name, deps)?;
            // (cache prune removed: node-persist never prunes on write; pruning 6k
            // entries costs ~18s sync IO and hung every create/edit)
            Ok(png_name)
        }
        _ => Err(format!("Unsupported format: {format}")),
    }
}

/// Read card JSON from raw PNG bytes (import path; upload is in memory).
fn read_card_bytes(png_bytes: &[u8]) -> Result<String, String> {
    // Reuse the streaming chunk reader via a short-lived temp file; uploads
    // are MBs. Avoids duplicating the chunk parser for in-memory buffers.
    let tmp = std::env::temp_dir().join(format!("st-import-{}.png", std::process::id()));
    std::fs::write(&tmp, png_bytes).map_err(|e| e.to_string())?;
    let result = (|| {
        let chunks = png::read_text_chunks(&tmp).map_err(|e| e.to_string())?;
        png::extract_card_json(&chunks)
    })();
    let _ = std::fs::remove_file(&tmp);
    result
}

/// V1 flat object literal (upstream builds this then convertToV2).
fn v1_to_card(json_data: &serde_json::Value, now_iso: &str) -> serde_json::Value {
    let s = |k: &str| json_data.get(k).and_then(|v| v.as_str()).unwrap_or("");
    let name = s("name");
    serde_json::json!({
        "name": name,
        "description": json_data.get("description").cloned().unwrap_or(json!("")),
        "creatorcomment": json_data.get("creatorcomment")
            .or_else(|| json_data.get("creator_notes"))
            .and_then(|v| v.as_str()).unwrap_or(""),
        "personality": json_data.get("personality").cloned().unwrap_or(json!("")),
        "first_mes": json_data.get("first_mes").cloned().unwrap_or(json!("")),
        "avatar": "none",
        "chat": format!("{name} - {}", card::humanized_datetime()),
        "mes_example": json_data.get("mes_example").cloned().unwrap_or(json!("")),
        "scenario": json_data.get("scenario").cloned().unwrap_or(json!("")),
        "create_date": now_iso,
        "talkativeness": json_data.get("talkativeness").cloned().unwrap_or(json!(0.5)),
        "creator": json_data.get("creator").cloned().unwrap_or(json!("")),
        "tags": json_data.get("tags").cloned().unwrap_or(json!("")),
    })
}

/// importRisuSprites mirror: data.extensions.risuai additionalAssets +
/// emotions [label, base64] pairs → characters/<name>/<label>.png.
fn import_risu_sprites(data: Option<&serde_json::Value>, deps: &card::CardDeps) {
    let Some(data) = data else { return };
    let Some(name) = data.get("name").and_then(|n| n.as_str()) else {
        return;
    };
    let Some(risu) = data.get("extensions").and_then(|e| e.get("risuai")) else {
        return;
    };
    let mut images: Vec<(String, String)> = Vec::new();
    for key in ["additionalAssets", "emotions"] {
        if let Some(arr) = risu.get(key).and_then(|a| a.as_array()) {
            for item in arr {
                if let Some(pair) = item.as_array() {
                    if pair.len() == 2 {
                        let label = pair[0].as_str().unwrap_or("").to_string();
                        let b64 = pair[1].as_str().unwrap_or("").to_string();
                        if !label.is_empty() {
                            images.push((label, b64));
                        }
                    }
                }
            }
        }
    }
    if images.is_empty() {
        return;
    }
    let safe_name = card::sanitize_filename(name);
    if safe_name.is_empty() {
        return;
    }
    let sprites_dir = deps.characters_dir.join(&safe_name);
    if sprites_dir.exists() && !sprites_dir.is_dir() {
        return;
    }
    let _ = std::fs::create_dir_all(&sprites_dir);
    let existing: Vec<String> = std::fs::read_dir(&sprites_dir)
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .map(|e| e.file_name().to_string_lossy().to_string())
                .collect()
        })
        .unwrap_or_default();
    use base64::Engine;
    for (label, b64) in images {
        if existing.iter().any(|f| {
            std::path::Path::new(f)
                .file_stem()
                .map(|s| s.to_string_lossy() == label)
                .unwrap_or(false)
        }) {
            continue; // same-label sprite exists: skip
        }
        let filename = format!("{}.png", card::sanitize_filename(&label));
        if let Ok(bytes) = base64::engine::general_purpose::STANDARD.decode(&b64) {
            let _ = std::fs::write(sprites_dir.join(filename), bytes);
        }
    }
}

/// Minimal percent-decode for query values (crop JSON).
fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'%' if i + 2 < bytes.len() => {
                let hex = std::str::from_utf8(&bytes[i + 1..i + 3]).unwrap_or("");
                if let Ok(b) = u8::from_str_radix(hex, 16) {
                    out.push(b);
                    i += 3;
                } else {
                    out.push(bytes[i]);
                    i += 1;
                }
            }
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            b => {
                out.push(b);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).to_string()
}

#[cfg(test)]
mod cache_tests {
    use super::*;

    /// Cache hit must produce a response identical to a fresh PNG parse
    /// (same fields, same verbatim json_data) — and skip chunk scanning.
    #[tokio::test]
    async fn cache_hit_matches_full_parse() {
        let dir = tempfile::tempdir().unwrap();
        let chars = dir
            .path()
            .join("data")
            .join("default-user")
            .join("characters");
        std::fs::create_dir_all(&chars).unwrap();

        // Valid card PNG
        let card = serde_json::json!({
            "spec": "chara_card_v2", "spec_version": "2.0",
            "name": "Cached", "data": { "name": "Cached", "description": "d" }
        })
        .to_string();
        use base64::Engine;
        let b64 = base64::engine::general_purpose::STANDARD.encode(card.as_bytes());
        let mut data = b"chara\0".to_vec();
        data.extend_from_slice(b64.as_bytes());
        let mut png_bytes = Vec::new();
        png_bytes.extend_from_slice(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]);
        png_bytes.extend_from_slice(&png::build_chunk(
            b"IHDR",
            &[0, 0, 0, 1, 0, 0, 0, 1, 8, 6, 0, 0, 0],
        ));
        png_bytes.extend_from_slice(&png::build_chunk(b"tEXt", &data));
        png_bytes.extend_from_slice(&png::build_chunk(b"IEND", &[]));
        std::fs::write(chars.join("c.png"), &png_bytes).unwrap();

        let data_root = dir.path().join("data");
        let deps = card::CardDeps {
            characters_dir: chars.clone(),
            chats_dir: dir.path().join("data/default-user/chats"),
            worlds_dir: dir.path().join("data/default-user/worlds"),
        };
        let cache = disk_cache::DiskCache::new(&data_root);

        // First pass: miss → parse → write back
        let cold = process_character_cached("c.png", &deps, &cache, &data_root, "default-user");
        assert_eq!(cold["name"], "Cached");
        assert!(
            cache.dir().read_dir().unwrap().count() >= 1,
            "entry written"
        );

        // Second pass: hit → identical result without PNG parsing.
        let warm = process_character_cached("c.png", &deps, &cache, &data_root, "default-user");
        // Exception: `chat` carries a fresh humanizedDateTime() per cold parse
        // (upstream behavior too — readFromV2 backfills missing chat with
        // now()). The cache hit replays the parsed card including the chat
        // default generated at cache-fill time; compare everything else.
        let normalize = |v: &serde_json::Value| {
            let mut v = v.clone();
            if let Some(o) = v.as_object_mut() {
                o.remove("chat");
            }
            v
        };
        assert_eq!(normalize(&warm), normalize(&cold));
        // json_data verbatim round-trip
        assert_eq!(
            warm["json_data"].as_str().unwrap(),
            cold["json_data"].as_str().unwrap()
        );
    }

    /// Corrupted cache entry must fall back to a full parse, not fail.
    #[tokio::test]
    async fn corrupt_cache_entry_falls_back_to_parse() {
        let dir = tempfile::tempdir().unwrap();
        let chars = dir
            .path()
            .join("data")
            .join("default-user")
            .join("characters");
        std::fs::create_dir_all(&chars).unwrap();

        let card = serde_json::json!({
            "spec": "chara_card_v2", "spec_version": "2.0",
            "name": "Fallback", "data": { "name": "Fallback", "description": "d" }
        })
        .to_string();
        use base64::Engine;
        let b64 = base64::engine::general_purpose::STANDARD.encode(card.as_bytes());
        let mut data = b"chara\0".to_vec();
        data.extend_from_slice(b64.as_bytes());
        let mut png_bytes = Vec::new();
        png_bytes.extend_from_slice(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]);
        png_bytes.extend_from_slice(&png::build_chunk(
            b"IHDR",
            &[0, 0, 0, 1, 0, 0, 0, 1, 8, 6, 0, 0, 0],
        ));
        png_bytes.extend_from_slice(&png::build_chunk(b"tEXt", &data));
        png_bytes.extend_from_slice(&png::build_chunk(b"IEND", &[]));
        std::fs::write(chars.join("f.png"), &png_bytes).unwrap();

        let data_root = dir.path().join("data");
        let deps = card::CardDeps {
            characters_dir: chars.clone(),
            chats_dir: dir.path().join("data/default-user/chats"),
            worlds_dir: dir.path().join("data/default-user/worlds"),
        };
        let cache = disk_cache::DiskCache::new(&data_root);

        // Plant a corrupt cache entry under the correct hashed name
        let meta = std::fs::metadata(chars.join("f.png")).unwrap();
        let (secs, nanos) = disk_cache::mtime_secs_nanos(&meta);
        let key = disk_cache::cache_key(
            &data_root,
            "default-user",
            "f.png",
            &disk_cache::format_mtime_ms(secs, nanos),
        );
        cache.put(&key, "NOT VALID JSON{{{").unwrap();

        // Must parse the PNG instead and still succeed
        let value = process_character_cached("f.png", &deps, &cache, &data_root, "default-user");
        assert_eq!(value["name"], "Fallback");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn all_route_streams_array() {
        let dir = tempfile::tempdir().unwrap();
        // Fabricate a user dir with one card
        let chars = dir.path().join("default-user/characters");
        std::fs::create_dir_all(&chars).unwrap();
        let mut png_bytes = Vec::new();
        png_bytes.extend_from_slice(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]);
        let card = serde_json::json!({
            "spec": "chara_card_v2", "spec_version": "2.0",
            "name": "T", "data": { "name": "T", "description": "d" }
        })
        .to_string();
        use base64::Engine;
        let b64 = base64::engine::general_purpose::STANDARD.encode(card.as_bytes());
        let mut data = b"chara\0".to_vec();
        data.extend_from_slice(b64.as_bytes());
        let chunk = png::build_chunk(b"tEXt", &data);
        png_bytes.extend_from_slice(&png::build_chunk(
            b"IHDR",
            &[0, 0, 0, 1, 0, 0, 0, 1, 8, 6, 0, 0, 0],
        ));
        png_bytes.extend_from_slice(&chunk);
        png_bytes.extend_from_slice(&png::build_chunk(b"IEND", &[]));
        std::fs::write(chars.join("t.png"), &png_bytes).unwrap();

        let cfg_text = format!(
            "dataRoot: {}\nport: 8000\n",
            dir.path().to_string_lossy().replace('\\', "/")
        );
        let cfg = crate::config::ResolvedConfig::from_text(&cfg_text, dir.path().to_path_buf());
        let state = Arc::new(AppState {
            thumb_limiter: std::sync::Arc::new(crate::thumbnails::ThumbLimiter::new(
                cfg.thumb_concurrency(),
            )),
            disk_cache: crate::disk_cache::DiskCache::new(&cfg.data_root),
            cfg: Arc::new(cfg),
            node: forward::NodeClient::new(),
        });

        let deps = user_dirs(&state, "default-user");
        let value = card::process_character("t.png", &deps);
        assert_eq!(value["name"], "T");
        assert_eq!(value["avatar"], "t.png");
        assert_eq!(value["data_size"], 2); // String(name="T") + String(description="d") = 1+1
    }

    /// Helper: build a card PNG file with the given name.
    fn write_card_png(chars: &std::path::Path, file: &str, card_name: &str) {
        use base64::Engine;
        let card = serde_json::json!({
            "spec": "chara_card_v2", "spec_version": "2.0",
            "name": card_name,
            "data": { "name": card_name, "description": "d" }
        })
        .to_string();
        let b64 = base64::engine::general_purpose::STANDARD.encode(card.as_bytes());
        let mut data = b"chara\0".to_vec();
        data.extend_from_slice(b64.as_bytes());
        let mut png_bytes = Vec::new();
        png_bytes.extend_from_slice(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]);
        png_bytes.extend_from_slice(&png::build_chunk(
            b"IHDR",
            &[0, 0, 0, 1, 0, 0, 0, 1, 8, 6, 0, 0, 0],
        ));
        png_bytes.extend_from_slice(&png::build_chunk(b"tEXt", &data));
        png_bytes.extend_from_slice(&png::build_chunk(b"IEND", &[]));
        std::fs::write(chars.join(file), &png_bytes).unwrap();
    }

    #[tokio::test]
    async fn batched_processing_preserves_readdir_order() {
        let dir = tempfile::tempdir().unwrap();
        let chars = dir.path().join("default-user/characters");
        std::fs::create_dir_all(&chars).unwrap();

        // 10 cards; readdir order is filesystem-dependent, so capture it.
        for i in 0..10 {
            write_card_png(&chars, &format!("card{i}.png"), &format!("C{i}"));
        }
        let names: Vec<String> = std::fs::read_dir(&chars)
            .unwrap()
            .flatten()
            .filter(|e| e.path().extension().map(|x| x == "png").unwrap_or(false))
            .filter_map(|e| e.file_name().into_string().ok())
            .collect();
        assert_eq!(names.len(), 10);

        // Process in batches of 3 (forces multiple batches with partial tails)
        let batch_size = 3;
        let deps = crate::card::CardDeps {
            characters_dir: chars.clone(),
            chats_dir: dir.path().join("default-user/chats"),
            worlds_dir: dir.path().join("default-user/worlds"),
        };

        let mut out_names: Vec<String> = Vec::new();
        for batch in names.chunks(batch_size) {
            let tasks: Vec<_> = batch
                .iter()
                .cloned()
                .map(|name| {
                    let d = deps.clone();
                    tokio::task::spawn_blocking(move || card::process_character(&name, &d))
                })
                .collect();
            for task in tasks {
                let value = task.await.unwrap();
                out_names.push(value["avatar"].as_str().unwrap().to_string());
            }
        }

        // Output order must equal readdir order despite concurrent parsing
        assert_eq!(out_names, names);
    }

    #[test]
    fn plan_batches_respects_byte_limit() {
        let entries = vec![
            ("a.png".to_string(), 60),
            ("b.png".to_string(), 60),
            ("c.png".to_string(), 60),
            ("d.png".to_string(), 10),
        ];
        let batches = plan_batches(&entries, 100);
        // a+b=120 > 100 → [a]; b+c=120 > 100 → [b]; c+d=70 ≤ 100 → [c,d]
        assert_eq!(batches.len(), 3);
        assert_eq!(batches[0].len(), 1);
        assert_eq!(batches[1].len(), 1);
        assert_eq!(batches[2].len(), 2);
    }

    #[test]
    fn plan_batches_allows_single_oversized_file() {
        let entries = vec![
            ("small.png".to_string(), 10),
            ("huge.png".to_string(), 1000),
            ("small2.png".to_string(), 10),
        ];
        let batches = plan_batches(&entries, 100);
        // small fits; huge exceeds limit but runs alone; small2 opens new batch
        assert_eq!(batches.len(), 3);
        assert_eq!(batches[0][0].0, "small.png");
        assert_eq!(batches[1][0].0, "huge.png");
        assert_eq!(batches[2][0].0, "small2.png");
    }

    #[test]
    fn plan_batches_empty_and_exact_fit() {
        assert!(plan_batches(&[], 100).is_empty());
        // exact boundary: sum == limit stays in one batch
        let entries = vec![("a".to_string(), 50), ("b".to_string(), 50)];
        let batches = plan_batches(&entries, 100);
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].len(), 2);
    }

    #[tokio::test]
    async fn process_batch_splits_and_retries_on_whole_batch_failure() {
        // All files corrupted → whole batch fails → split down to singles;
        // single failures become stubs without retry. Order must hold.
        let dir = tempfile::tempdir().unwrap();
        let chars = dir.path().join("default-user/characters");
        std::fs::create_dir_all(&chars).unwrap();
        for name in ["x1.png", "x2.png", "x3.png", "x4.png"] {
            std::fs::write(chars.join(name), b"\x89PNGgarbage").unwrap();
        }
        let deps = crate::card::CardDeps {
            characters_dir: chars,
            chats_dir: dir.path().join("default-user/chats"),
            worlds_dir: dir.path().join("default-user/worlds"),
        };
        let batch: Vec<BatchEntry> = ["x1.png", "x2.png", "x3.png", "x4.png"]
            .iter()
            .map(|n| (n.to_string(), 14))
            .collect();

        let (tx, mut rx) = tokio::sync::mpsc::channel::<Result<bytes::Bytes, std::io::Error>>(32);
        let mut first = true;
        process_batch(
            &batch,
            &deps,
            DEFAULT_BATCH_BYTES,
            false,
            &dir.path().to_path_buf(),
            "default-user",
            &mut first,
            &tx,
        )
        .await;
        drop(tx);

        // All stubs lack "name" → filtered → stream should be exactly [\n + \n]
        let mut collected = Vec::new();
        while let Some(Ok(chunk)) = rx.recv().await {
            collected.extend_from_slice(&chunk);
        }
        let text = String::from_utf8(collected).unwrap();
        // process_batch emits only card chunks (array framing lives in
        // characters_all); all stubs lack "name" → all filtered → empty.
        assert_eq!(text, "");
    }

    #[tokio::test]
    async fn process_batch_mixed_failure_keeps_order_and_good_results() {
        // good + bad: bad fails at single-file level (no split needed since not
        // whole batch), good must still be emitted, order preserved.
        let dir = tempfile::tempdir().unwrap();
        let chars = dir.path().join("default-user/characters");
        std::fs::create_dir_all(&chars).unwrap();
        write_card_png(&chars, "g1.png", "A");
        std::fs::write(chars.join("bad.png"), b"\x89PNGgarbage").unwrap();
        write_card_png(&chars, "g2.png", "B");
        let deps = crate::card::CardDeps {
            characters_dir: chars.clone(),
            chats_dir: dir.path().join("default-user/chats"),
            worlds_dir: dir.path().join("default-user/worlds"),
        };

        // Build entries with real sizes via metadata, readdir order
        let mut batch: Vec<BatchEntry> = Vec::new();
        for name in ["g1.png", "bad.png", "g2.png"] {
            let size = std::fs::metadata(chars.join(name)).unwrap().len();
            batch.push((name.to_string(), size));
        }

        let (tx, mut rx) = tokio::sync::mpsc::channel::<Result<bytes::Bytes, std::io::Error>>(32);
        let mut first = true;
        process_batch(
            &batch,
            &deps,
            DEFAULT_BATCH_BYTES,
            false,
            &dir.path().to_path_buf(),
            "default-user",
            &mut first,
            &tx,
        )
        .await;
        drop(tx);

        let mut collected = Vec::new();
        while let Some(Ok(chunk)) = rx.recv().await {
            collected.extend_from_slice(&chunk);
        }
        let text = String::from_utf8(collected).unwrap();
        // bad.png yields stub without name → filtered out; A and B in order
        assert!(text.contains("\"name\":\"A\""));
        assert!(text.contains("\"name\":\"B\""));
        let a_pos = text.find("\"name\":\"A\"").unwrap();
        let b_pos = text.find("\"name\":\"B\"").unwrap();
        assert!(a_pos < b_pos);
        assert!(!text.contains("\"name\":\"\""));
    }

    #[tokio::test]
    async fn corrupted_card_in_batch_yields_stub_and_does_not_break_others() {
        let dir = tempfile::tempdir().unwrap();
        let chars = dir.path().join("default-user/characters");
        std::fs::create_dir_all(&chars).unwrap();
        write_card_png(&chars, "good.png", "Good");
        // Truncated / non-PNG garbage
        std::fs::write(chars.join("bad.png"), b"\x89PNGcorrupted").unwrap();

        let deps = crate::card::CardDeps {
            characters_dir: chars.clone(),
            chats_dir: dir.path().join("default-user/chats"),
            worlds_dir: dir.path().join("default-user/worlds"),
        };

        let names = vec!["bad.png".to_string(), "good.png".to_string()];
        let mut results: Vec<serde_json::Value> = Vec::new();
        for batch in names.chunks(8) {
            let tasks: Vec<_> = batch
                .iter()
                .cloned()
                .map(|name| {
                    let d = deps.clone();
                    tokio::task::spawn_blocking(move || card::process_character(&name, &d))
                })
                .collect();
            for task in tasks {
                results.push(task.await.unwrap());
            }
        }

        // bad.png → zeroed stub (upstream catch semantics), no name → filtered
        assert!(results[0].get("name").is_none());
        assert_eq!(results[0]["date_added"], 0);
        // good.png parses fine alongside
        assert_eq!(results[1]["name"], "Good");
    }
}

/// Test-only wrappers around the B2 sync cores (HTTP wiring is exercised via
/// these; check_auth requires a live session which integration tests don't have).
#[doc(hidden)]
pub mod __test {
    pub async fn parse_multipart(
        content_type: &str,
        body: axum::body::Body,
    ) -> Result<super::MultipartUpload, String> {
        super::parse_multipart(content_type, body).await
    }
    use crate::card::CardDeps;

    pub fn create_sync(
        form: &serde_json::Value,
        ch_name: &str,
        upload: Option<&[u8]>,
        crop: Option<crate::avatar::Crop>,
        deps: &CardDeps,
        data_root: &std::path::Path,
    ) -> Result<String, String> {
        super::create_sync(form, ch_name, upload, crop, deps, data_root)
    }

    pub fn edit_full_sync(
        form: &serde_json::Value,
        avatar_url: &str,
        upload: Option<&[u8]>,
        crop: Option<crate::avatar::Crop>,
        deps: &CardDeps,
        data_root: &std::path::Path,
    ) -> Result<(), String> {
        super::edit_full_sync(form, avatar_url, upload, crop, deps, data_root)
    }

    pub fn edit_avatar_sync(
        upload: &[u8],
        avatar_url: &str,
        crop: Option<crate::avatar::Crop>,
        deps: &CardDeps,
        data_root: &std::path::Path,
    ) -> Result<(), String> {
        super::edit_avatar_sync(upload, avatar_url, crop, deps, data_root)
    }

    pub fn import_sync(
        upload: &[u8],
        format: &str,
        preserved: Option<&str>,
        deps: &CardDeps,
        data_root: &std::path::Path,
    ) -> Result<String, String> {
        super::import_sync(upload, format, preserved, deps, data_root)
    }
}
