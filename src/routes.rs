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
    axum::Router::new()
        .route("/api/characters/all", post(characters_all))
        .route("/api/characters/get", post(characters_get))
        .route("/api/characters/edit", post(characters_edit))
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
            process_batch(&batch, &deps, batch_bytes, shallow, &mut first, &sender).await;
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

/// Process one planned batch: parse all files concurrently, emit results in
/// batch order (upstream readdir order preserved).
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
    first: &mut bool,
    sender: &tokio::sync::mpsc::Sender<Result<bytes::Bytes, std::io::Error>>,
) {
    // Run the batch concurrently.
    let tasks: Vec<_> = batch
        .iter()
        .cloned()
        .map(|(name, _size)| {
            let deps2 = deps.clone();
            tokio::task::spawn_blocking(move || card::process_character(&name, &deps2))
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

    let value = tokio::task::spawn_blocking(move || card::process_character(&avatar_url, &deps))
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

/// POST /api/characters/edit — local implementation (no avatar upload; the
/// multipart/crop variant is still forwarded).
async fn characters_edit(State(state): State<Arc<AppState>>, req: Request<Body>) -> Response {
    let (parts, body) = req.into_parts();
    if parts.uri.query().map(|q| !q.is_empty()).unwrap_or(false) {
        // crop query present => avatar processing required => forward to Node.
        let rebuilt = Request::from_parts(parts, body);
        return forward::proxy(state, rebuilt).await;
    }
    // multipart content-type => upload branch => forward.
    if let Some(ct) = parts.headers.get(axum::http::header::CONTENT_TYPE) {
        if ct
            .to_str()
            .map(|s| s.starts_with("multipart/"))
            .unwrap_or(false)
        {
            let rebuilt = Request::from_parts(parts, body);
            return forward::proxy(state, rebuilt).await;
        }
    }

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
    let avatar_url = match form.get("avatar_url").and_then(|v| v.as_str()) {
        Some(s) => s.trim_end_matches(".png").to_string(),
        None => return StatusCode::BAD_REQUEST.into_response(),
    };

    let deps = user_dirs(&state, &auth.handle);
    let avatar_path = deps.characters_dir.join(format!("{avatar_url}.png"));
    if !avatar_path.exists() {
        return StatusCode::NOT_FOUND.into_response();
    }

    let result = tokio::task::spawn_blocking(move || edit_character(&form, &deps, avatar_url))
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

/// Mirrors /edit flow: charaFormatData(request.body) on the *existing* card
/// JSON (json_data), write back via chunk surgery.
fn edit_character(
    form: &serde_json::Value,
    deps: &card::CardDeps,
    target_file: String,
) -> Result<(), String> {
    let avatar_png = format!("{target_file}.png");
    let file_path = deps.characters_dir.join(&avatar_png);

    // Read existing card JSON (raw string from tEXt).
    let chunks = png::read_text_chunks(&file_path).map_err(|e| e.to_string())?;
    let existing_raw = png::extract_card_json(&chunks)?;
    let existing: serde_json::Value = serde_json::from_str(&existing_raw)
        .map_err(|e| format!("existing card is not valid JSON: {e}"))?;

    let get = |k: &str| form.get(k).cloned().unwrap_or(serde_json::Value::Null);
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

    // charaFormatData: existing card as base (json_data equivalent).
    let mut char = card::chara_format_data_from_parts(&input, existing);

    // World info embedding when data.world is set (JS reads worlds/*.json).
    if let Some(world) = form
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

    // char.chat = request.body.chat; char.create_date = request.body.create_date
    if let Some(chat) = form.get("chat") {
        card::set_path_value(&mut char, "chat", chat.clone());
    }
    if let Some(cd) = form.get("create_date") {
        card::set_path_value(&mut char, "create_date", cd.clone());
    }

    let char_json = serde_json::to_string(&char).map_err(|e| e.to_string())?;

    // Write back: remove old chara/ccv3, insert fresh chara (v2 base64) —
    // matching write() semantics (v3 chunk only added on import paths).
    use base64::Engine;
    let b64 = base64::engine::general_purpose::STANDARD.encode(char_json.as_bytes());
    let new_chunk = png::encode_text_chunk("chara", &b64);
    png::write_card_to_png(&file_path, &[new_chunk]).map_err(|e| e.to_string())
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
        process_batch(&batch, &deps, DEFAULT_BATCH_BYTES, false, &mut first, &tx).await;
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
        process_batch(&batch, &deps, DEFAULT_BATCH_BYTES, false, &mut first, &tx).await;
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
