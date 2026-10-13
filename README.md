# st-gateway

Rust gateway for SillyTavern. Zero-modification deployment: Node keeps running
untouched; the gateway is the single public entry and transparently forwards
everything not yet migrated.

## Why

`/api/characters/all` and `/api/characters/get` OOM on large PNGs (~30MB):
upstream reads the whole file, copies all chunks (including multi-MB IDAT),
parses JSON, re-serializes — with ~5-6x file-size peak memory and unbounded
concurrency. `/api/characters/edit` is worse: it decodes the full image to
RGBA via Jimp and re-encodes just to update a text chunk.

The gateway serves these routes locally with chunk-level surgery: memory is
proportional to tEXt sizes (KBs), independent of image dimensions. Pixels are
never decoded; images stay byte-identical except for the metadata chunks.

`performance.lazyLoadCharacters` is mirrored: when enabled (see config.yaml),
`/api/characters/all` returns the upstream `toShallow` projection per card
(list metadata only, no card bodies / `json_data` / character books). The
frontend lazy-loads full cards via `/api/characters/get` on demand. With
tousands of cards this shrinks the list payload ~100x and avoids proxy-buffer
OOMs (nginx included).

## Thumbnail route

`/thumbnail` is served locally with the `image` crate (native decode, no
WASM): ~10x faster cold generation than Jimp, fully parallel. Decode
concurrency derives from `gateway.batch_bytes` (`batch_bytes / 8 MiB` per
in-flight decode, clamped 1..16) so total decode memory stays inside the
same budget that governs `/all` batching. Animated APNG/WebP detection,
Firefox cache headers, sanitize validation, mtime invalidation, and
config-driven format/quality/dimensions all mirror upstream.

## Shared disk cache

The gateway reads AND writes the same on-disk cache the Node backend uses
(`data/_cache/characters/`, node-persist format):

- key: `path.join(charactersDir, item) + "-" + mtimeMs` (identical to Node's
  `getCacheKey`, including OS separators and JS double formatting of
  mtimeMs)
- entry file: `sha256hex(key)`, content `JSON.stringify({key, value})`
- cache hit ⇒ PNG chunk scan skipped entirely (the expensive step: tEXt
  extraction + base64 + JSON parse); only the cheap per-request fields
  (avatar, stats, chat sizes) are recomputed
- cache miss ⇒ parse, then write back — Node's next read hits the entry the
  gateway filled, and vice versa
- `/edit` invalidates via mtime change (same mechanism as Node) plus a
  prune pass that removes entries whose source file changed or vanished

This means a mixed deployment (some users on Node, some via the gateway, or
Node still serving non-migrated routes that touch the cache) stays coherent
with zero coordination.

## Configuration

The gateway reads **`./config.yaml`** — the SillyTavern config file, resolved
against the working directory (i.e. where `start.sh` runs). Override the path
with `--config <path>` or the `ST_GATEWAY_CONFIG` environment variable.

It shares the standard ST keys (`port`, `dataRoot`) and adds a `gateway:`
section, which the Node backend ignores:

```yaml
gateway:
  listen_addr: "127.0.0.1:8000"   # omit => auto-bind 0.0.0.0:<port>
  node_addr: "127.0.0.1:8001"     # omit => 127.0.0.1:<port + 1>
  routes:
    "characters/all": local       # local = Rust, forward = proxy to Node
    "characters/get": local
    "characters/edit": local
```

## Auth replication (verified against upstream sources)

- cookie-session: `session-<sha256(hostname)[0..8]>` cookie = base64(JSON),
  signature = keygrip HMAC-SHA1/base64url of `"<name>=<value>"` with the
  secret from `data/cookie-secret.txt`.
- csrf-sync: mutating requests must carry `x-csrf-token` equal to
  `session.csrfToken`.
- User resolution mirrors `setUserDataMiddleware`: `session.handle` →
  `data/<handle>/`; empty handle → `default-user`.

The gateway only *verifies* sessions; Node remains the issuer (login/logout
are forwarded).

## Layout

- `src/png.rs` — streamed PNG chunk scanner (read/write, CRC32, lenient base64)
- `src/card.rs` — card normalization mirror: `getCharaCardV2`, `readFromV2`,
  `convertToV2`/`charaFormatData`, `convertWorldInfoToCharacterBook`,
  `unsetPrivateFields`, `calculateDataSize` (UTF-16 semantics),
  `humanizedDateTime`
- `src/auth.rs` — keygrip/cookie-session/CSRF replication
- `src/forward.rs` — transparent proxy to Node
- `src/routes.rs` — `/api/characters/all|get|edit` local handlers + fallback
- `src/config.rs` — config.yaml loading (`--config` / `ST_GATEWAY_CONFIG` /
  `./config.yaml`)

## Running

1. Start SillyTavern's Node backend on the `node_addr` port, e.g.
   `node server.js --port 8001`. Node code is unchanged.
2. `cargo run --release` in `src-rs/` from the repo root (reads
   `./config.yaml`).
3. Point users at the gateway's `listen_addr`.

## Migration / rollback

Flip a route between `local` and `forward` in `config.yaml` — instant,
per-route, no restarts of Node required. `/edit` with multipart upload or a
`crop` query is automatically forwarded (pixel processing stays in Node until
phase B).

## Tests

`cargo test` from `src-rs/`:

- 33 unit tests (CRC32 vectors, chunk surgery, V2 defaults, config schema,
  UTF-16 data size, size-based batching, split-and-retry on whole-batch
  failure, single-file stub semantics without retry, toShallow projection
  and lodash-get defaults, cache hit == full parse, corrupt-cache fallback,
  node-persist datum format, ...).
- 6 B2 tests: create (default avatar + card chunks), create with upload+crop
  (512×768 want_resize), edit-avatar (image replaced, card data survives),
  import png v2 roundtrip, import yaml, edit-full JSON branch (chunk surgery
  keeps image decodable).
- 5 B1 tests: sanitize-filename vectors from the REAL Node package, PNG-name
  dedup, edit-attribute dual-level field writes, rename end-to-end (card
  rewrite + chats migration + unlink), export-json private-field stripping.
- 1 shared-cache proof: reads a REAL Node-produced cache entry from
  `data/_cache/characters/` and verifies the Rust hash lands on the same
  file with identical value (skip-vacuous when no cache exists).
- 11 golden tests in `tests/golden.rs` verified against **real upstream Node
  modules**: reference vectors are generated by `node
  src-rs/tests/golden/generate.js` (uses `keygrip`, `png-chunk-text`, and
  `character-card-parser.js` themselves) and pinned in
  `tests/golden/golden.json`. Keygrip signatures, tEXt byte layout, PNG
  round-trips, and chunk ordering are compared byte-for-byte.

Regenerate golden vectors after touching `tests/golden/generate.js` or when
upstream parser behavior changes:
`node src-rs/tests/golden/generate.js && cd src-rs && cargo test`.
