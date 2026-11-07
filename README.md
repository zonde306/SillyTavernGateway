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
    "characters/edit": local      # JSON + multipart/crop branches
    "characters/edit-attribute": local
    "characters/edit-avatar": local
    "characters/rename": local
    "characters/export": local
    "characters/create": local
    "characters/import": local    # png/json/yaml; charx/byaf forward
    "thumbnail": local
    "backends/chat-completions/generate": local
  # batch_bytes: 268435456      # size budget for characters/all + decode/generation gates
  # heartbeat_secs: 15          # idle keep-alive for generations, 0 = off
```

All character PNG paths run through chunk surgery or the native `image`
crate — Node never decodes a character image again.

## Chat completions

`POST /api/backends/chat-completions/generate` is generated in Rust for the
providers upstream implements inline — `openai`, `openrouter`, `custom`,
`perplexity`, `groq`, `fireworks`, `nanogpt`, `pollinations`, `moonshot`,
`zai`, `siliconflow`, `workers_ai` (`cometapi` is disabled upstream and
returns the same 502). Claude, MakerSuite/Vertex, Cohere, DeepSeek, xAI,
Chutes, MiniMax, Electron Hub and Azure requests are forwarded to Node, so a
provider ported later keeps working without touching the route.

Body construction mirrors `src/endpoints/backends/chat-completions.js`
statement for statement — same `bodyParams` per provider, same key order,
same `undefined`-means-omitted semantics, same `excludeKeysByYaml` /
`mergeObjectWithYaml` / `flattenSchema` behaviour, and prompt
post-processing runs through the ported `mergeMessages` (locked against 140
golden vectors from the real `src/prompt-converters.js`).

Two behaviours differ from upstream on purpose:

**Idle keep-alive.** Providers can think for minutes before the first token;
nginx `proxy_read_timeout` and CDNs (Cloudflare's ~100 s) drop silent
connections. The gateway injects bytes that are valid for the protocol the
provider picked:

| upstream `Content-Type`        | injected while idle               |
|--------------------------------|-----------------------------------|
| `text/event-stream`            | `: keep-alive` comment lines      |
| `application/x-ndjson`, `jsonl`| blank lines                       |
| `application/json`             | leading space, then blank lines   |

Blank lines are ignored by every line/SSE parser, and leading whitespace is
ignored by `JSON.parse`. Interval: `gateway.heartbeat_secs` (default 15).
Responses also carry `Cache-Control: no-cache` and
`X-Accel-Buffering: no` so nginx does not buffer the stream.

**Priming token.** The first body bytes are written the moment the 200 header
is produced, before the provider has sent anything — the same token the idle
timer uses. Cloudflare and friends treat a response that has started flowing
far more leniently than one that is silent after the header, and some CLIs
refuse to hand over the connection until a byte arrives. It costs 14 bytes
and cannot be observed by a client parser.

**Size-weighted admission.** Node kept each request body alive (plus a
`JSON.stringify` copy) for the whole generation, so concurrent image-heavy
chats multiplied peak RSS. The gateway charges each generation
`1 + body/16MiB` permits against a semaphore sized from
`gateway.batch_bytes` (256 MiB default) and holds the permit until the
response stream ends. Oversized single requests are still allowed (their cost
is capped at the whole budget) so nobody is starved — they simply run alone.

Not ported: OpenRouter's live `isOpenRouterModelCacheable` probe (a 5 s
`GET /models` call) — Gemini prompt caching on OpenRouter therefore follows
the static allow-list only.

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
- `src/chat_completions.rs` — provider table, request translation, wire
  protocol detection, size-weighted generation gate
- `src/heartbeat.rs` — protocol-aware idle keep-alive body wrapper
- `src/prompt.rs` — mirror of `src/prompt-converters.js`
- `src/secrets.rs` — `secrets.json` reader (`readSecret` semantics)
- `src/routes.rs` — `/api/characters/*` and `/api/backends/chat-completions/
  generate` local handlers + fallback
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
- 2 multipart e2e proofs (real gateway process + signed session cookies):
  /create with the exact browser payload shape (empty-file avatar part) →
  200 + valid PNG; /edit multipart without upload → 200 + card JSON updated.
- 2 multipart empty-filename tests + 1 edit-preserves-image regression
  (multer `!filename → discard` mirror: /edit favoriting no longer resets
  the avatar to the default image).
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

- 4 prompt golden tests: 140 `postProcessPrompt` vectors (20 cases x 7
  processing types) plus `convertTextCompletionPrompt`,
  `embedOpenRouterMedia` and `addOpenRouterSignatures`, generated from the
  real `src/prompt-converters.js` by
  `node src-rs/tests/golden/generate_prompt.js`.
- 13 chat-completions tests: request translation per provider (URL, auth
  header, key order, omitted `undefined`s, YAML include/exclude),
  `flattenSchema` inlining and Google key filtering, cometapi parity,
  HMAC session affinity vs `node:crypto`, protocol detection, gate sizing.
- 6 heartbeat tests: SSE comments vs NDJSON blank lines vs JSON leading
  space, split JSON bodies staying parseable, unknown types getting the
  safest token, 40 MiB passing through intact.
- `python tests/e2e_generate.py` — end-to-end against the release binary and
  a fake OpenAI-compatible upstream: request translation, first-byte
  latency, heartbeats during a 3 s upstream silence, and two 20 MiB
  generations serialized by the gate at a 32 MiB budget.

Regenerate golden vectors after touching the generator scripts or when
upstream parser behaviour changes:
`node src-rs/tests/golden/generate.js && node src-rs/tests/golden/generate_prompt.js && cd src-rs && cargo test`.
