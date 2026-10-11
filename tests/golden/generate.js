// Golden vector generator — run from the SillyTavern repo root:
//   node src-rs/tests/golden/generate.js
//
// Produces golden.json using the REAL upstream modules so the Rust
// implementation can be verified byte-for-byte against Node behavior:
//  - keygrip signing (HMAC-SHA1/base64url)
//  - cookie-session value encoding (base64(JSON))
//  - character-card-parser.js write()/read() round-trip on a real PNG
//  - png-chunk-text encode byte layout
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import Keygrip from 'keygrip';
import PNGtext from 'png-chunk-text';
import extract from 'png-chunks-extract';
import { write, read } from '../../../src/character-card-parser.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const out = {};

// 1. keygrip vectors: secret × data → signature
const keygrip = new Keygrip(['secret-one']);
const sessionPayloads = [
    { handle: 'default-user', csrfToken: 'abc123' },
    { handle: 'alice', csrfToken: 'd4f5e6', version: 2 },
    {},
];
out.keygrip = sessionPayloads.map((payload) => {
    const value = Buffer.from(JSON.stringify(payload), 'utf8').toString('base64');
    const data = `session=${value}`;
    return {
        payload,
        value,
        data,
        sig: keygrip.sign(data),
        // second key to test key rotation indexing
        sig_alt_secret: new Keygrip(['other']).sign(data),
    };
});

// 2. cookie-session serialize: Session.serialize = encode(sess) =
//    Buffer.from(JSON.stringify(body)).toString('base64') — captured above
//    via `value`.

// 3. Real PNG round-trip through character-card-parser (uses
//    png-chunks-extract + png-chunk-text internally):
//    write() a card onto a minimal PNG, then read() it back.
const cardJson = JSON.stringify({
    spec: 'chara_card_v2',
    spec_version: '2.0',
    name: 'Golden Card',
    description: 'A card with unicode: é😀',
    data: {
        name: 'Golden Card',
        description: 'A card with unicode: é😀',
        talkativeness: '0.5',
        fav: false,
        tags: ['a', 'b'],
    },
});

// Minimal 1x1 RGBA PNG (valid IHDR + IDAT + IEND), like a real avatar.
import zlib from 'node:zlib';
function crc32(buf) {
    let c;
    const table = [];
    for (let n = 0; n < 256; n++) {
        c = n;
        for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
        table[n] = c;
    }
    let crc = 0 ^ -1;
    for (let i = 0; i < buf.length; i++) crc = (crc >>> 8) ^ table[(crc ^ buf[i]) & 0xff];
    return (crc ^ -1) >>> 0;
}
function chunk(type, data) {
    const len = Buffer.alloc(4);
    len.writeUInt32BE(data.length);
    const typeBuf = Buffer.from(type, 'ascii');
    const crcBuf = Buffer.alloc(4);
    crcBuf.writeUInt32BE(crc32(Buffer.concat([typeBuf, data])));
    return Buffer.concat([len, typeBuf, data, crcBuf]);
}
const ihdr = Buffer.alloc(13);
ihdr.writeUInt32BE(1, 0);
ihdr.writeUInt32BE(1, 4);
ihdr[8] = 8; ihdr[9] = 6; ihdr[10] = 0; ihdr[11] = 0; ihdr[12] = 0;
const rawScanline = Buffer.from([0, 0x12, 0x34, 0x56, 0x78]); // filter 0 + RGBA
const idat = zlib.deflateSync(rawScanline);
const basePng = Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk('IHDR', ihdr),
    chunk('IDAT', idat),
    chunk('IEND', Buffer.alloc(0)),
]);

// write() adds chara (+ccv3 since card JSON is parseable).
// read() prefers ccv3, so readback has spec mutated to v3 — expected.
const cardPng = write(basePng, cardJson);
out.png_base64 = cardPng.toString('base64');
out.card_json = cardJson;
const readback = read(cardPng);
out.card_json_readback = readback;
{
    // Verify the only delta is the v3 spec mutation done by write()
    const v3 = JSON.parse(readback);
    const v2 = JSON.parse(cardJson);
    v3.spec = 'chara_card_v2';
    v3.spec_version = '2.0';
    out.readback_equals_card_after_unspec = JSON.stringify(v3) === JSON.stringify(v2);
}

// Also verify against raw chunk decode what keywords the file has:
const chunks = extract(new Uint8Array(cardPng));
out.chunk_keywords = chunks
    .filter((c) => c.name === 'tEXt')
    .map((c) => PNGtext.decode(c.data).keyword);

// 4. png-chunk-text encode byte layout for direct comparison with Rust
const textChunk = PNGtext.encode('chara', 'plain-value');
const textChunkBytes = Buffer.from(textChunk.data);
out.pngtext_encode_hex = textChunkBytes.toString('hex');
out.pngtext_encode_decoded = PNGtext.decode(textChunk.data);

// 5. Expected chunk structure of the card PNG (names in order)
out.chunk_names = chunks.map((c) => c.name);

fs.writeFileSync(path.join(__dirname, 'golden.json'), JSON.stringify(out, null, 2));
console.log('golden.json written:', path.join(__dirname, 'golden.json'));
console.log('readback matches card_json:', out.card_json_readback === cardJson);
