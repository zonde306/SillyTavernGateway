//! PNG chunk-level reader/writer. Never decodes pixel data: memory usage is
//! proportional to tEXt chunk sizes (KBs), independent of image dimensions.
//!
//! Mirrors upstream semantics:
//! - `character-card-parser.js#read`: prefer tEXt keyword `ccv3` over `chara`
//!   (case-insensitive), base64-decode the text.
//! - `png-chunk-text.js#encode`: tEXt data = `keyword\0text`, latin1 bytes.
//! - `character-card-parser.js#write`: remove existing `chara`/`ccv3` tEXt
//!   chunks, insert new ones before IEND.

use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

const PNG_SIGNATURE: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];

pub struct TextChunk {
    pub keyword: String,
    pub text: Vec<u8>, // raw latin1 bytes of the text field
}

/// Streamed scan of a PNG file collecting only tEXt chunks. Non-tEXt chunks
/// are skipped via seek, so peak memory is the sum of tEXt sizes only.
pub fn read_text_chunks(path: &std::path::Path) -> std::io::Result<Vec<TextChunk>> {
    let mut file = std::fs::File::open(path)?;
    let mut sig = [0u8; 8];
    file.read_exact(&mut sig)?;
    if sig != PNG_SIGNATURE {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "not a PNG file",
        ));
    }

    let mut out = Vec::new();
    loop {
        let mut len_buf = [0u8; 4];
        if file.read_exact(&mut len_buf).is_err() {
            break; // truncated tail: treat as end
        }
        let len = u32::from_be_bytes(len_buf) as u64;
        let mut ctype = [0u8; 4];
        file.read_exact(&mut ctype)?;

        if &ctype == b"tEXt" {
            let mut data = vec![0u8; len as usize];
            file.read_exact(&mut data)?;
            if let Some(chunk) = parse_text_chunk(&data) {
                out.push(chunk);
            }
            file.seek(SeekFrom::Current(4))?; // skip CRC
        } else {
            // skip data + CRC
            file.seek(SeekFrom::Current((len as i64) + 4))?;
            if &ctype == b"IEND" {
                break;
            }
        }
    }
    Ok(out)
}

/// tEXt data = keyword \0 text (latin1).
fn parse_text_chunk(data: &[u8]) -> Option<TextChunk> {
    let nul = data.iter().position(|&b| b == 0)?;
    let keyword = String::from_utf8_lossy(&data[..nul]).to_string();
    Some(TextChunk {
        keyword,
        text: data[nul + 1..].to_vec(),
    })
}

/// Extract the character card JSON string from tEXt chunks, mirroring
/// `character-card-parser.js#read`: ccv3 wins over chara, case-insensitive.
/// The JS pipeline is latin1-decode → base64-decode → utf8; latin1 maps bytes
/// 1:1, so we base64-decode raw bytes directly (lenient like JS atob).
pub fn extract_card_json(chunks: &[TextChunk]) -> Result<String, String> {
    let find = |kw: &str| {
        chunks
            .iter()
            .find(|c| c.keyword.eq_ignore_ascii_case(kw))
            .map(|c| c.text.clone())
    };
    if let Some(raw) = find("ccv3").or_else(|| find("chara")) {
        // Node: `Buffer.from(text, 'base64').toString('utf8')` — the base64
        // decode never throws and the utf8 decode is lossy (U+FFFD for bad
        // bytes). Cards written by other tools (trailing partial groups, a
        // stray '=' or whitespace, non-utf8 payloads) must still load, so
        // mirror exactly instead of failing with a JSON parse error.
        let bytes = base64_decode_node(&raw);
        return Ok(String::from_utf8_lossy(&bytes).to_string());
    }
    Err("no chara/ccv3 tEXt chunk".to_string())
}

/// Base64 decode matching Node's `Buffer.from(str, 'base64')` byte-for-byte.
///
/// Node stops at the first `=`, ignores every non-alphabet character, and
/// keeps a trailing group of 2 or 3 characters (1 byte / 2 bytes) while
/// discarding a lone leftover character. It never reports an error.
pub fn base64_decode_node(input: &[u8]) -> Vec<u8> {
    let mut sextets: Vec<u8> = Vec::with_capacity(input.len());
    for &b in input {
        if b == b'=' {
            break; // Node treats padding as end-of-data
        }
        let v = match b {
            b'A'..=b'Z' => b - b'A',
            b'a'..=b'z' => b - b'a' + 26,
            b'0'..=b'9' => b - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            _ => continue, // whitespace, newlines and any other junk
        };
        sextets.push(v);
    }

    let mut out = Vec::with_capacity(sextets.len() / 4 * 3 + 2);
    for group in sextets.chunks(4) {
        // 4 chars → 3 bytes, 3 → 2, 2 → 1, 1 → nothing (Node's leftovers)
        let n = group.len();
        if n < 2 {
            break;
        }
        out.push((group[0] << 2) | (group[1] >> 4));
        if n >= 3 {
            out.push((group[1] << 4) | (group[2] >> 2));
        }
        if n == 4 {
            out.push((group[2] << 6) | group[3]);
        }
    }
    out
}

/// Backwards-compatible alias: decoding is now always Node-lenient, so this
/// never returns None. Kept for the existing test surface.
pub fn base64_decode_lenient(input: &[u8]) -> Option<Vec<u8>> {
    Some(base64_decode_node(input))
}

/// Encode a tEXt chunk like `png-chunk-text.encode`: latin1 bytes of
/// `keyword\0text`, wrapped in a full chunk with length/type/CRC.
/// Non-latin1 characters are replaced with `?` (matches JS behavior of
/// latin1 encoding with replacement for png-chunk-text's `write` usage).
pub fn encode_text_chunk(keyword: &str, text: &str) -> Vec<u8> {
    let to_latin1 = |s: &str| -> Vec<u8> {
        s.chars()
            .map(|c| {
                let cp = c as u32;
                if cp <= 0xFF {
                    cp as u8
                } else {
                    b'?'
                }
            })
            .collect()
    };
    let mut data = to_latin1(keyword);
    data.push(0);
    data.extend(to_latin1(text));
    build_chunk(b"tEXt", &data)
}

fn crc32(data: &[u8]) -> u32 {
    // Standard PNG CRC-32 (IEEE), bitwise implementation.
    let mut crc: u32 = 0xFFFF_FFFF;
    for &b in data {
        crc ^= b as u32;
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

/// Build a full PNG chunk: len + type + data + CRC.
pub fn build_chunk(ctype: &[u8; 4], data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() + 12);
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    out.extend_from_slice(ctype);
    out.extend_from_slice(data);
    let mut crc_input = Vec::with_capacity(4 + data.len());
    crc_input.extend_from_slice(ctype);
    crc_input.extend_from_slice(data);
    out.extend_from_slice(&crc32(&crc_input).to_be_bytes());
    out
}

/// Rewrite a PNG file's tEXt chunks: drop existing `chara`/`ccv3` chunks and
/// insert the provided new chunks before IEND (mirrors
/// `character-card-parser.js#write`). All other chunks (IDAT etc.) are copied
/// byte-for-byte — no pixel decoding, no re-encoding, image stays lossless.
///
/// Writes via temp file + rename (write-file-atomic semantics).
pub fn write_card_to_png(src: &Path, new_chunks: &[Vec<u8>]) -> std::io::Result<()> {
    write_card_to_png_from_to(src, src, new_chunks)
}

/// Same surgery but writing to a (possibly different) destination path —
/// used by /rename to create the renamed PNG without touching the original.
///
/// The destination is never written in place: data goes to a UNIQUE temp file
/// next to it and is renamed over it only once complete. A failure therefore
/// leaves the original byte-identical (upstream writeFileAtomic semantics).
/// A fixed temp name would let two concurrent writers truncate each other's
/// output and then race on the rename (one of them failing with ENOENT on a
/// temp that the other already renamed away) — that corrupted cards.
pub fn write_card_to_png_from_to(
    src: &Path,
    dst: &Path,
    new_chunks: &[Vec<u8>],
) -> std::io::Result<()> {
    let mut file = std::fs::File::open(src)?;
    crate::atomic::write_with(dst, |out| splice_chunks(&mut file, new_chunks, out))
}

/// Splice card tEXt chunks into a complete PNG already held in memory.
/// Used by the avatar-upload write path so the *destination* file is never
/// truncated before the splice succeeds.
pub fn splice_card_bytes(png: &[u8], new_chunks: &[Vec<u8>]) -> std::io::Result<Vec<u8>> {
    let mut src = std::io::Cursor::new(png);
    let mut out = Vec::with_capacity(png.len() + 4096);
    splice_chunks(&mut src, new_chunks, &mut out)?;
    Ok(out)
}

/// Stream `src` (a PNG) to `out`, dropping existing `chara`/`ccv3` tEXt chunks
/// and inserting `new_chunks` right before IEND. Non-card chunks are copied
/// byte-for-byte (CRC preserved), so the image stream is never re-encoded.
fn splice_chunks<R: Read, W: std::io::Write>(
    src: &mut R,
    new_chunks: &[Vec<u8>],
    out: &mut W,
) -> std::io::Result<()> {
    let mut sig = [0u8; 8];
    src.read_exact(&mut sig)?;
    if sig != PNG_SIGNATURE {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "not a PNG file",
        ));
    }
    out.write_all(&PNG_SIGNATURE)?;

    loop {
        let mut header = [0u8; 8]; // len + type
        if src.read_exact(&mut header).is_err() {
            break; // truncated tail: nothing left to append to
        }
        let len = u32::from_be_bytes([header[0], header[1], header[2], header[3]]) as u64;
        let ctype = [header[4], header[5], header[6], header[7]];

        let mut data = vec![0u8; len as usize];
        src.read_exact(&mut data)?;
        let mut crc = [0u8; 4];
        src.read_exact(&mut crc)?;

        let is_text = &ctype == b"tEXt";
        let is_card_text = is_text && {
            let nul = data.iter().position(|&b| b == 0).unwrap_or(0);
            let kw = String::from_utf8_lossy(&data[..nul]);
            kw.eq_ignore_ascii_case("chara") || kw.eq_ignore_ascii_case("ccv3")
        };

        if &ctype == b"IEND" {
            for chunk in new_chunks {
                out.write_all(chunk)?;
            }
            out.write_all(&header)?;
            out.write_all(&data)?;
            out.write_all(&crc)?;
            break;
        }

        if !is_card_text {
            out.write_all(&header)?;
            out.write_all(&data)?;
            out.write_all(&crc)?;
        }
        // card tEXt chunks are silently dropped
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal valid PNG: sig + IHDR + tEXt + IDAT + tEXt + IEND.
    fn minimal_png() -> Vec<u8> {
        let mut png = Vec::new();
        png.extend_from_slice(&PNG_SIGNATURE);
        // IHDR with plausible dims (1x1, 8-bit RGBA)
        let mut ihdr = Vec::new();
        ihdr.extend_from_slice(&1u32.to_be_bytes());
        ihdr.extend_from_slice(&1u32.to_be_bytes());
        ihdr.extend_from_slice(&[8, 6, 0, 0, 0]);
        png.extend_from_slice(&build_chunk(b"IHDR", &ihdr));
        png.extend_from_slice(&build_chunk(b"tEXt", b"chara\0AAAA"));
        png.extend_from_slice(&build_chunk(b"IDAT", &[1, 2, 3, 4]));
        png.extend_from_slice(&build_chunk(b"tEXt", b"ccv3\0BBBB"));
        png.extend_from_slice(&build_chunk(b"IEND", &[]));
        png
    }

    #[test]
    fn read_chunks_and_extract_prefers_ccv3() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("c.png");
        std::fs::write(&path, minimal_png()).unwrap();
        let chunks = read_text_chunks(&path).unwrap();
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].keyword, "chara");
        let json = extract_card_json(&chunks).unwrap();
        // "AAAA" base64-decodes to 3 bytes; "BBBB" likewise. We must get ccv3's.
        // base64("ccv3 text") here is literally "BBBB" → bytes \x04\x14\x41 ... verify len
        let chara_bytes = base64_decode_lenient(b"AAAA").unwrap();
        let ccv3_bytes = base64_decode_lenient(b"BBBB").unwrap();
        assert_ne!(chara_bytes, ccv3_bytes);
        assert_eq!(json.as_bytes(), ccv3_bytes);
    }

    #[test]
    fn write_replaces_card_chunks_and_keeps_idat() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("c.png");
        std::fs::write(&path, minimal_png()).unwrap();

        let new_chunk = encode_text_chunk("chara", "newdata");
        write_card_to_png(&path, &[new_chunk]).unwrap();

        let chunks = read_text_chunks(&path).unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].keyword, "chara");
        assert_eq!(chunks[0].text, b"newdata");
    }

    #[test]
    fn crc32_known_vector() {
        // CRC-32("123456789") = 0xCBF43926
        assert_eq!(crc32(b"123456789"), 0xCBF43926);
    }

    /// Vectors captured from Node:
    ///   Buffer.from(s, 'base64').toString('hex')
    #[test]
    fn base64_matches_node_leniency() {
        let cases: &[(&str, &str)] = &[
            ("eyJhIjoxfQ", "7b2261223a317d"),
            ("eyJhIjoxf", "7b2261223a31"), // missing final sextet
            ("eyJh=IjoxfQ", "7b2261"),     // padding ends the data
            ("", ""),
            ("A", ""), // lone leftover: dropped
            ("AB", "00"),
            ("ABC", "0010"),
            ("ABCD", "001083"),
            ("AB=C", "00"),
            ("AB==CD", "00"),
            ("eyJh Ijox fQ==", "7b2261223a317d"), // embedded spaces
        ];
        for (input, expect_hex) in cases {
            let got: String = base64_decode_node(input.as_bytes())
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect();
            assert_eq!(&got, expect_hex, "input {input}");
        }
    }

    /// A card whose payload is not valid UTF-8 must still parse (Node uses a
    /// lossy utf8 decode), instead of failing the whole character list.
    #[test]
    fn non_utf8_card_payload_is_lossy_not_fatal() {
        // base64 of 0xFF 0xFE 'A'
        let raw = base64_decode_node(b"//5B");
        assert_eq!(raw, vec![0xFF, 0xFE, b'A']);
        let chunk = TextChunk {
            keyword: "chara".into(),
            text: b"//5B".to_vec(),
        };
        let out = extract_card_json(&[chunk]).unwrap();
        assert!(out.ends_with('A'));
        assert!(out.contains('\u{FFFD}'));
    }
}
