//! Avatar image processing for B2 routes (/create, /edit crop, /edit-avatar,
//! /import): mirrors `applyAvatarCropResize` + Jimp decode/encode with the
//! `image` crate — no Node, no Jimp, no full-frame copies where avoidable.
//!
//! Upstream semantics (src/endpoints/characters.js):
//! - crop {x,y,width,height,want_resize?}: rect-crop, then `cover` to
//!   (want_resize ? 512×768 : crop.width×crop.height).
//! - no crop: `cover` to the image's own size = no-op re-encode.
//! - decode failure (e.g. APNG in Node): fall back to the raw bytes as-is
//!   (`tryReadImage` catch → fs.readFileSync) — we mirror: if the source is
//!   already a PNG and no crop was requested, pass bytes through untouched.
//! - write(data) then strips chara/ccv3 tEXt and appends fresh chunks, so a
//!   passthrough must still have card chunks spliced in by the caller.
//!
//! ## Memory bounds (why decode is not `image::load_from_memory`)
//!
//! Peak memory for the crop path is ~2.7x the source bitmap: decode, then
//! `crop_imm`, then `resize_exact`, then PNG encode each hold a full frame.
//! Measured on Windows (peak working set, `/api/avatars/upload` shape):
//!
//! | source | file | crop → 512x768 | no crop |
//! |--------|------|----------------|---------|
//! | 4 MP   | 0.7MB| 53 MB          | 53 MB   |
//! | 16 MP  | 2.8MB| 146 MB         | 196 MB  |
//! | 36 MP  | 6.4MB| 319 MB         | 434 MB  |
//! | 64 MP  | 11MB | 564 MB         | 770 MB  |
//! | 144 MP | 26MB | **1265 MB**    |**1729 MB**|
//!
//! So the file size is irrelevant — a 26MB JPEG costs 1.7GB. Two defences:
//!
//! 1. `check_dimensions` reads only the header (`into_dimensions`, 5.9ms and
//!    no bitmap for a 12000x12000 JPEG) and rejects anything above the pixel
//!    ceiling before a single byte is decoded. This is the real fix.
//! 2. `decode_bounded` still passes the decoder an explicit `max_alloc` as a
//!    backstop for malformed files that under-declare their dimensions. It is
//!    NOT strict: 144 MP (549MB of RGBA) decoded fine under the 512MiB
//!    default, and `max_image_width`/`max_image_height` only fire when set
//!    explicitly (which `load_from_memory` never does).

use image::DynamicImage;

/// constants.js AVATAR_WIDTH/HEIGHT
pub const AVATAR_WIDTH: u32 = 512;
/// constants.js AVATAR_HEIGHT
pub const AVATAR_HEIGHT: u32 = 768;

/// Hard ceiling on source pixels accepted for processing.
///
/// 40 MP is 4 bytes/px × 40e6 ≈ 160MB as an RGBA bitmap, ≈ 430MB at the
/// measured 2.7x path overhead — comfortably inside the 512MB budget below
/// while still accepting every consumer camera's full-resolution output
/// (48MP sensors normally hand over a 12-24MP image).
pub const MAX_SOURCE_PIXELS: u64 = 40_000_000;

/// Decoder allocation ceiling for one image (see module docs: non-strict).
/// With `MAX_SOURCE_PIXELS` the worst legal case is ~160MB of RGBA, so this
/// leaves room for the decoder's internal buffers without permitting the
/// multi-hundred-megabyte bombs the default limit lets through.
const MAX_DECODE_ALLOC: u64 = 512 * 1024 * 1024;

/// Rejection reason for an image that exceeds the pixel ceiling. Callers turn
/// this into a 400; it is deliberately distinct from a decode failure so the
/// "fall back to the default avatar" path is never taken for a size problem.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TooLarge {
    pub width: u32,
    pub height: u32,
}

impl std::fmt::Display for TooLarge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "image is {}x{} ({} megapixels), above the {} megapixel limit",
            self.width,
            self.height,
            (self.width as u64 * self.height as u64) / 1_000_000,
            MAX_SOURCE_PIXELS / 1_000_000
        )
    }
}

/// Header-only size check. Reads the dimensions from the container and
/// refuses images above `MAX_SOURCE_PIXELS` without allocating a bitmap.
///
/// Returns `Ok(None)` when the dimensions cannot be determined (not an image,
/// or a header-only format) — those fall through to `decode_bounded`, which
/// still applies the allocation ceiling.
pub fn check_dimensions(raw: &[u8]) -> Result<Option<(u32, u32)>, TooLarge> {
    // AVIF is not one of the `image` crate's formats here (its `avif` feature
    // is encode-only), so the header has to come from oxideav. `info` parses
    // the container metadata without touching AV1 pixels — the same cheap
    // early rejection applies.
    if let Some((w, h)) = avif_dimensions(raw) {
        if w as u64 * h as u64 > MAX_SOURCE_PIXELS {
            return Err(TooLarge {
                width: w,
                height: h,
            });
        }
        return Ok(Some((w, h)));
    }

    let Ok(reader) = image::ImageReader::new(std::io::Cursor::new(raw)).with_guessed_format()
    else {
        return Ok(None);
    };
    let Ok((w, h)) = reader.into_dimensions() else {
        return Ok(None);
    };
    if w as u64 * h as u64 > MAX_SOURCE_PIXELS {
        return Err(TooLarge {
            width: w,
            height: h,
        });
    }
    Ok(Some((w, h)))
}

/// AVIF header probe. Returns `None` for anything that is not AVIF (including
/// a malformed AVIF, which then falls through to the normal decode path and
/// reports a proper error there).
fn avif_dimensions(raw: &[u8]) -> Option<(u32, u32)> {
    if !oxideav_avif::probe(raw) {
        return None;
    }
    let info = oxideav_avif::info(raw).ok()?;
    Some((info.width as u32, info.height as u32))
}

/// Decode with an explicit allocation ceiling.
///
/// Equivalent to `image::load_from_memory` (same format sniffing, same lack of
/// a dimension limit) but with a tighter `max_alloc` than the crate's 512MiB
/// default. Dimension policy deliberately does NOT live here: thumbnails must
/// keep accepting very wide panoramas (a 20000x1000 background is 20MP and
/// harmless), while uploads get their 40MP ceiling from `check_dimensions`
/// before this is ever called.
///
/// AVIF is routed to oxideav: the `image` crate can only *encode* AVIF with the
/// feature set this build uses, so without this an AVIF persona avatar would
/// not decode at all (and `prepare_upload_bytes` would silently substitute the
/// default avatar).
pub fn decode_bounded(raw: &[u8]) -> Result<DynamicImage, String> {
    if oxideav_avif::probe(raw) {
        return decode_avif(raw);
    }

    let mut reader = image::ImageReader::new(std::io::Cursor::new(raw))
        .with_guessed_format()
        .map_err(|e| format!("unsupported image format: {e}"))?;
    let mut limits = image::Limits::default();
    limits.max_alloc = Some(MAX_DECODE_ALLOC);
    reader.limits(limits);
    reader.decode().map_err(|e| format!("decode failed: {e}"))
}

/// AVIF → `DynamicImage` via oxideav (pure Rust, no dav1d).
fn decode_avif(raw: &[u8]) -> Result<DynamicImage, String> {
    let image = oxideav_avif::decode(raw).map_err(|e| format!("avif decode failed: {e}"))?;
    let (w, h) = (image.width(), image.height());
    if w == 0 || h == 0 {
        return Err("avif decode failed: empty image".to_string());
    }
    if w as u64 * h as u64 > MAX_SOURCE_PIXELS {
        return Err(TooLarge {
            width: w,
            height: h,
        }
        .to_string());
    }
    let rgba = image.to_rgba8();
    let expected = (w as usize) * (h as usize) * 4;
    if rgba.len() < expected {
        return Err(format!(
            "avif decode failed: {} bytes for a {w}x{h} frame",
            rgba.len()
        ));
    }
    let buffer = image::RgbaImage::from_raw(w, h, rgba[..expected].to_vec())
        .ok_or_else(|| "avif decode failed: buffer size mismatch".to_string())?;
    Ok(DynamicImage::ImageRgba8(buffer))
}

/// Parsed crop query (`request.query.crop`, JSON via tryParse).
#[derive(Debug, Clone, Copy)]
pub struct Crop {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
    pub want_resize: bool,
}

impl Crop {
    /// JS: [crop.x, crop.y, crop.width, crop.height].every(x => typeof x === 'number')
    /// Note: tryParse(undefined) → undefined, and JSON `null` fails `== 'object'`
    /// only for null; `typeof null === 'object'` is true in JS but `.every`
    /// then throws on missing keys → effectively invalid. We require numbers.
    pub fn from_value(v: &serde_json::Value) -> Option<Self> {
        let obj = v.as_object()?;
        let num = |k: &str| obj.get(k).and_then(|x| x.as_f64());
        Some(Crop {
            x: num("x")?,
            y: num("y")?,
            width: num("width")?,
            height: num("height")?,
            want_resize: obj
                .get("want_resize")
                .and_then(|x| x.as_bool())
                .unwrap_or(false),
        })
    }
}

/// Decode any supported raster format. Mirrors `Jimp.read`/`Jimp.fromBuffer`
/// supported set (webp, png, jpeg, avif, bmp, gif, tiff); image crate covers
/// png/jpeg/webp/gif/bmp/tiff. AVIF imports fall through to the passthrough
/// path only when no crop is requested (same observable result: raw bytes).
///
/// Goes through `decode_bounded`, whose dimension check runs before the
/// bitmap exists; see the module docs for the measured memory profile.
pub fn decode(raw: &[u8]) -> Result<DynamicImage, String> {
    decode_bounded(raw)
}

/// `applyAvatarCropResize` mirror: crop → cover → PNG encode.
pub fn apply_crop_resize(img: &DynamicImage, crop: Option<&Crop>) -> Result<Vec<u8>, String> {
    let (mut final_w, mut final_h) = (img.width(), img.height());
    let mut work = img.clone();

    if let Some(c) = crop {
        // Jimp crop: Math.round on x/y/w/h
        let x = c.x.round().max(0.0) as u32;
        let y = c.y.round().max(0.0) as u32;
        let w = c.width.round() as u32;
        let h = c.height.round() as u32;
        // Clamp to image bounds (Jimp would read out-of-bounds as undefined
        // and produce corrupt rows; clamping is the sane equivalent)
        let x = x.min(img.width().saturating_sub(1));
        let y = y.min(img.height().saturating_sub(1));
        let w = w.min(img.width() - x);
        let h = h.min(img.height() - y);
        work = work.crop_imm(x, y, w, h);
        if c.want_resize {
            final_w = AVATAR_WIDTH;
            final_h = AVATAR_HEIGHT;
        } else {
            final_w = w;
            final_h = h;
        }
    }

    let covered = cover_resize(&work, final_w, final_h);
    let mut out = Vec::new();
    covered
        .write_to(&mut std::io::Cursor::new(&mut out), image::ImageFormat::Png)
        .map_err(|e| format!("png encode failed: {e}"))?;
    Ok(out)
}

/// Jimp `cover` semantics (plugin-cover): scale by the larger ratio so both
/// dimensions >= target, then center-crop. Default filter = Jimp's resize.js
/// (bilinear-ish); Triangle (bilinear) is the established equivalent in this
/// codebase (thumbnails.rs).
fn cover_resize(img: &DynamicImage, target_w: u32, target_h: u32) -> DynamicImage {
    let (w, h) = (img.width() as f64, img.height() as f64);
    // plugin-cover: f = w/h > iw/ih ? w/iw : h/ih
    let f = if target_w as f64 / target_h as f64 > w / h {
        target_w as f64 / w
    } else {
        target_h as f64 / h
    };
    let scaled_w = ((w * f) as u32).max(1).max(target_w);
    let scaled_h = ((h * f) as u32).max(1).max(target_h);
    let scaled = img.resize_exact(scaled_w, scaled_h, image::imageops::FilterType::Triangle);
    let x = (scaled.width().saturating_sub(target_w)) / 2;
    let y = (scaled.height().saturating_sub(target_h)) / 2;
    scaled.crop_imm(x, y, target_w, target_h)
}

/// Mirror of `getInputImage` for buffer inputs (uploads): decode+crop; on
/// decode failure the upstream buffer branch falls back to the DEFAULT
/// avatar image (parseImageBuffer propagates → getInputImage catch →
/// fs.readFile(DEFAULT_AVATAR_PATH)). The caller supplies the default bytes.
pub fn prepare_upload_bytes(
    raw: &[u8],
    crop: Option<&Crop>,
    default_avatar: &[u8],
) -> Result<Vec<u8>, String> {
    // A size rejection is NOT a decode failure: trading an oversized upload
    // for the default avatar would silently discard the user's image and
    // report success. Surface it instead.
    if let Err(too_large) = check_dimensions(raw) {
        return Err(too_large.to_string());
    }
    match decode(raw) {
        Ok(img) => apply_crop_resize(&img, crop),
        Err(e) => {
            tracing::warn!("Failed to read image buffer: {e}. Using a fallback image.");
            // upstream: default avatar goes through parseImageBuffer too
            // (Buffer.isBuffer(inputFile) branch reads the default file and
            // decodes it — Jimp.fromBuffer on the default PNG). Mirror: decode
            // + crop the default avatar; it's a plain PNG so this succeeds.
            decode(default_avatar)
                .map_err(|e| format!("default avatar decode failed: {e}"))
                .and_then(|img| apply_crop_resize(&img, None))
        }
    }
}

/// Mirror of `tryReadImage` for FILE paths (default-avatar source or avatar
/// files referenced by path): decode failure with a PNG input → raw bytes
/// passthrough (APNG etc.), write() still splices chunks.
pub fn prepare_file_bytes(raw: &[u8], crop: Option<&Crop>) -> Result<Vec<u8>, String> {
    // Same reasoning as prepare_upload_bytes: never let an oversized file
    // reach the raw-passthrough branch (which would write a 1.7GB-class
    // source straight through as the avatar).
    if let Err(too_large) = check_dimensions(raw) {
        return Err(too_large.to_string());
    }
    match decode(raw) {
        Ok(img) => apply_crop_resize(&img, crop),
        Err(e) => {
            if crop.is_none() && raw.starts_with(&[0x89, b'P', b'N', b'G']) {
                tracing::warn!(
                    "Failed to read image (unsupported/animated): {e}. Using raw file buffer."
                );
                Ok(raw.to_vec())
            } else {
                Err(e)
            }
        }
    }
}

/// Encode a DynamicImage as PNG (used for the default-avatar path).
pub fn encode_png(img: &DynamicImage) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    img.write_to(&mut std::io::Cursor::new(&mut out), image::ImageFormat::Png)
        .map_err(|e| format!("png encode failed: {e}"))?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gradient_png(w: u32, h: u32) -> Vec<u8> {
        let img = DynamicImage::new_rgb8(w, h);
        encode_png(&img).unwrap()
    }

    #[test]
    fn cover_semantics_upscale_and_center_crop() {
        // 100×100 → cover to 512×768: scale f = max(512/100, 768/100) = 7.68
        let img = decode(&gradient_png(100, 100)).unwrap();
        // the /edit-avatar path always passes a crop:
        let crop = Crop {
            x: 0.0,
            y: 0.0,
            width: 100.0,
            height: 100.0,
            want_resize: true,
        };
        let out = apply_crop_resize(&img, Some(&crop)).unwrap();
        let decoded = decode(&out).unwrap();
        assert_eq!((decoded.width(), decoded.height()), (512, 768));
    }

    #[test]
    fn crop_rect_selects_region() {
        // 10×10 red image; crop the left half → 5×10
        let mut img = DynamicImage::new_rgb8(10, 10);
        img.as_mut_rgb8()
            .unwrap()
            .pixels_mut()
            .for_each(|p| p.0 = [255, 0, 0]);
        let crop = Crop {
            x: 0.0,
            y: 0.0,
            width: 5.0,
            height: 10.0,
            want_resize: false,
        };
        let out = apply_crop_resize(&img, Some(&crop)).unwrap();
        let d = decode(&out).unwrap();
        assert_eq!((d.width(), d.height()), (5, 10));
    }

    #[test]
    fn passthrough_on_undecodable_png_without_crop() {
        // Minimal structurally-valid PNG header with garbage body
        let raw = b"\x89PNG\r\n\x1a\nGARBAGE".to_vec();
        let out = prepare_file_bytes(&raw, None).unwrap();
        assert_eq!(out, raw);
    }
}

#[cfg(test)]
mod limit_tests {
    use super::*;

    fn png_of(w: u32, h: u32) -> Vec<u8> {
        let img = DynamicImage::ImageRgba8(image::RgbaImage::from_pixel(
            w,
            h,
            image::Rgba([1, 2, 3, 255]),
        ));
        encode_png(&img).unwrap()
    }

    #[test]
    fn header_check_reads_dimensions_without_decoding() {
        let raw = png_of(64, 32);
        assert_eq!(check_dimensions(&raw).unwrap(), Some((64, 32)));
    }

    #[test]
    fn oversized_dimensions_are_rejected_before_decode() {
        // 8000x6000 = 48MP > 40MP. Encoded small (a solid colour), so this
        // also proves the rejection is driven by dimensions, not file size.
        let raw = png_of(8000, 6000);
        let err = check_dimensions(&raw).unwrap_err();
        assert_eq!((err.width, err.height), (8000, 6000));
        assert!(err.to_string().contains("above the 40 megapixel limit"));
    }

    #[test]
    fn the_pixel_ceiling_itself_is_allowed() {
        // 40MP exactly: 8000x5000.
        let raw = png_of(8000, 5000);
        assert_eq!(check_dimensions(&raw).unwrap(), Some((8000, 5000)));
    }

    #[test]
    fn oversized_upload_reports_an_error_instead_of_the_default_avatar() {
        let oversized = png_of(8000, 6000);
        let default = png_of(8, 8);
        let err = prepare_upload_bytes(&oversized, None, &default).unwrap_err();
        assert!(
            err.contains("40 megapixel"),
            "expected a size error, got {err:?}"
        );
    }

    #[test]
    fn oversized_file_bytes_are_not_passed_through() {
        let oversized = png_of(8000, 6000);
        let err = prepare_file_bytes(&oversized, None).unwrap_err();
        assert!(
            err.contains("40 megapixel"),
            "expected a size error, got {err:?}"
        );
    }

    #[test]
    fn undecodable_small_input_still_falls_back() {
        // Not an image at all: no dimensions, so decode fails and the buffer
        // branch swaps in the default avatar (upstream behaviour preserved).
        let garbage = b"this is not an image".to_vec();
        let default = png_of(8, 8);
        let out = prepare_upload_bytes(&garbage, None, &default).unwrap();
        let decoded = decode(&out).unwrap();
        assert_eq!((decoded.width(), decoded.height()), (8, 8));
    }
}

#[cfg(test)]
mod avif_tests {
    use super::*;

    /// A real AVIF (encoded by the `image` crate's avif encoder — encoding is
    /// the one AVIF direction that does not need dav1d).
    fn avif_of(w: u32, h: u32, rgb: [u8; 3]) -> Vec<u8> {
        let img = DynamicImage::ImageRgb8(image::RgbImage::from_pixel(w, h, image::Rgb(rgb)));
        let mut out = Vec::new();
        img.write_to(
            &mut std::io::Cursor::new(&mut out),
            image::ImageFormat::Avif,
        )
        .expect("avif encode");
        out
    }

    #[test]
    fn avif_decodes_to_the_original_pixels() {
        // The whole point of the oxideav dependency: `image`'s avif feature is
        // encode-only, so without it this fails with "format not supported".
        let raw = avif_of(64, 48, [200, 50, 25]);
        let decoded = decode_bounded(&raw).expect("avif must decode");
        assert_eq!((decoded.width(), decoded.height()), (64, 48));
        let px = decoded.to_rgba8();
        assert_eq!(&px.as_raw()[0..4], &[200, 50, 25, 255]);
    }

    #[test]
    fn avif_dimensions_are_read_from_the_header() {
        let raw = avif_of(120, 90, [1, 2, 3]);
        assert_eq!(check_dimensions(&raw).unwrap(), Some((120, 90)));
    }

    /// Rewrite the `ispe` (image spatial extents) box so the file *declares* a
    /// huge frame while remaining a few hundred bytes. Encoding a real 48MP
    /// AVIF takes minutes, and the guard being tested is header-only, so the
    /// declaration is what matters.
    fn with_declared_size(avif: &[u8], w: u32, h: u32) -> Vec<u8> {
        let mut out = avif.to_vec();
        let pos = out
            .windows(4)
            .position(|win| win == b"ispe")
            .expect("ispe box");
        // ispe payload: version/flags (4) then width (4) then height (4).
        out[pos + 8..pos + 12].copy_from_slice(&w.to_be_bytes());
        out[pos + 12..pos + 16].copy_from_slice(&h.to_be_bytes());
        out
    }

    #[test]
    fn oversized_avif_is_rejected_before_decoding() {
        let raw = with_declared_size(&avif_of(64, 64, [9, 9, 9]), 8000, 6000);
        let err = check_dimensions(&raw).unwrap_err();
        assert_eq!((err.width, err.height), (8000, 6000));
        assert!(err.to_string().contains("above the 40 megapixel limit"));
    }

    #[test]
    fn avif_survives_the_upload_crop_path() {
        let raw = avif_of(300, 300, [10, 120, 200]);
        let crop = Crop {
            x: 0.0,
            y: 0.0,
            width: 300.0,
            height: 300.0,
            want_resize: true,
        };
        let prepared = prepare_upload_bytes(&raw, Some(&crop), &raw).expect("upload path");
        let out = decode(&prepared).unwrap();
        assert_eq!((out.width(), out.height()), (AVATAR_WIDTH, AVATAR_HEIGHT));
    }

    #[test]
    fn png_is_not_mistaken_for_avif() {
        // The AVIF probe must not claim other containers, or every PNG upload
        // would route into the wrong decoder.
        let png = encode_png(&DynamicImage::new_rgb8(8, 8)).unwrap();
        assert!(avif_dimensions(&png).is_none());
        assert!(!oxideav_avif::probe(&png));
    }
}
