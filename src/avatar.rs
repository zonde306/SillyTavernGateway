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

use image::DynamicImage;

/// constants.js AVATAR_WIDTH/HEIGHT
pub const AVATAR_WIDTH: u32 = 512;
pub const AVATAR_HEIGHT: u32 = 768;

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
            want_resize: obj.get("want_resize").and_then(|x| x.as_bool()).unwrap_or(false),
        })
    }
}

/// Decode any supported raster format. Mirrors `Jimp.read`/`Jimp.fromBuffer`
/// supported set (webp, png, jpeg, avif, bmp, gif, tiff); image crate covers
/// png/jpeg/webp/gif/bmp/tiff. AVIF imports fall through to the passthrough
/// path only when no crop is requested (same observable result: raw bytes).
pub fn decode(raw: &[u8]) -> Result<DynamicImage, String> {
    image::load_from_memory(raw).map_err(|e| format!("decode failed: {e}"))
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

/// Mirror of `getInputImage` + `write` composition:
/// - Buffer input → parseImageBuffer (decode+crop)
/// - decode failure → raw file bytes (tryReadImage catch)
/// Returns the processed PNG bytes ready for chunk splicing.
pub fn prepare_image_bytes(raw: &[u8], crop: Option<&Crop>) -> Result<Vec<u8>, String> {
    match decode(raw) {
        Ok(img) => apply_crop_resize(&img, crop),
        Err(e) => {
            // tryReadImage catch: unsupported image (APNG) → raw file buffer.
            // write() will still splice chunks; the image data stays as-is.
            if crop.is_none() && raw.starts_with(&[0x89, b'P', b'N', b'G']) {
                Ok(raw.to_vec())
            } else {
                // crop requested but undecodable: upstream would fall back to
                // DEFAULT_AVATAR_PATH for buffers... but only on read error,
                // not decode error (decode error → raw buffer). Keep raw.
                tracing::warn!("avatar decode failed, passing through raw: {e}");
                Ok(raw.to_vec())
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
        let crop = Crop { x: 0.0, y: 0.0, width: 100.0, height: 100.0, want_resize: true };
        let out = apply_crop_resize(&img, Some(&crop)).unwrap();
        let decoded = decode(&out).unwrap();
        assert_eq!((decoded.width(), decoded.height()), (512, 768));
    }

    #[test]
    fn crop_rect_selects_region() {
        // 10×10 red image; crop the left half → 5×10
        let mut img = DynamicImage::new_rgb8(10, 10);
        img.as_mut_rgb8().unwrap().pixels_mut().for_each(|p| p.0 = [255, 0, 0]);
        let crop = Crop { x: 0.0, y: 0.0, width: 5.0, height: 10.0, want_resize: false };
        let out = apply_crop_resize(&img, Some(&crop)).unwrap();
        let d = decode(&out).unwrap();
        assert_eq!((d.width(), d.height()), (5, 10));
    }

    #[test]
    fn passthrough_on_undecodable_png_without_crop() {
        // Minimal structurally-valid PNG header with garbage body
        let raw = b"\x89PNG\r\n\x1a\nGARBAGE".to_vec();
        let out = prepare_image_bytes(&raw, None).unwrap();
        assert_eq!(out, raw);
    }
}
