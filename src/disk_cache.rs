//! Shared on-disk cache, byte-compatible with the Node backend's
//! node-persist DiskCache (`src/endpoints/characters.js`), so both processes
//! read AND write the same cache entries in `data/_cache/characters/`.
//!
//! Format verified against a real cache file produced by Node:
//! - entry path: `<cacheDir>/<sha256hex(key)>`
//! - key: `path.join(charactersDir, item) + "-" + stat.mtimeMs`
//!   (OS-native separators, relative if DATA_ROOT is relative; mtimeMs is a
//!   JS double, e.g. `1738053032117.1682`)
//! - file content: `JSON.stringify({ key, value, ttl })` — `ttl` is absent
//!   when undefined (Node passes `ttl: false` → `calcTTL` → undefined);
//!   `value` is the raw card JSON string.
//! - expiry: `datum.ttl && datum.ttl < Date.now()` (no ttl ⇒ never expires).

use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

/// node-persist's hashKey: sha256 hex of the key string.
pub fn hash_key(key: &str) -> String {
    let digest = Sha256::digest(key.as_bytes());
    let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
    hex
}

/// Format milliseconds like Node's `stat.mtimeMs` (JS double shortest
/// round-trip repr). Arithmetic mirrors Node's `fromPrimitives`:
/// `seconds * 1000 + nanos / 1e6` in f64.
pub fn format_mtime_ms(secs: u64, nanos: u32) -> String {
    let ms = (secs as f64) * 1000.0 + (nanos as f64) / 1_000_000.0;
    format_double(ms)
}

/// JS-like shortest round-trip formatting of a double. Rust's f64 Display
/// already produces the shortest string that round-trips — same guarantee as
/// JS Number.prototype.toString.
fn format_double(v: f64) -> String {
    if v == v.trunc() && v.abs() < 1e21 {
        // Integral values print without a decimal point in both languages.
        format!("{}", v as i64)
    } else {
        format!("{}", v)
    }
}

/// Mirror of Node `path.join(...)` for the cache key's path component:
/// components joined with the OS separator; a leading `./` in the data root
/// is normalized away exactly like path.join does.
pub fn node_path_join(components: &[&str]) -> String {
    let sep = std::path::MAIN_SEPARATOR;
    let mut parts: Vec<&str> = Vec::with_capacity(components.len());
    for (i, c) in components.iter().enumerate() {
        let mut c = *c;
        // path.join drops a leading './' only where it would be the prefix
        // (it appears as './data'); inner './' segments are unlikely here.
        if i == 0 {
            if let Some(stripped) = c.strip_prefix(format!(".{sep}").as_str()) {
                c = stripped;
            } else if c == "." {
                c = "";
            }
        }
        if c.is_empty() {
            continue;
        }
        parts.push(c);
    }
    parts.join(&sep.to_string())
}

/// Build the cache key for a character file, mirroring
/// `getCacheKey(inputFile)`: `path.join(charactersDir, item) + "-" + mtimeMs`.
pub fn cache_key(data_root: &Path, handle: &str, item: &str, mtime_ms: &str) -> String {
    let characters_dir = node_path_join(&[&data_root.to_string_lossy(), handle, "characters"]);
    format!(
        "{}{}{}-{}",
        characters_dir,
        std::path::MAIN_SEPARATOR,
        item,
        mtime_ms
    )
}

/// A node-persist cache directory (e.g. `data/_cache/characters`).
#[derive(Clone)]
pub struct DiskCache {
    dir: PathBuf,
}

impl DiskCache {
    pub fn new(data_root: &Path) -> Self {
        Self {
            dir: data_root.join("_cache").join("characters"),
        }
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Read a cached value: returns the card JSON string on a hit.
    /// Corrupt/foreign files are treated as a miss (never fatal), mirroring
    /// node-persist's forgiveParseErrors behavior.
    pub fn get(&self, key: &str) -> Option<String> {
        let path = self.dir.join(hash_key(key));
        let raw = std::fs::read_to_string(path).ok()?;
        let datum: serde_json::Value = serde_json::from_str(&raw).ok()?;
        // Expiry check (node-persist isExpired): no ttl ⇒ valid.
        if let Some(ttl) = datum.get("ttl").and_then(|t| t.as_f64()) {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .ok()?
                .as_millis() as f64;
            if ttl < now {
                return None;
            }
        }
        match datum.get("value")? {
            serde_json::Value::String(s) => Some(s.clone()),
            _ => None,
        }
    }

    /// Write a cache entry: `{"key": ..., "value": ...}` (key before value,
    /// ttl omitted — byte-compatible with Node's JSON.stringify output).
    /// Uses temp file + rename so a concurrent Node read never sees a partial
    /// write.
    pub fn put(&self, key: &str, value: &str) -> std::io::Result<()> {
        std::fs::create_dir_all(&self.dir)?;
        let datum = serde_json::json!({ "key": key, "value": value });
        let bytes = serde_json::to_string(&datum).unwrap_or_default();

        let final_path = self.dir.join(hash_key(key));
        let tmp_path = final_path.with_extension("tmp");
        std::fs::write(&tmp_path, bytes)?;
        std::fs::rename(&tmp_path, &final_path)?;
        Ok(())
    }

    /// Remove entries whose cached key no longer matches disk state — the
    /// mirror of Node's `DiskCache.verify`: parses each datum's `key`, splits
    /// `path-mtimeMs` on the last `-`, and deletes the entry when the source
    /// file is gone or its current mtime differs.
    pub fn prune(&self) -> usize {
        let mut removed = 0;
        let Ok(entries) = std::fs::read_dir(&self.dir) else {
            return 0;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let Ok(raw) = std::fs::read_to_string(&path) else {
                continue;
            };
            let Ok(datum) = serde_json::from_str::<serde_json::Value>(&raw) else {
                continue;
            };
            let Some(key) = datum.get("key").and_then(|k| k.as_str()) else {
                continue;
            };
            let Some((path_part, mtime_part)) = key.rsplit_once('-') else {
                continue;
            };
            let source = PathBuf::from(path_part);
            let still_valid = std::fs::metadata(&source)
                .ok()
                .map(|meta| {
                    let (secs, nanos) = mtime_secs_nanos(&meta);
                    format_mtime_ms(secs, nanos) == mtime_part
                })
                .unwrap_or(false);
            if !still_valid {
                let _ = std::fs::remove_file(&path);
                removed += 1;
            }
        }
        removed
    }
}

/// (secs, nanos) of a file's mtime, mirroring Node's stat mtimeMs source.
pub fn mtime_secs_nanos(meta: &std::fs::Metadata) -> (u64, u32) {
    meta.modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| (d.as_secs(), d.subsec_nanos()))
        .unwrap_or((0, 0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_matches_node_vector() {
        // Real key captured from a Node-produced cache file in data/_cache/characters:
        // file content's `key` field, whose sha256 equals the stored filename.
        let key = "data\\default-user\\characters\\default_Seraphina.png-1738053032117.1682";
        assert_eq!(
            hash_key(key),
            "dafca6d05d8c047cade7219732018e5065174813b9806fc2ea3179721210d033"
        );
    }

    #[test]
    fn mtime_format_matches_node() {
        // 1738053032 secs + 117168200 ns (= 117.1682 ms)
        // → Node prints "1738053032117.1682"
        assert_eq!(format_mtime_ms(1738053032, 117168200), "1738053032117.1682");
        // Integral millis print without decimals (JS behavior)
        assert_eq!(format_mtime_ms(1738053032, 0), "1738053032000");
    }

    #[cfg(windows)]
    #[test]
    fn cache_key_matches_node_join_windows() {
        let key = cache_key(
            Path::new("data"),
            "default-user",
            "default_Seraphina.png",
            "1738053032117.1682",
        );
        assert_eq!(
            key,
            "data\\default-user\\characters\\default_Seraphina.png-1738053032117.1682"
        );
    }

    #[cfg(unix)]
    #[test]
    fn cache_key_matches_node_join_unix() {
        let key = cache_key(
            Path::new("data"),
            "default-user",
            "default_Seraphina.png",
            "1738053032117.1682",
        );
        assert_eq!(
            key,
            "data/default-user/characters/default_Seraphina.png-1738053032117.1682"
        );
    }

    #[test]
    fn datum_file_format_matches_node() {
        let dir = tempfile::tempdir().unwrap();
        let cache = DiskCache {
            dir: dir.path().to_path_buf(),
        };
        cache.put("k", "{\"a\":1}").unwrap();
        let file = cache.dir().join(hash_key("k"));
        let raw = std::fs::read_to_string(file).unwrap();
        // key first, value second, no ttl — identical to JSON.stringify({key, value})
        assert_eq!(raw, r#"{"key":"k","value":"{\"a\":1}"}"#);
        // And it reads back
        assert_eq!(cache.get("k").unwrap(), "{\"a\":1}");
    }

    #[test]
    fn corrupt_file_is_a_miss() {
        let dir = tempfile::tempdir().unwrap();
        let cache = DiskCache {
            dir: dir.path().to_path_buf(),
        };
        std::fs::write(cache.dir().join(hash_key("bad")), "not json{").unwrap();
        assert!(cache.get("bad").is_none());
    }

    #[test]
    fn prune_removes_stale_and_keeps_valid() {
        let dir = tempfile::tempdir().unwrap();
        let chars = dir.path().join("data").join("u").join("characters");
        std::fs::create_dir_all(&chars).unwrap();
        let cache = DiskCache {
            dir: dir.path().join("_cache").join("characters"),
        };

        // Write a card png so its mtime is stable, then derive the key from it
        let card_path = chars.join("a.png");
        std::fs::write(&card_path, b"png").unwrap();
        let meta = std::fs::metadata(&card_path).unwrap();
        let (secs, nanos) = mtime_secs_nanos(&meta);
        let mtime = format_mtime_ms(secs, nanos);
        let valid_key = format!(
            "{}{}{}",
            chars.to_string_lossy(),
            std::path::MAIN_SEPARATOR,
            format!("a.png-{}", mtime)
        );

        cache.put(&valid_key, "{}").unwrap();
        cache
            .put("data\\gone\\characters\\deleted.png-123", "{}")
            .unwrap();

        let removed = cache.prune();
        assert_eq!(removed, 1);
        assert!(cache.get(&valid_key).is_some());
        assert!(cache
            .get("data\\gone\\characters\\deleted.png-123")
            .is_none());
    }
}
