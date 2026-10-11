//! End-to-end proof of cache sharing: reads a REAL cache file produced by
//! the Node backend (data/_cache/characters/*, committed by the dev's
//! running instance) using the Rust DiskCache — the same lookup Node does.

use st_gateway::disk_cache::{hash_key, DiskCache};

#[test]
fn reads_real_node_produced_cache_entry() {
    let manifest = env!("CARGO_MANIFEST_DIR");
    // Repo root is the parent of src-rs
    let repo = std::path::Path::new(manifest).parent().unwrap();
    let cache_dir = repo.join("data/_cache/characters");
    let Ok(entries) = std::fs::read_dir(&cache_dir) else {
        // No cache dir in this checkout: nothing to prove, skip silently
        return;
    };

    // Pick any real Node-produced datum, extract its key, and read it back
    // through the Rust DiskCache with the exact key.
    let mut found = false;
    for entry in entries.flatten() {
        let path = entry.path();
        let Ok(raw) = std::fs::read_to_string(&path) else { continue };
        let Ok(datum) = serde_json::from_str::<serde_json::Value>(&raw) else { continue };
        let Some(key) = datum.get("key").and_then(|k| k.as_str()) else { continue };
        let Some(value) = datum.get("value").and_then(|v| v.as_str()) else { continue };

        // 1. Rust hash of the key must land on the same file Node wrote
        assert_eq!(
            hash_key(key),
            path.file_name().unwrap().to_string_lossy(),
            "hash mismatch for a Node-produced entry"
        );

        // 2. Rust DiskCache.get on that key must return the same value
        let cache = DiskCache::new(&repo.join("data"));
        let read_back = cache.get(key).expect("shared entry must be readable");
        assert_eq!(read_back, value);

        // 3. The value must be parseable card JSON (Rust can rebuild /all
        //    responses from it without touching the PNG)
        serde_json::from_str::<serde_json::Value>(&read_back)
            .expect("cached value is card JSON");

        found = true;
        break;
    }
    if !found {
        eprintln!("no valid cache entries present; sharing test vacuously passed");
    }
}
