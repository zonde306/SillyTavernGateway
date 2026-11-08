//! Atomic file writes with UNIQUE temp names.
//!
//! Every writer in this gateway replaces a file that the frontend may be
//! reading concurrently (character cards, thumbnails, disk-cache entries).
//! A fixed temp name (`<file>.tmp`) is a data-corruption bug, not a style
//! issue: two concurrent writers truncate and interleave each other's bytes
//! in the same temp file, and whichever rename lands second fails with ENOENT
//! (the other already consumed the temp). Upstream `write-file-atomic` avoids
//! this with `getTmpname()` = name + murmur(pid, threadId, ++invocations).
//!
//! We mirror that with a process-wide counter: the temp lives in the
//! destination's own directory (same filesystem, so rename is atomic) and is
//! never shared between writers.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Unique temp path in `dst`'s directory.
pub fn temp_path(dst: &Path) -> PathBuf {
    let n = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let name = dst
        .file_name()
        .map(|f| f.to_string_lossy().to_string())
        .unwrap_or_else(|| "tmp".to_string());
    let tmp = format!(".{name}.{}.{}.tmp", std::process::id(), n);
    match dst.parent() {
        Some(dir) if !dir.as_os_str().is_empty() => dir.join(tmp),
        _ => PathBuf::from(tmp),
    }
}

/// Rename with a few short retries. On Windows a reader that opened the file
/// without FILE_SHARE_DELETE (antivirus, search indexer, a concurrent reader)
/// makes the rename fail transiently with ACCESS_DENIED/SHARING_VIOLATION.
/// The temp is removed on final failure so no garbage accumulates.
pub fn rename_with_retry(from: &Path, to: &Path) -> std::io::Result<()> {
    let mut last = None;
    for attempt in 0..5u64 {
        match std::fs::rename(from, to) {
            Ok(()) => return Ok(()),
            Err(e) => {
                last = Some(e);
                std::thread::sleep(std::time::Duration::from_millis(10 * (attempt + 1)));
            }
        }
    }
    let err = last.expect("loop runs at least once");
    let _ = std::fs::remove_file(from);
    Err(err)
}

/// Remove a file with a few short retries. Windows refuses deletion while any
/// other handle is open (antivirus, indexer, an in-flight read), and that
/// failure is transient. Returns the last error if all attempts fail.
pub fn remove_with_retry(path: &Path) -> std::io::Result<()> {
    let mut last = None;
    for attempt in 0..5u64 {
        match std::fs::remove_file(path) {
            Ok(()) => return Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(e) => {
                last = Some(e);
                std::thread::sleep(std::time::Duration::from_millis(10 * (attempt + 1)));
            }
        }
    }
    Err(last.expect("loop runs at least once"))
}

/// Write `data` to `path` atomically: unique temp file, then rename. Readers
/// observe either the previous complete file or the new complete file.
pub fn write(path: &Path, data: &[u8]) -> std::io::Result<()> {
    write_with(path, |out| out.write_all(data))
}

/// Same, but the caller streams the content (used for chunk surgery so large
/// images are never buffered twice).
pub fn write_with<F>(path: &Path, fill: F) -> std::io::Result<()>
where
    F: FnOnce(&mut std::io::BufWriter<std::fs::File>) -> std::io::Result<()>,
{
    let tmp = temp_path(path);
    let written = (|| -> std::io::Result<()> {
        let mut out = std::io::BufWriter::new(std::fs::File::create(&tmp)?);
        fill(&mut out)?;
        out.flush()?;
        Ok(())
    })();
    match written {
        Ok(()) => rename_with_retry(&tmp, path),
        Err(e) => {
            let _ = std::fs::remove_file(&tmp);
            Err(e)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temp_paths_are_unique_and_sibling() {
        let dir = std::path::Path::new("/tmp/chars");
        let a = temp_path(&dir.join("Hero.png"));
        let b = temp_path(&dir.join("Hero.png"));
        assert_ne!(a, b, "concurrent writers must not share a temp file");
        assert_eq!(a.parent(), Some(dir));
        assert!(a.file_name().unwrap().to_string_lossy().ends_with(".tmp"));
    }

    #[test]
    fn no_temp_file_survives_success() {
        let d = tempfile::tempdir().unwrap();
        let p = d.path().join("x.png");
        std::fs::write(&p, b"old").unwrap();
        write(&p, b"new").unwrap();
        assert_eq!(std::fs::read(&p).unwrap(), b"new");
        let leftovers: Vec<_> = std::fs::read_dir(d.path())
            .unwrap()
            .flatten()
            .filter(|e| e.file_name().to_string_lossy().ends_with(".tmp"))
            .collect();
        assert!(leftovers.is_empty(), "temp files must be renamed away");
    }

    #[test]
    fn remove_with_retry_is_ok_for_missing_files() {
        let d = tempfile::tempdir().unwrap();
        assert!(remove_with_retry(&d.path().join("nope.png")).is_ok());
    }

    #[test]
    fn failure_leaves_original_untouched() {
        let d = tempfile::tempdir().unwrap();
        let p = d.path().join("x.png");
        std::fs::write(&p, b"original").unwrap();
        let err = write_with(&p, |_| Err(std::io::Error::other("boom")));
        assert!(err.is_err());
        assert_eq!(std::fs::read(&p).unwrap(), b"original");
    }
}
