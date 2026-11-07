//! Keep-alive behaviour: an idle upstream must keep producing bytes that are
//! valid for whichever protocol the provider picked, and must never corrupt
//! the payload.

use std::future::{poll_fn, Future};
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Duration;

use bytes::Bytes;
use st_gateway::chat_completions::Protocol;
use st_gateway::heartbeat::HeartbeatBody;
use tokio::sync::mpsc;

/// A body fed by an mpsc channel, so the test decides when frames appear.
struct ChannelBody {
    rx: mpsc::Receiver<Bytes>,
}

impl http_body::Body for ChannelBody {
    type Data = Bytes;
    type Error = core::convert::Infallible;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<http_body::Frame<Self::Data>, Self::Error>>> {
        match self.rx.poll_recv(cx) {
            Poll::Ready(Some(b)) => Poll::Ready(Some(Ok(http_body::Frame::data(b)))),
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
    }
}

fn channel_body() -> (mpsc::Sender<Bytes>, ChannelBody) {
    let (tx, rx) = mpsc::channel::<Bytes>(8);
    (tx, ChannelBody { rx })
}

/// Collect everything the wrapped body yields until it ends or `budget`
/// elapses (returns the string collected so far on timeout).
async fn drain(mut body: HeartbeatBody, budget: Duration) -> String {
    let mut out = String::new();
    let deadline = tokio::time::Instant::now() + budget;
    let fut = body.poll_next_frame_fn();
    let mut fut = Box::pin(fut);
    loop {
        match tokio::time::timeout_at(deadline, poll_fn(|cx| fut.as_mut().poll(cx))).await {
            Ok(Some(Ok(frame))) => {
                if let Some(data) = frame.data_ref() {
                    out.push_str(&String::from_utf8_lossy(data));
                }
                // Re-borrow after the frame was consumed.
                fut = Box::pin(body.poll_next_frame_fn());
            }
            Ok(Some(Err(_))) | Ok(None) => break,
            Err(_) => break,
        }
    }
    out
}

/// Helper: a `Future` resolving to the next frame, borrowing the body.
trait NextFrame<'a> {
    fn poll_next_frame_fn(&'a mut self) -> NextFrameFut<'a>;
}

struct NextFrameFut<'a> {
    body: &'a mut HeartbeatBody,
}

impl Future for NextFrameFut<'_> {
    type Output = Option<Result<http_body::Frame<Bytes>, core::convert::Infallible>>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let body = &mut *self.body;
        Pin::new(body).poll_next_frame(cx)
    }
}

impl<'a> NextFrame<'a> for HeartbeatBody {
    fn poll_next_frame_fn(&'a mut self) -> NextFrameFut<'a> {
        NextFrameFut { body: self }
    }
}

#[tokio::test]
async fn sse_idle_emits_comment_lines() {
    let (tx, inner) = channel_body();
    let body = HeartbeatBody::new(inner, Protocol::Sse, Duration::from_millis(20));
    let collect = tokio::spawn(drain(body, Duration::from_millis(120)));
    tokio::time::sleep(Duration::from_millis(100)).await;
    tx.send(Bytes::from_static(b"data: hi\n\n")).await.unwrap();
    drop(tx);
    let out = collect.await.unwrap();
    assert!(
        out.contains(": keep-alive\n\n"),
        "expected SSE comment heartbeats, got {out:?}"
    );
    assert!(out.contains("data: hi\n\n"));
}

#[tokio::test]
async fn ndjson_idle_emits_blank_lines_only() {
    let (tx, inner) = channel_body();
    let body = HeartbeatBody::new(inner, Protocol::Ndjson, Duration::from_millis(20));
    let collect = tokio::spawn(drain(body, Duration::from_millis(120)));
    tokio::time::sleep(Duration::from_millis(100)).await;
    tx.send(Bytes::from_static(b"{\"a\":1}\n")).await.unwrap();
    drop(tx);
    let out = collect.await.unwrap();
    assert!(
        out.matches('\n').count() > 1,
        "no heartbeat captured: {out:?}"
    );
    for line in out.lines() {
        if line.is_empty() {
            continue;
        }
        serde_json::from_str::<serde_json::Value>(line).expect("line stays valid JSON");
    }
    assert!(
        !out.contains("keep-alive"),
        "NDJSON must not receive SSE comments"
    );
}

#[tokio::test]
async fn json_prefix_keeps_body_parseable() {
    let (tx, inner) = channel_body();
    // Long interval: the leading space must be emitted eagerly, not on the
    // first idle tick (a provider may answer immediately).
    let body = HeartbeatBody::new(inner, Protocol::Json, Duration::from_millis(500));
    let collect = tokio::spawn(drain(body, Duration::from_millis(80)));
    tokio::time::sleep(Duration::from_millis(10)).await;
    tx.send(Bytes::from_static(b"{\"ok\":true}")).await.unwrap();
    drop(tx);
    let out = collect.await.unwrap();
    let parsed: serde_json::Value =
        serde_json::from_str(&out).expect("leading whitespace stays valid JSON");
    assert_eq!(parsed["ok"], true);
    assert!(out.starts_with(' '), "expected lazy leading-space prefix");
}

#[tokio::test]
async fn json_split_body_stays_valid() {
    let (tx, inner) = channel_body();
    let body = HeartbeatBody::new(inner, Protocol::Json, Duration::from_millis(20));
    let collect = tokio::spawn(drain(body, Duration::from_millis(120)));
    tx.send(Bytes::from_static(b"{\"a\":")).await.unwrap();
    tokio::time::sleep(Duration::from_millis(50)).await;
    tx.send(Bytes::from_static(b"1}")).await.unwrap();
    drop(tx);
    let out = collect.await.unwrap();
    let parsed: serde_json::Value =
        serde_json::from_str(&out).expect("interstitial whitespace stays valid JSON");
    assert_eq!(parsed["a"], 1);
}

#[tokio::test]
async fn unknown_protocol_uses_safest_token() {
    let (tx, inner) = channel_body();
    let body = HeartbeatBody::new(inner, Protocol::Unknown, Duration::from_millis(20));
    let _keep_alive = tx;
    let out = drain(body, Duration::from_millis(80)).await;
    assert!(out.contains('\n'));
    assert!(!out.contains(':'));
}

#[tokio::test]
async fn large_payload_passes_through_intact() {
    // 40 MiB, one MiB at a time: the wrapper must not accumulate anything.
    let (tx, inner) = channel_body();
    let wrapped = HeartbeatBody::new(inner, Protocol::Sse, Duration::from_secs(30));
    let chunk = Bytes::from(vec![b'x'; 1024 * 1024]);
    let producer = tokio::spawn(async move {
        for _ in 0..40 {
            if tx.send(chunk.clone()).await.is_err() {
                break;
            }
        }
    });
    let out = drain(wrapped, Duration::from_secs(20)).await;
    producer.await.unwrap();
    // One priming token is prepended; everything after it is the payload.
    let payload = out
        .strip_prefix(": keep-alive\n\n")
        .expect("priming token first");
    assert_eq!(
        payload.len(),
        40 * 1024 * 1024,
        "payload must pass through intact"
    );
}

#[tokio::test]
async fn first_frame_is_priming_token_even_when_upstream_is_silent() {
    // CF punishes a 200 whose body has not started; the very first frame must
    // be the priming token, not something the provider sent.
    for (protocol, expected) in [
        (Protocol::Sse, ": keep-alive\n\n"),
        (Protocol::Ndjson, "\n"),
        (Protocol::Json, " "),
        (Protocol::Unknown, "\n"),
    ] {
        let (tx, inner) = channel_body();
        // Long interval: nothing may come from the timer in this window.
        let mut body = HeartbeatBody::new(inner, protocol, Duration::from_secs(60));
        let first = tokio::time::timeout(
            Duration::from_millis(100),
            poll_fn(|cx| Pin::new(&mut body).poll_next_frame(cx)),
        )
        .await
        .expect("priming frame must not wait for upstream data")
        .expect("body must yield a frame")
        .expect("infallible");
        let data = first.into_data().expect("data frame");
        assert_eq!(
            &data[..],
            expected.as_bytes(),
            "protocol {protocol:?} priming token"
        );
        let _keep_alive = tx;
    }
}

#[tokio::test]
async fn json_prefix_is_not_duplicated_by_priming() {
    let (tx, inner) = channel_body();
    let body = HeartbeatBody::new(inner, Protocol::Json, Duration::from_secs(60));
    let collect = tokio::spawn(drain(body, Duration::from_millis(80)));
    tx.send(Bytes::from_static(b"{\"ok\":true}")).await.unwrap();
    drop(tx);
    let out = collect.await.unwrap();
    assert_eq!(out, " {\"ok\":true}");
    let parsed: serde_json::Value = serde_json::from_str(&out).unwrap();
    assert_eq!(parsed["ok"], true);
}
