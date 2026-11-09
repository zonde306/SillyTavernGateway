//! Keep-alive behaviour: an idle upstream must keep producing bytes that are
//! valid for whichever protocol the provider picked, and must never corrupt
//! the payload.

use http_body::Body as _;
use std::future::{poll_fn, Future};
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Duration;

use bytes::Bytes;
use st_gateway::chat_completions::Protocol;
use st_gateway::heartbeat::{HeartbeatBody, NormalizeBody, PendingBody, UpstreamResult};
use tokio::sync::mpsc;

/// A body fed by an mpsc channel, so the test decides when frames appear.
#[derive(Debug)]
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

// --- PendingBody: answering before the provider does ------------------------
//
// A provider that is thinking sends no status line at all, so a gateway that
// awaits it reproduces the silence Cloudflare turns into a 524. `PendingBody`
// commits to the response immediately and keeps the client fed while the
// upstream call is still in flight.

/// Collect frames from any body until it ends or `budget` elapses.
async fn drain_any<B>(mut body: B, budget: Duration) -> String
where
    B: http_body::Body<Data = Bytes, Error = core::convert::Infallible> + Unpin,
{
    let mut out = String::new();
    let deadline = tokio::time::Instant::now() + budget;
    loop {
        let next = poll_fn(|cx| Pin::new(&mut body).poll_frame(cx));
        match tokio::time::timeout_at(deadline, next).await {
            Ok(Some(Ok(frame))) => {
                if let Some(data) = frame.data_ref() {
                    out.push_str(&String::from_utf8_lossy(data));
                }
            }
            Ok(Some(Err(_))) | Ok(None) => break,
            Err(_) => break,
        }
    }
    out
}

/// The whole point: bytes must flow while the upstream has not answered yet.
#[tokio::test]
async fn keep_alives_flow_before_the_upstream_answers() {
    let (tx, rx) = tokio::sync::oneshot::channel::<UpstreamResult>();
    let body = PendingBody::new(rx, Duration::from_millis(50));

    let collect = tokio::spawn(drain_any(body, Duration::from_millis(280)));
    tokio::time::sleep(Duration::from_millis(200)).await;
    drop(tx); // never answered: the client must still have been fed
    let out = collect.await.unwrap();

    // A blank line is safe for SSE, NDJSON and JSON alike; the concrete
    // protocol is not known until the upstream answers.
    let count = out.matches('\n').count();
    assert!(
        count >= 5,
        "expected the priming byte plus ~4 periodic ones, got {count}: {out:?}"
    );
}

/// Once the response arrives its body takes over unchanged.
#[tokio::test]
async fn upstream_body_takes_over_once_it_arrives() {
    let (tx, rx) = tokio::sync::oneshot::channel::<UpstreamResult<ChannelBody>>();
    let body = PendingBody::new(rx, Duration::from_millis(1000));

    let (body_tx, inner) = channel_body();
    let response = axum::http::Response::builder()
        .status(200)
        .header("content-type", "text/event-stream")
        .body(inner)
        .unwrap();
    let collect = tokio::spawn(drain_any(body, Duration::from_millis(250)));
    tokio::time::sleep(Duration::from_millis(30)).await;
    tx.send(Ok(response)).unwrap();
    tokio::time::sleep(Duration::from_millis(30)).await;
    body_tx
        .send(Bytes::from_static(b"data: hi\n\n"))
        .await
        .unwrap();
    drop(body_tx);
    let out = collect.await.unwrap();

    assert!(
        out.contains("data: hi\n\n"),
        "upstream bytes must pass through: {out:?}"
    );
    assert!(
        out.starts_with('\n'),
        "the priming byte still comes first: {out:?}"
    );
}

/// A status code can no longer be sent once the 200 is on the wire, so a
/// failure has to travel as an SSE event the frontend already reports.
#[tokio::test]
async fn upstream_failure_becomes_a_streaming_error_event() {
    let (tx, rx) = tokio::sync::oneshot::channel::<UpstreamResult>();
    let body = PendingBody::new(rx, Duration::from_millis(1000));
    let collect = tokio::spawn(drain_any(body, Duration::from_millis(200)));
    tokio::time::sleep(Duration::from_millis(20)).await;
    tx.send(Err(
        r#"{"error":{"message":"insufficient quota"}}"#.to_string()
    ))
    .unwrap();
    let out = collect.await.unwrap();

    let data = out
        .lines()
        .find(|line| line.starts_with("data: "))
        .expect("an error event is emitted");
    let parsed: serde_json::Value =
        serde_json::from_str(data.trim_start_matches("data: ")).expect("valid JSON");
    // The provider's own JSON is forwarded verbatim, so the frontend's
    // `tryParseStreamingError` reports the real reason instead of a status.
    assert_eq!(parsed["error"]["message"], "insufficient quota");
}

/// A dangling sender must not hang the client forever.
#[tokio::test]
async fn dropped_upstream_sender_ends_the_stream() {
    let (tx, rx) = tokio::sync::oneshot::channel::<UpstreamResult>();
    let body = PendingBody::new(rx, Duration::from_millis(1000));
    drop(tx);
    let out = drain_any(body, Duration::from_millis(500)).await;
    assert!(
        out.contains("data: "),
        "the client is told the request failed: {out:?}"
    );
}

// --- cadence: wall-clock, not idle -----------------------------------------

/// Sharing a timer with the upstream was wrong: a provider that emits a cheap
/// comment every few seconds (Google sends `: PING`) kept resetting it, so our
/// own keep-alive never fired — and when a hop buffers those comments the
/// client sees nothing at all until the final payload.
#[tokio::test]
async fn chatty_upstream_does_not_suppress_our_heartbeats() {
    let (tx, inner) = channel_body();
    let body = HeartbeatBody::new(inner, Protocol::Sse, Duration::from_millis(50));
    let collect = tokio::spawn(drain(body, Duration::from_millis(400)));
    let chatter = tokio::spawn(async move {
        for _ in 0..10 {
            if tx.send(Bytes::from_static(b": PING\n\n")).await.is_err() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        drop(tx);
    });
    let out = collect.await.unwrap();
    chatter.await.unwrap();

    let upstream_pings = out.matches(": PING").count();
    let ours = out.matches(": keep-alive").count();
    // Priming emits one immediately; the periodic ticks must add more.
    assert!(upstream_pings >= 5, "upstream chatter missing: {out:?}");
    assert!(
        ours >= 3,
        "chatty upstream suppressed our keep-alives (got {ours}, expected the \
         wall-clock cadence to keep firing): {out:?}"
    );
}

/// A zero interval means "disabled" and must not busy-loop.
#[tokio::test]
async fn zero_interval_emits_no_periodic_heartbeats() {
    let (tx, inner) = channel_body();
    let body = HeartbeatBody::new(inner, Protocol::Sse, Duration::ZERO);
    let collect = tokio::spawn(drain(body, Duration::from_millis(150)));
    tokio::time::sleep(Duration::from_millis(100)).await;
    tx.send(Bytes::from_static(b"data: hi\n\n")).await.unwrap();
    drop(tx);
    let out = collect.await.unwrap();
    // Only the priming frame, never a stream of back-to-back keep-alives.
    assert_eq!(
        out.matches(": keep-alive").count(),
        1,
        "heartbeat_secs: 0 must disable periodic heartbeats: {out:?}"
    );
    assert!(
        out.contains("data: hi\n\n"),
        "upstream bytes must pass through: {out:?}"
    );
}

/// A non-stream JSON body is a single document. Whitespace is only safe before
/// it starts; injecting it mid-document could land inside a string literal.
#[tokio::test]
async fn json_body_gets_no_interstitial_heartbeat() {
    let (tx, inner) = channel_body();
    let body = HeartbeatBody::new(inner, Protocol::Json, Duration::from_millis(20));
    let collect = tokio::spawn(drain(body, Duration::from_millis(250)));
    tx.send(Bytes::from_static(b"{\"a\":\"x")).await.unwrap();
    tokio::time::sleep(Duration::from_millis(120)).await;
    tx.send(Bytes::from_static(b"yz\"}")).await.unwrap();
    drop(tx);
    let out = collect.await.unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&out).expect("JSON must stay parseable");
    assert_eq!(parsed["a"], "xyz");
}

// --- termination: an error must never be followed by more bytes -------------

/// A body that always yields data and never ends.
struct EndlessBody {
    chunk: Bytes,
}

impl http_body::Body for EndlessBody {
    type Data = Bytes;
    type Error = core::convert::Infallible;

    fn poll_frame(
        self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
    ) -> Poll<Option<Result<http_body::Frame<Self::Data>, Self::Error>>> {
        Poll::Ready(Some(Ok(http_body::Frame::data(self.chunk.clone()))))
    }
}

/// A body that fails immediately.
struct FailingBody {
    done: bool,
}

impl http_body::Body for FailingBody {
    type Data = Bytes;
    type Error = &'static str;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
    ) -> Poll<Option<Result<http_body::Frame<Self::Data>, Self::Error>>> {
        if self.done {
            return Poll::Ready(None);
        }
        self.done = true;
        Poll::Ready(Some(Err("connection reset")))
    }
}

/// The user-visible bug: keep-alives kept arriving after the error event, so
/// the request looked stuck instead of finished. Whatever else happens, an
/// error frame must be the LAST frame and nothing may follow it.
#[tokio::test]
async fn nothing_is_sent_after_an_upstream_stream_error() {
    let body = HeartbeatBody::new(
        NormalizeBody::new(FailingBody { done: false }),
        Protocol::Sse,
        Duration::from_millis(10),
    );
    let out = drain_any(body, Duration::from_millis(300)).await;

    let error_at = out
        .find("\"error\"")
        .unwrap_or_else(|| panic!("no error event in {out:?}"));
    let after = &out[error_at..];
    let tail = after
        .split_once("\n\n")
        .map(|(_, rest)| rest)
        .unwrap_or_default();
    assert!(
        tail.trim().is_empty(),
        "bytes were sent after the error event: {tail:?} (full output {out:?})"
    );
}

/// Heartbeats are for waiting only. Once the upstream is producing data
/// continuously there is nothing to keep alive, and the timer must not
/// interleave extra bytes into a live stream.
#[tokio::test]
async fn a_busy_upstream_gets_no_heartbeats_between_frames() {
    let body = HeartbeatBody::new(
        EndlessBody {
            chunk: Bytes::from_static(b"data: x\n\n"),
        },
        Protocol::Sse,
        Duration::from_millis(500),
    );
    // Drain a fixed number of frames rather than by time; the body never ends.
    let mut body = Box::pin(body);
    let mut out = String::new();
    for _ in 0..6 {
        let frame = poll_fn(|cx| body.as_mut().poll_frame(cx)).await;
        if let Some(Ok(frame)) = frame {
            if let Some(data) = frame.data_ref() {
                out.push_str(&String::from_utf8_lossy(data));
            }
        }
    }
    // The priming frame is expected (it proves liveness before the first byte
    // of payload); what must not appear is a *periodic* heartbeat, which would
    // mean the timer interleaved itself into a stream that was never idle.
    assert_eq!(
        out.matches(": keep-alive").count(),
        1,
        "a continuously busy upstream must not receive periodic heartbeats: {out:?}"
    );
    // Six polls: one priming frame plus five upstream frames.
    assert_eq!(out.matches("data: x").count(), 5, "frames must pass intact");
}
