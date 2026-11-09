//! Idle keep-alive for long-lived upstream responses.
//!
//! Chat completions can think for minutes before emitting the first token.
//! Reverse proxies and CDNs (Cloudflare in particular) drop connections that
//! stay silent, and `proxy_read_timeout` in nginx does the same. Upstream
//! SillyTavern sends nothing, so the fix has to happen in the gateway.
//!
//! What can be sent depends on the protocol the upstream chose, and the
//! content type is only known once the 200 header arrives — which is exactly
//! when the silence starts. So:
//!
//! | response                      | heartbeat                          |
//! |-------------------------------|------------------------------------|
//! | `text/event-stream` (SSE)     | `: keep-alive\n\n` comment line    |
//! | NDJSON / `application/jsonl`  | `\n` (an empty line, no record)     |
//! | JSON (non-stream)             | leading ` ` instead of a heartbeat  |
//!
//! A bare `\n` is a no-op for every line/TEXT_EVENT_STREAM parser: `EventSource`
//! ignores blank lines, and JSON parsers ignore leading/interstitial
//! whitespace (upstream continues to `JSON.parse` the whole text).
//!
//! Cadence is WALL-CLOCK, not "idle for N seconds". Sharing a timer with the
//! upstream was wrong: a provider that emits a cheap comment every few seconds
//! (Google sends `: PING`) kept resetting our timer, so we never sent anything
//! of our own — and if a hop in between buffers those comments, the client sees
//! nothing at all until the final payload. The client only needs *something* on
//! a predictable schedule, so a heartbeat goes out whenever `interval` has
//! elapsed since the last one, no matter how chatty the upstream is.

use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Duration;

use bytes::Bytes;
use http_body::Frame;
use http_body_util::BodyExt;

use crate::chat_completions::Protocol;

/// The gateway default: 15 s of silence before a heartbeat is emitted.
pub const DEFAULT_HEARTBEAT_SECS: u64 = 15;

/// Wrap an upstream body so it keeps emitting heartbeats while idle.
pub struct HeartbeatBody {
    inner: Pin<Box<dyn http_body::Body<Data = Bytes, Error = core::convert::Infallible> + Send>>,
    protocol: Protocol,
    interval: Duration,
    /// Wall-clock heartbeat timer. `None` when heartbeats are disabled
    /// (`gateway.heartbeat_secs: 0`).
    timer: Option<Pin<Box<tokio::time::Sleep>>>,
    /// Whether the first data byte has been seen. Only used to decide the JSON
    /// prefix; the heartbeat cadence no longer depends on it.
    started: bool,
    /// Whether `Protocol::Json` already emitted its leading space.
    json_prefix_sent: bool,
    /// One immediately-flushed frame queued at construction time.
    ///
    /// Cloudflare and similar CDNs punish a request whose response has not
    /// started flowing, so a protocol-safe token goes out right after the 200
    /// header; the idle timer only covers what happens after that.
    priming: Option<Bytes>,

    done: bool,
}

impl HeartbeatBody {
    pub fn new<B>(inner: B, protocol: Protocol, interval: Duration) -> Self
    where
        B: http_body::Body<Data = Bytes, Error = core::convert::Infallible> + Send + 'static,
    {
        let mut body = Self {
            inner: Box::pin(inner),
            protocol,
            interval,
            // A zero interval means "disabled". Without this guard it
            // degrades into a busy loop of back-to-back heartbeats.
            timer: (interval > Duration::ZERO).then(|| Box::pin(tokio::time::sleep(interval))),
            started: false,
            json_prefix_sent: false,
            priming: None,
            done: false,
        };
        body.priming = body.priming_bytes();
        body
    }

    /// Token written immediately after the headers. Same vocabulary as the
    /// periodic heartbeat, so the client parser cannot tell them apart.
    fn priming_bytes(&self) -> Option<Bytes> {
        match self.protocol {
            Protocol::Sse => Some(Bytes::from_static(b": keep-alive\n\n")),
            Protocol::Ndjson | Protocol::Unknown => Some(Bytes::from_static(b"\n")),
            // A single leading space keeps the document a valid `JSON.parse`
            // input while still counting as a flushed byte.
            Protocol::Json => Some(Bytes::from_static(b" ")),
        }
    }

    fn heartbeat_bytes(&self) -> Option<Bytes> {
        match self.protocol {
            Protocol::Sse => Some(Bytes::from_static(b": keep-alive\n\n")),
            Protocol::Ndjson | Protocol::Unknown => Some(Bytes::from_static(b"\n")),
            // Once a JSON body has started, whitespace still keeps the byte
            // stream alive without corrupting it.
            Protocol::Json => Some(Bytes::from_static(b"\n")),
        }
    }
}

impl HeartbeatBody {
    /// Convenience wrapper used by tests: drives the body without requiring a
    /// `Stream` adapter.
    pub fn poll_next_frame(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Bytes>, core::convert::Infallible>>> {
        http_body::Body::poll_frame(self, cx)
    }
}

impl http_body::Body for HeartbeatBody {
    type Data = Bytes;
    type Error = core::convert::Infallible;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        if self.done {
            return Poll::Ready(None);
        }

        // 0. Flush the priming token first, before upstream has said anything.
        //    Waiting for the provider here is exactly the silence CF punishes.
        if let Some(priming) = self.priming.take() {
            if self.protocol == Protocol::Json {
                // The space doubles as the JSON prefix, so the first data
                // frame must not add a second one.
                self.json_prefix_sent = true;
            }
            return Poll::Ready(Some(Ok(Frame::data(priming))));
        }

        // 1. Drain the upstream first.
        //
        //    Real data, end-of-stream and errors must ALL get out before a
        //    heartbeat does. Checking the timer first put one last keep-alive
        //    in front of a terminal frame — and in front of the error event
        //    `NormalizeBody` queues — so the client kept receiving heartbeats
        //    after it had already been told the request failed.
        //
        //    The timer is deliberately NOT reset here: a chatty provider
        //    (Google sends `: PING` every few seconds) must not be able to
        //    postpone our own keep-alive, because any hop that buffers those
        //    comments leaves the client with silence until the final payload.
        match self.inner.as_mut().poll_frame(cx) {
            Poll::Ready(Some(Ok(frame))) => {
                if let Some(data) = frame.data_ref() {
                    self.started = true;
                    if self.protocol == Protocol::Json && !self.json_prefix_sent {
                        self.json_prefix_sent = true;
                        // Leading whitespace keeps the JSON valid while proving
                        // liveness to any proxy buffering the first bytes.
                        let mut combined = Bytes::from_static(b" ");
                        combined = [combined, data.clone()].concat().into();
                        return Poll::Ready(Some(Ok(Frame::data(combined))));
                    }
                }
                return Poll::Ready(Some(Ok(frame)));
            }
            Poll::Ready(Some(Err(e))) => {
                self.done = true;
                return Poll::Ready(Some(Err(e)));
            }
            Poll::Ready(None) => {
                self.done = true;
                return Poll::Ready(None);
            }
            Poll::Pending => {}
        }

        // 2. Nothing to forward right now: keep the connection alive on the
        //    wall-clock cadence. This is the only thing left that can be sent,
        //    so a heartbeat can never jump the queue.
        if let Some(timer) = self.timer.as_mut() {
            if timer.as_mut().poll(cx).is_ready() {
                self.reset_timer();
                // A non-stream JSON body is one document: whitespace is only
                // safe BEFORE it starts (leading whitespace is ignored by
                // `JSON.parse`). Once upstream bytes are flowing, a heartbeat
                // could land inside a string literal and corrupt the payload.
                let json_still_safe = self.protocol != Protocol::Json || !self.started;
                if json_still_safe {
                    if let Some(bytes) = self.heartbeat_bytes() {
                        return Poll::Ready(Some(Ok(Frame::data(bytes))));
                    }
                }
            }
        }

        Poll::Pending
    }

    #[allow(clippy::type_complexity)]
    fn size_hint(&self) -> http_body::SizeHint {
        // The priming frame (or heartbeats) makes the length unknown; claiming
        // the upstream's exact length would contradict what we send.
        if self.priming.is_some() {
            return http_body::SizeHint::default();
        }
        self.inner.size_hint()
    }

    fn is_end_stream(&self) -> bool {
        // Reporting the inner stream's end here would be wrong while a priming
        // frame is still queued: hyper checks this BEFORE polling and would
        // finalise the chunked encoding without ever asking for the frame.
        if self.priming.is_some() {
            return false;
        }
        self.done && self.inner.is_end_stream()
    }
}

impl HeartbeatBody {
    fn reset_timer(&mut self) {
        self.timer =
            (self.interval > Duration::ZERO).then(|| Box::pin(tokio::time::sleep(self.interval)));
    }
}

/// The upstream call a [`PendingBody`] is waiting on.
///
/// Generic over the body so the routing layer can hand over a `hyper`
/// `Incoming` while tests hand over a plain channel-backed body.
pub type UpstreamResult<B = hyper::body::Incoming> = Result<hyper::Response<B>, String>;

/// A response body that starts flowing **before** the upstream has answered.
///
/// `HeartbeatBody` can only speak once the headers are already on the wire,
/// which is a case that never happens here: the whole point is the window
/// where the provider has not even sent its status line. This body closes that
/// window by emitting a blank line on the heartbeat cadence while it waits for
/// a oneshot carrying the upstream response. When the response turns up it
/// hands over to a regular [`HeartbeatBody`], which knows the real protocol and
/// can speak that protocol's heartbeat from then on.
pub struct PendingBody<B = hyper::body::Incoming> {
    pending: Option<tokio::sync::oneshot::Receiver<UpstreamResult<B>>>,
    inner: Option<HeartbeatBody>,
    /// One blank line emitted before anything else so the client sees a byte
    /// immediately.
    ///
    /// A blank line is the only token that is safe for every protocol the
    /// upstream might pick: SSE skips empty events, NDJSON skips empty lines,
    /// and neither parser can be confused by a comment line it did not expect.
    priming: Option<Bytes>,
    timer: Option<Pin<Box<tokio::time::Sleep>>>,
    interval: Duration,
    done: bool,
}

impl<B> PendingBody<B>
where
    B: http_body::Body<Data = Bytes> + Send + 'static + Unpin,
{
    pub fn new(
        pending: tokio::sync::oneshot::Receiver<UpstreamResult<B>>,
        interval: Duration,
    ) -> Self {
        Self {
            pending: Some(pending),
            inner: None,
            priming: Some(Bytes::from_static(b"\n")),
            timer: (interval > Duration::ZERO).then(|| Box::pin(tokio::time::sleep(interval))),
            interval,
            done: false,
        }
    }

    fn reset_timer(&mut self) {
        self.timer =
            (self.interval > Duration::ZERO).then(|| Box::pin(tokio::time::sleep(self.interval)));
    }
}

/// Render an error as an SSE event the client already knows how to report.
///
/// The 200 was necessarily sent before the upstream answered, so a failure can
/// no longer travel as a status code. The frontend runs every `data:` payload
/// through `tryParseStreamingError`, which parses the JSON and raises a toast
/// for an `error` field — the same shape a provider error would have had.
fn error_event(message: &str) -> Bytes {
    // A provider error is usually already a JSON object with the shape the
    // frontend looks for (`error`, `message` or `detail`). Passing it through
    // verbatim keeps the provider's own wording; only a non-JSON body (an HTML
    // error page, say) needs wrapping.
    let payload = match serde_json::from_str::<serde_json::Value>(message) {
        Ok(value) if value.is_object() => value.to_string(),
        _ => serde_json::json!({ "error": { "message": message } }).to_string(),
    };
    Bytes::from(format!("data: {payload}\n\n"))
}

impl<B> http_body::Body for PendingBody<B>
where
    B: http_body::Body<Data = Bytes> + Send + 'static + Unpin,
    B::Error: std::fmt::Display,
{
    type Data = Bytes;
    type Error = core::convert::Infallible;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        if self.done {
            return Poll::Ready(None);
        }

        // 0. Proving liveness costs one comment line, sent before anything else.
        if let Some(priming) = self.priming.take() {
            return Poll::Ready(Some(Ok(Frame::data(priming))));
        }

        // 1. Settle the upstream call, if it has settled.
        if self.pending.is_some() {
            let receiver = self.pending.as_mut().expect("checked above");
            match Pin::new(receiver).poll(cx) {
                Poll::Ready(Ok(Ok(response))) => {
                    self.pending = None;
                    let (parts, body) = response.into_parts();
                    let protocol = Protocol::from_content_type(
                        parts
                            .headers
                            .get(axum::http::header::CONTENT_TYPE)
                            .and_then(|v| v.to_str().ok()),
                    );
                    self.inner = Some(HeartbeatBody::new(
                        NormalizeBody {
                            inner: body,
                            done: false,
                        },
                        protocol,
                        self.interval,
                    ));
                }
                Poll::Ready(Ok(Err(message))) => {
                    self.pending = None;
                    self.done = true;
                    return Poll::Ready(Some(Ok(Frame::data(error_event(message.as_str())))));
                }
                Poll::Ready(Err(_)) => {
                    self.pending = None;
                    self.done = true;
                    return Poll::Ready(Some(Ok(Frame::data(error_event(
                        "upstream request was cancelled",
                    )))));
                }
                Poll::Pending => {
                    // 2. Still silent upstream: keep the cadence going.
                    if let Some(timer) = self.timer.as_mut() {
                        if timer.as_mut().poll(cx).is_ready() {
                            self.reset_timer();
                            return Poll::Ready(Some(Ok(Frame::data(Bytes::from_static(b"\n")))));
                        }
                    }
                    return Poll::Pending;
                }
            }
        }

        // 3. The upstream is now the source of truth.
        match self.inner.as_mut() {
            Some(inner) => {
                let poll = std::pin::Pin::new(inner).poll_frame(cx);
                if matches!(poll, Poll::Ready(None)) {
                    self.done = true;
                }
                poll
            }
            None => {
                self.done = true;
                Poll::Ready(None)
            }
        }
    }

    fn size_hint(&self) -> http_body::SizeHint {
        http_body::SizeHint::default()
    }

    fn is_end_stream(&self) -> bool {
        self.done
    }
}

/// Turns a body whose reads can fail into one that cannot, reporting the
/// failure as a final SSE event.
///
/// `HeartbeatBody` speaks `Infallible` so it can be handed to axum directly;
/// without this adapter a mid-stream transport error would have to be faked
/// (a panic, or a silent truncation the client cannot distinguish from a
/// provider that simply stopped talking).
pub struct NormalizeBody<B> {
    inner: B,
    done: bool,
}

impl<B> NormalizeBody<B> {
    pub fn new(inner: B) -> Self {
        Self { inner, done: false }
    }
}

impl<B> http_body::Body for NormalizeBody<B>
where
    B: http_body::Body<Data = Bytes> + Unpin,
    B::Error: std::fmt::Display,
{
    type Data = Bytes;
    type Error = core::convert::Infallible;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        if self.done {
            return Poll::Ready(None);
        }
        match Pin::new(&mut self.inner).poll_frame(cx) {
            Poll::Ready(Some(Ok(frame))) => Poll::Ready(Some(Ok(frame))),
            Poll::Ready(Some(Err(e))) => {
                // Deliver the error as this very frame instead of parking it in
                // `tail` behind a `Poll::Pending`. Returning `Pending` here let
                // the wrapper above poll its heartbeat timer and slip one more
                // keep-alive in *before* the error, which is exactly the
                // "error, then more keep-alives" behaviour that looks like a
                // stuck connection. `self.done = true` makes this the last
                // frame too, so nothing can follow it.
                self.done = true;
                Poll::Ready(Some(Ok(Frame::data(error_event(&format!(
                    "upstream stream failed: {e}"
                ))))))
            }
            Poll::Ready(None) => {
                self.done = true;
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }

    fn size_hint(&self) -> http_body::SizeHint {
        http_body::SizeHint::default()
    }
}

/// Resolve `gateway.heartbeat_secs` into a duration. `0` disables heartbeats;
/// an absent value uses the default.
pub fn interval_from_config(secs: Option<u64>) -> Duration {
    Duration::from_secs(secs.unwrap_or(DEFAULT_HEARTBEAT_SECS))
}

/// Read the full body of `resp`, then wrap it for heartbeat delivery.
pub async fn wrap_response(
    resp: axum::http::Response<hyper::body::Incoming>,
    protocol: Protocol,
    interval: Duration,
) -> axum::response::Response {
    let (mut parts, body) = resp.into_parts();
    // Heartbeats are emitted by us, so the length must become chunked.
    parts.headers.remove(axum::http::header::CONTENT_LENGTH);
    let wrapped = HeartbeatBody::new(
        body.map_err(|_| -> core::convert::Infallible { unreachable!() }),
        protocol,
        interval,
    );
    axum::response::Response::from_parts(parts, axum::body::Body::new(wrapped))
}
