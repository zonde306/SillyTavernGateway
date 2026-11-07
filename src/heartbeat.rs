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
//! | response                      | heartbeat                         |
//! |-------------------------------|-----------------------------------|
//! | `text/event-stream` (SSE)     | `: keep-alive\n\n` comment line   |
//! | NDJSON / `application/jsonl`  | `\n` (an empty line, no record)    |
//! | JSON (non-stream)             | nothing after the body starts;     |
//! |                               | leading ` ` before it             |
//! | `application/x-ndjson`, chunked streaming JSON | `\n`      |
//!
//! A bare `\n` is a no-op for every line/TEXT_EVENT_STREAM parser: `EventSource`
//! ignores blank lines, and JSON parsers ignore leading/interstitial
//! whitespace (upstream continues to `JSON.parse` the whole text).
//! Leading whitespace is inserted lazily on the first byte of the body so a
//! later `JSON.parse` still sees valid JSON.

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
    timer: Option<Pin<Box<tokio::time::Sleep>>>,
    /// Whether the first data byte has been seen (JSON needs no heartbeat
    /// after that; SSE/NDJSON keep them coming).
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
            timer: Some(Box::pin(tokio::time::sleep(interval))),
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

        // 1. Drain whatever the upstream already has.
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
                        self.reset_timer();
                        return Poll::Ready(Some(Ok(Frame::data(combined))));
                    }
                }
                self.reset_timer();
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

        // 2. Upstream is idle — is it time for a heartbeat?
        let Some(timer) = self.timer.as_mut() else {
            return Poll::Pending;
        };
        match timer.as_mut().poll(cx) {
            Poll::Ready(()) => {
                self.reset_timer();
                match self.heartbeat_bytes() {
                    Some(bytes) => Poll::Ready(Some(Ok(Frame::data(bytes)))),
                    None => {
                        cx.waker().wake_by_ref();
                        Poll::Pending
                    }
                }
            }
            Poll::Pending => Poll::Pending,
        }
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
        self.timer = Some(Box::pin(tokio::time::sleep(self.interval)));
    }
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
