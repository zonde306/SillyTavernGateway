"""End-to-end check for /api/backends/chat-completions/generate.

Runs a fake OpenAI-compatible upstream whose first byte is delayed, puts the
real gateway in front of it, and asserts:

  1. request translation (URL, headers, body shape) survives the gateway;
  2. a streaming response reaches the client immediately (headers first);
  3. idle time before the first token produces protocol-safe heartbeats;
  4. the final client-side text is parseable.
"""

import base64
import hashlib
import hmac
import http.server
import json
import os
import socket
import socketserver
import subprocess
import tempfile
import threading
import time
import urllib.parse
import urllib.parse
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # src-rs
EXE = os.path.join(ROOT, "target", "release", "st-gateway.exe")

FIRST_TOKEN_DELAY = 3.0   # seconds of silence before the first SSE event
HEARTBEAT_SECS = 1        # gateway heartbeat interval under test

captured = {}
UPSTREAM_PORT = [0]


class Upstream(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        captured["path"] = self.path
        captured["port"] = UPSTREAM_PORT[0]
        captured["headers"] = {k.lower(): v for k, v in self.headers.items()}
        captured["body"] = body

        stream = bool(body.get("stream"))
        self.send_response(200)
        if stream:
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            # hyper (gateway -> upstream) needs either a length or chunked;
            # http.server speaks HTTP/1.1, so chunked is honoured.
        else:
            payload = json.dumps({"choices": [{"message": {"content": "hi"}}]}).encode()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
        self.end_headers()

        def send_chunk(data: bytes):
            self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
            self.wfile.flush()

        time.sleep(FIRST_TOKEN_DELAY)
        if stream:
            send_chunk(b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n')
            send_chunk(b"data: [DONE]\n\n")
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        else:
            self.wfile.write(json.dumps({"choices": [{"message": {"content": "hi"}}]}).encode())
            self.wfile.flush()
        captured["served_at"] = time.time()
        self.close_connection = True

    def log_message(self, format, *args):  # noqa: A002 - base class signature
        pass


class Threaded(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def wait_for_port(port, timeout=10.0):
    """Poll until the gateway accepts connections (avoids flaky sleeps)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.05)
    raise AssertionError(f"port {port} never opened")


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def session_cookies(secret_path, hostname):
    """Build the session cookie the way cookie-session + keygrip do.

    keygrip signs `name=value` and encodes the HMAC-SHA1 with the base64
    *url-safe* alphabet (`/`->`_`, `+`->`-`) without padding; the cookie value
    itself is URI-encoded.
    """
    secret = open(secret_path, "rb").read()
    name = "session-" + hashlib.sha256(hostname.encode()).hexdigest()[:8]
    payload = base64.b64encode(
        json.dumps({"cookie": {}, "handle": "default-user", "csrfToken": "tok"}).encode()
    ).decode()
    value = urllib.parse.quote(payload, safe="")
    raw = hmac.new(secret, f"{name}={value}".encode(), hashlib.sha1).digest()
    sig = base64.b64encode(raw).decode().replace("/", "_").replace("+", "-").rstrip("=")
    return f"{name}={value}; {name}.sig={sig}"


def check_gate(config_path, gateway_port, cookie):
    """Run the gateway with a tiny budget and prove oversized requests queue.

    Two concurrent requests, each carrying ~20 MiB of "images", against a
    32 MiB budget: 16 MiB shares -> every request costs 2 permits -> `total`
    is 2 permits, so exactly one may run while the other waits. The upstream
    measures how many requests are in flight at once.
    """
    import re

    with open(config_path, "a") as f:
        f.write("")
    text = open(config_path).read()
    text = re.sub(r"batch_bytes: \d+", "batch_bytes: 33554432", text)
    with open(config_path, "w") as f:
        f.write(text)

    with open(config_path) as f:
        text = f.read()
    text = re.sub(r"listen_addr: \"[^\"]*\"", f'listen_addr: "127.0.0.1:{gateway_port + 1}"', text)
    with open(config_path, "w") as f:
        f.write(text)

    proc = subprocess.Popen(
        [EXE, "--config", config_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        wait_for_port(gateway_port + 1)
        inflight = {"now": 0, "max": 0, "lock": threading.Lock()}
        original = Upstream.do_POST

        def tracked(self):
            with inflight["lock"]:
                inflight["now"] += 1
                inflight["max"] = max(inflight["max"], inflight["now"])
            try:
                return original(self)
            finally:
                with inflight["lock"]:
                    inflight["now"] -= 1

        Upstream.do_POST = tracked
        try:
            blobs = [
                {
                    "chat_completion_source": "custom",
                    "custom_url": f"http://127.0.0.1:{captured['port']}/v1",
                    "model": "m",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "hi"},
                                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + "A" * 20 * 1024 * 1024}},
                            ],
                        }
                    ],
                    "stream": False,
                }
                for _ in range(2)
            ]

            errors = []

            def send(payload):
                req = urllib.request.Request(
                    f"http://127.0.0.1:{gateway_port + 1}/api/backends/chat-completions/generate",
                    data=json.dumps(payload).encode(),
                    headers={
                        "Content-Type": "application/json",
                        "Cookie": cookie,
                        "x-csrf-token": "tok",
                    },
                    method="POST",
                )
                try:
                    with urllib.request.urlopen(req, timeout=60) as resp:
                        resp.read()
                except Exception as exc:  # noqa: BLE001 - surfaced below
                    errors.append(exc)

            threads = [threading.Thread(target=send, args=(b,)) for b in blobs]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            assert not errors, errors
            assert inflight["max"] == 1, (
                f"gate allowed {inflight['max']} concurrent oversized requests"
            )
            print(f"OK: gate serialized the oversized requests (max in flight = {inflight['max']})")
        finally:
            Upstream.do_POST = original
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def main():
    upstream_port = free_port()
    gateway_port = free_port()

    UPSTREAM_PORT[0] = upstream_port
    server = Threaded(("127.0.0.1", upstream_port), Upstream)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    data_root = tempfile.mkdtemp(prefix="stgw-e2e-")
    user_dir = os.path.join(data_root, "default-user")
    os.makedirs(user_dir, exist_ok=True)
    with open(os.path.join(user_dir, "secrets.json"), "w") as f:
        json.dump(
            {"api_key_custom": [{"id": "1", "value": "sk-e2e", "label": "", "active": True}]},
            f,
        )
    secret_path = os.path.join(data_root, "cookie-secret.txt")
    with open(secret_path, "wb") as f:
        f.write(b"0123456789abcdef0123456789abcdef")

    cfg = os.path.join(data_root, "config.yaml")
    with open(cfg, "w") as f:
        f.write(
            "\n".join(
                [
                    f'dataRoot: "{data_root.replace(os.sep, "/")}"',
                    "port: 8000",
                    "gateway:",
                    f'  listen_addr: "127.0.0.1:{gateway_port}"',
                    f'  node_addr: "127.0.0.1:{upstream_port}"',
                    f"  heartbeat_secs: {HEARTBEAT_SECS}",
                    "  batch_bytes: 268435456",
                ]
            )
        )

    proc = subprocess.Popen(
        [EXE, "--config", cfg],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        wait_for_port(gateway_port)
        hostname = socket.gethostname()
        cookie = session_cookies(secret_path, hostname)

        body = {
            "chat_completion_source": "custom",
            "custom_url": f"http://127.0.0.1:{upstream_port}/v1",
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "temperature": 0.5,
        }
        req = urllib.request.Request(
            f"http://127.0.0.1:{gateway_port}/api/backends/chat-completions/generate",
            data=json.dumps(body).encode(),
            headers={
                "Content-Type": "application/json",
                "Cookie": cookie,
                "x-csrf-token": "tok",
            },
            method="POST",
        )

        started = time.time()
        with urllib.request.urlopen(req, timeout=30) as resp:
            first_byte_at = 0.0
            chunks = []
            while True:
                # `read(n)` blocks until n bytes arrive, which would hide the
                # priming token behind the first real token. `read1` returns as
                # soon as anything is available.
                part = resp.read1(65536)
                if not part:
                    break
                if not first_byte_at:
                    first_byte_at = time.time()
                chunks.append(part)
        text = b"".join(chunks).decode()

        assert captured["path"] == "/v1/chat/completions", captured["path"]
        assert captured["headers"]["authorization"] == "Bearer sk-e2e"
        assert captured["body"]["model"] == "test-model"
        assert captured["body"]["stream"] is True

        silent_for = captured["served_at"] - started
        assert silent_for >= FIRST_TOKEN_DELAY - 0.5, silent_for
        heartbeats = text.count(": keep-alive")
        # One priming token right after the 200 header + one per idle second.
        assert heartbeats >= 3, f"expected heartbeats during {silent_for:.1f}s silence, got {text!r}"
        assert text.startswith(": keep-alive"), (
            "the first body byte must be the priming token so CDNs see a "
            f"live response immediately, got {text[:40]!r}"
        )
        assert 'data: {"choices":[{"delta":{"content":"hello"}}]}' in text
        assert text.rstrip().endswith("data: [DONE]")

        first_byte_delay = (first_byte_at or started) - started
        # The priming token means the client sees bytes before the provider
        # has produced anything.
        assert first_byte_delay < 0.5, (
            f"first byte took {first_byte_delay:.2f}s; it must not wait for the "
            f"provider's {FIRST_TOKEN_DELAY}s think time"
        )
        check_gate(config_path=cfg, gateway_port=gateway_port, cookie=cookie)

        print(f"OK: first gateway byte after {first_byte_delay:.2f}s, "
              f"{heartbeats} heartbeats during {silent_for:.2f}s of upstream silence")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        server.shutdown()


if __name__ == "__main__":
    main()
