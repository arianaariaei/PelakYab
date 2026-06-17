"""App-free phone camera over the browser.

Instead of the third-party *IP Webcam* app, this runs a tiny HTTPS server on the
PC and shows a QR code. The phone's built-in camera app recognizes the QR and
opens a link in the **browser** (no install). The page asks for camera
permission and streams JPEG frames back over a WebSocket; the newest frame is
handed to the pipeline exactly like ``CameraStream`` does.

Why HTTPS: browsers only allow ``getUserMedia`` (camera) on a *secure context*
(HTTPS or localhost). On a LAN there is no public certificate, so we generate a
self-signed one (valid for the PC's LAN IP). The phone shows a one-time
"not secure -> proceed" prompt, then it just works.

This class is a drop-in for ``camera.CameraStream`` (same start/read/connected/
frame_age/rotate_by/stop surface), so ``Pipeline`` only needs to pick
it when ``camera.source == "browser"``.
"""
from __future__ import annotations

import asyncio
import socket
import threading
import time
from io import StringIO
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
CERT_DIR = ROOT / "certs"
TOOLS_DIR = ROOT / "tools"

# cloudflared download per OS (quick tunnel: no account, public HTTPS URL)
_CLOUDFLARED_ASSET = {
    "Windows": "cloudflared-windows-amd64.exe",
    "Darwin": "cloudflared-darwin-amd64.tgz",
    "Linux": "cloudflared-linux-amd64",
}
_CLOUDFLARED_BASE = ("https://github.com/cloudflare/cloudflared/releases/"
                     "latest/download/")


class BrowserCameraStream:
    """Threaded HTTPS+WebSocket server that hands you the most recent frame."""

    def __init__(self, port: int = 8443, rotate: int = 0, flip: str = "",
                 host_ip: Optional[str] = None, stale_after: float = 3.0,
                 tunnel: bool = False, tunnel_provider: str = "auto",
                 relay_url: Optional[str] = None):
        self.port = int(port)
        self.rotate = self._norm_rotate(rotate)
        self.flip = flip if flip in ("h", "v", "hv") else ""
        # Detected LAZILY (in start()/detect_host_ip()), never at construction —
        # so the URL/IP isn't generated until the user has joined the hotspot.
        self.host_ip = host_ip
        self.stale_after = stale_after
        self.tunnel = bool(tunnel)
        self.tunnel_provider = tunnel_provider or "auto"
        # Relay mode: connect OUTBOUND to a hosted relay (relay/server.py on a
        # PaaS). Beats tunnels in filtered networks — both ends dial out, the
        # URL is static, and the cert is real. Overrides tunnel/LAN when set.
        self.relay_url = relay_url.rstrip("/") if relay_url else None
        self.relay = bool(self.relay_url)

        self._frame: Optional[np.ndarray] = None
        self._frame_ts: float = 0.0
        self._lock = threading.Lock()

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._runner = None
        self._thread: Optional[threading.Thread] = None
        self._clients: set = set()
        self._started = threading.Event()
        self._bound = False

        self._public_url: Optional[str] = None   # set once the tunnel is up
        self._cf_proc = None
        self._tunnel_error: Optional[str] = None

        self._relay_ws = None                    # active laptop->relay socket
        self._relay_connected = False
        self._relay_stop = False

    # ------------------------------------------------------------- networking
    @staticmethod
    def _lan_ip() -> str:
        """Best-effort primary LAN IP (the address the phone will dial)."""
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))   # no packets sent; just picks the iface
            return s.getsockname()[0]
        except Exception:
            try:
                return socket.gethostbyname(socket.gethostname())
            except Exception:
                return "127.0.0.1"
        finally:
            s.close()

    @property
    def connect_url(self) -> Optional[str]:
        """The URL the phone opens. Relay mode: the static relay URL. Tunnel
        mode: the public https://*.trycloudflare.com (None until established).
        LAN mode: the local HTTPS URL."""
        if self.relay:
            return self.relay_url + "/"
        if self.tunnel:
            return self._public_url
        return f"https://{self.host_ip}:{self.port}/" if self.host_ip else None

    def detect_host_ip(self) -> str:
        """Resolve the LAN IP now (call only after the PC has joined the
        hotspot / target network)."""
        if not self.host_ip:
            self.host_ip = self._lan_ip()
        return self.host_ip

    @property
    def server_bound(self) -> bool:
        """True once the local web server has successfully bound its port."""
        return self._bound

    @property
    def tunnel_status(self) -> str:
        """'ready' | 'starting' | 'error' | 'lan' for UI/CLI feedback."""
        if self.relay:
            return "ready" if self._relay_connected else "starting"
        if not self.tunnel:
            return "lan"
        if self._public_url:
            return "ready"
        if self._tunnel_error:
            return "error"
        return "starting"

    def wait_for_url(self, timeout: float = 25.0) -> Optional[str]:
        """Block until the connect URL is known (tunnel up) or timeout."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if self.connect_url:
                return self.connect_url
            time.sleep(0.3)
        return self.connect_url

    # ------------------------------------------------------------------- cert
    def _ensure_cert(self) -> Tuple[Path, Path]:
        """Generate (and cache) a self-signed cert whose SAN covers the LAN IP.

        Regenerated automatically if the cached cert doesn't cover the current
        IP (e.g. the laptop moved to a different Wi-Fi).
        """
        CERT_DIR.mkdir(parents=True, exist_ok=True)
        cert_p, key_p, ip_p = (CERT_DIR / "cert.pem", CERT_DIR / "key.pem",
                               CERT_DIR / "host_ip.txt")
        if cert_p.exists() and key_p.exists() and ip_p.exists() \
                and ip_p.read_text().strip() == self.host_ip:
            return cert_p, key_p

        import datetime
        import ipaddress
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import NameOID

        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "PelakYab")])
        san = [x509.DNSName("localhost"), x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]
        try:
            san.append(x509.IPAddress(ipaddress.ip_address(self.host_ip)))
        except ValueError:
            pass
        now = datetime.datetime.utcnow()
        cert = (
            x509.CertificateBuilder()
            .subject_name(name).issuer_name(name)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(days=1))
            .not_valid_after(now + datetime.timedelta(days=3650))
            .add_extension(x509.SubjectAlternativeName(san), critical=False)
            .sign(key, hashes.SHA256())
        )
        key_p.write_bytes(key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption()))
        cert_p.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
        ip_p.write_text(self.host_ip)
        return cert_p, key_p

    def _ssl_context(self):
        import ssl
        cert_p, key_p = self._ensure_cert()
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(str(cert_p), str(key_p))
        return ctx

    # --------------------------------------------------------------- QR codes
    def qr_image(self):
        """Return a PIL image of the connect-URL QR (for the GUI)."""
        import qrcode
        if not self.connect_url:
            raise RuntimeError("connect URL not ready yet")
        return qrcode.make(self.connect_url)

    def qr_terminal(self) -> str:
        """Return an ASCII-art QR of the connect URL (for the console)."""
        import qrcode
        if not self.connect_url:
            return "(connect URL not ready yet)"
        qr = qrcode.QRCode(border=2)
        qr.add_data(self.connect_url)
        qr.make(fit=True)
        buf = StringIO()
        qr.print_ascii(out=buf, invert=True)
        return buf.getvalue()

    # ---------------------------------------------------------------- server
    def start(self) -> "BrowserCameraStream":
        if self._thread and self._thread.is_alive():
            return self
        if self.relay:                       # dial out to the hosted relay
            self._relay_stop = False
            self._thread = threading.Thread(target=self._run_relay, daemon=True,
                                            name="BrowserCameraRelay")
            self._thread.start()
            return self
        self.detect_host_ip()                # resolve LAN IP now (post-hotspot)
        self._thread = threading.Thread(target=self._serve, daemon=True,
                                        name="BrowserCameraStream")
        self._thread.start()
        self._started.wait(timeout=10.0)
        if self.tunnel and self._bound:
            self._start_tunnel()
        return self

    # ----------------------------------------------------------------- relay
    def _run_relay(self) -> None:
        """Maintain an outbound WebSocket to <relay>/laptop, receiving phone
        frames and (re)connecting forever until stop()."""
        import aiohttp
        ws_url = (self.relay_url + "/laptop").replace(
            "https://", "wss://", 1).replace("http://", "ws://", 1)
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        async def client() -> None:
            while not self._relay_stop:
                try:
                    async with aiohttp.ClientSession() as sess:
                        async with sess.ws_connect(
                                ws_url, heartbeat=20,
                                max_msg_size=16 * 1024 * 1024) as ws:
                            self._relay_ws = ws
                            self._relay_connected = True
                            print(f"[BrowserCam] relay connected: {ws_url}")
                            async for msg in ws:
                                if msg.type == aiohttp.WSMsgType.BINARY:
                                    self._on_frame(msg.data)
                                elif msg.type == aiohttp.WSMsgType.ERROR:
                                    break
                except Exception as exc:
                    print(f"[BrowserCam] relay connection failed: {exc}")
                self._relay_connected = False
                self._relay_ws = None
                if self._relay_stop:
                    break
                await asyncio.sleep(2.0)       # reconnect backoff

        try:
            self._loop.run_until_complete(client())
        finally:
            self._loop.close()

    # --------------------------------------------------------------- tunnel
    def _ensure_cloudflared(self) -> Optional[Path]:
        """Return a path to the cloudflared binary, downloading it once."""
        import platform
        asset = _CLOUDFLARED_ASSET.get(platform.system())
        if not asset:
            self._tunnel_error = f"no cloudflared build for {platform.system()}"
            return None
        exe = TOOLS_DIR / ("cloudflared.exe" if asset.endswith(".exe")
                           else "cloudflared")
        if exe.exists():
            return exe
        TOOLS_DIR.mkdir(parents=True, exist_ok=True)
        try:
            import urllib.request
            print("[BrowserCam] downloading cloudflared (one-time, ~20MB)…")
            urllib.request.urlretrieve(_CLOUDFLARED_BASE + asset, exe)
            if not asset.endswith(".exe"):
                exe.chmod(0o755)
            return exe
        except Exception as exc:
            self._tunnel_error = f"cloudflared download failed: {exc}"
            print(f"[BrowserCam] {self._tunnel_error}")
            return None

    def _provider_order(self) -> list[str]:
        """Which tunnel providers to try, in order. 'auto' tries the SSH-based
        localhost.run first (no account, no binary, gets through most
        restrictive networks), then Cloudflare (works where SSH/22 is blocked)."""
        p = (self.tunnel_provider or "auto").lower()
        if p in ("localhostrun", "localhost.run", "lhr", "ssh"):
            return ["localhostrun"]
        if p in ("cloudflare", "cf", "cloudflared"):
            return ["cloudflare"]
        return ["localhostrun", "cloudflare"]

    def _provider_command(self, prov: str):
        """(argv, url_regex) for a provider, or (None, None) if unavailable."""
        import re
        if prov == "localhostrun":
            argv = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes",
                    "-o", "ServerAliveInterval=30", "-R",
                    f"80:localhost:{self.port}", "nokey@localhost.run"]
            return argv, re.compile(r"https://[a-z0-9]+\.lhr\.life")
        if prov == "cloudflare":
            exe = self._ensure_cloudflared()
            if exe is None:
                return None, None
            argv = [str(exe), "tunnel", "--no-autoupdate", "--protocol", "http2",
                    "--url", f"http://localhost:{self.port}"]
            return argv, re.compile(
                r"https://[a-z0-9]+(?:-[a-z0-9]+)+\.trycloudflare\.com")
        return None, None

    def _try_provider(self, prov: str, timeout: float):
        """Launch one provider and wait up to `timeout` for its public URL.
        Returns (proc, url_or_None). A reader thread keeps draining the pipe so
        the winning process doesn't block on a full buffer."""
        import subprocess
        argv, pat = self._provider_command(prov)
        if argv is None:
            return None, None
        flags = subprocess.CREATE_NO_WINDOW if hasattr(
            subprocess, "CREATE_NO_WINDOW") else 0
        try:
            proc = subprocess.Popen(argv, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True,
                                    bufsize=1, creationflags=flags)
        except Exception as exc:
            self._tunnel_error = f"{prov} could not start: {exc}"
            return None, None
        result = {"url": None}
        done = threading.Event()

        def reader():
            try:
                for line in proc.stdout:
                    if result["url"] is None:
                        m = pat.search(line)
                        if m:
                            result["url"] = m.group(0).rstrip("/") + "/"
                            done.set()
                        elif "failed to request quick Tunnel" in line:
                            done.set()
            except Exception:
                pass
            done.set()                       # process ended / pipe closed

        threading.Thread(target=reader, daemon=True, name=f"tunnel-{prov}").start()
        done.wait(timeout)
        return proc, result["url"]

    def _start_tunnel(self) -> None:
        """Try each provider in order; first public URL wins. If none work, fall
        back to direct same-Wi-Fi HTTPS so the app still works (never hangs)."""
        for prov in self._provider_order():
            print(f"[BrowserCam] trying tunnel provider: {prov}…")
            proc, url = self._try_provider(prov, timeout=18.0)
            if url:
                self._cf_proc = proc
                self._public_url = url
                print(f"[BrowserCam] public URL ({prov}): {url}")
                return
            if proc is not None:
                try:
                    proc.terminate()
                except Exception:
                    pass
        if not self._tunnel_error:
            self._tunnel_error = "no tunnel provider worked on this network"
        print(f"[BrowserCam] {self._tunnel_error}; falling back to same-Wi-Fi.")
        self.tunnel = False
        self._restart_for_lan()

    def _restart_for_lan(self) -> None:
        """If the tunnel failed, rebind the server as LAN HTTPS so the QR still
        works on the same Wi-Fi."""
        try:
            self.stop()
        except Exception:
            pass
        self._bound = False
        self._started.clear()
        self._thread = threading.Thread(target=self._serve, daemon=True,
                                        name="BrowserCameraStream")
        self._thread.start()
        self._started.wait(timeout=10.0)

    def _serve(self) -> None:
        from aiohttp import web

        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        app = web.Application()
        app.add_routes([web.get("/", self._handle_index),
                        web.get("/ws", self._handle_ws)])
        runner = web.AppRunner(app)
        self._loop.run_until_complete(runner.setup())
        # Tunnel mode: serve plain HTTP bound to localhost (cloudflared connects
        # locally and provides public HTTPS). LAN mode: self-signed HTTPS on all
        # interfaces so the phone on the same Wi-Fi can reach it directly.
        if self.tunnel:
            site = web.TCPSite(runner, host="127.0.0.1", port=self.port)
        else:
            site = web.TCPSite(runner, host="0.0.0.0", port=self.port,
                               ssl_context=self._ssl_context())
        try:
            self._loop.run_until_complete(site.start())
        except OSError as exc:
            print(f"[BrowserCam] could not bind port {self.port}: {exc}")
            self._started.set()
            return
        self._runner = runner
        self._bound = True
        self._started.set()
        try:
            self._loop.run_forever()
        finally:
            self._loop.run_until_complete(runner.cleanup())
            self._loop.close()

    async def _handle_index(self, request):
        from aiohttp import web
        return web.Response(text=_PAGE_HTML, content_type="text/html")

    async def _handle_ws(self, request):
        from aiohttp import web
        ws = web.WebSocketResponse(max_msg_size=16 * 1024 * 1024, heartbeat=20)
        await ws.prepare(request)
        self._clients.add(ws)
        try:
            async for msg in ws:
                if msg.type == web.WSMsgType.BINARY:
                    self._on_frame(msg.data)
                # TEXT messages (hello/heartbeat) are ignored
        finally:
            self._clients.discard(ws)
        return ws

    def _on_frame(self, data: bytes) -> None:
        arr = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return
        self._store(self._orient(frame))

    def _store(self, frame: np.ndarray) -> None:
        with self._lock:
            self._frame = frame
            self._frame_ts = time.monotonic()

    # ------------------------------------------------------------------- read
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self._lock:
            if self._frame is None or (time.monotonic() - self._frame_ts) > self.stale_after:
                return False, None
            return True, self._frame.copy()

    @property
    def connected(self) -> bool:
        with self._lock:
            fresh = self._frame is not None and \
                (time.monotonic() - self._frame_ts) <= self.stale_after
        if self.relay:
            return self._relay_connected and fresh
        return bool(self._clients) and fresh

    @property
    def frame_age(self) -> float:
        with self._lock:
            if self._frame_ts == 0:
                return float("inf")
            return time.monotonic() - self._frame_ts

    # ------------------------------------------------------------ orientation
    @staticmethod
    def _norm_rotate(deg) -> int:
        try:
            deg = int(deg) % 360
        except Exception:
            return 0
        return deg if deg in (0, 90, 180, 270) else (round(deg / 90) * 90) % 360

    def set_rotation(self, deg: int) -> None:
        self.rotate = self._norm_rotate(deg)

    def rotate_by(self, delta: int) -> int:
        self.rotate = self._norm_rotate(self.rotate + delta)
        return self.rotate

    def _orient(self, frame: np.ndarray) -> np.ndarray:
        if self.rotate == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif self.rotate == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif self.rotate == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if self.flip == "h":
            frame = cv2.flip(frame, 1)
        elif self.flip == "v":
            frame = cv2.flip(frame, 0)
        elif self.flip == "hv":
            frame = cv2.flip(frame, -1)
        return frame

    # ----------------------------------------------------------------- close
    def stop(self) -> None:
        if self.relay:
            self._relay_stop = True
            if self._relay_ws is not None and self._loop:
                try:        # closing the socket breaks the receive loop
                    asyncio.run_coroutine_threadsafe(
                        self._relay_ws.close(), self._loop)
                except Exception:
                    pass
            if self._thread:
                self._thread.join(timeout=4.0)
            return
        if self._cf_proc is not None:
            try:
                self._cf_proc.terminate()
            except Exception:
                pass
            self._cf_proc = None
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=3.0)

    def __enter__(self) -> "BrowserCameraStream":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


# ---------------------------------------------------------------------------
# The phone-side page: request the back camera and stream JPEG frames over a
# WebSocket. Kept dependency-free and tolerant of iOS Safari (needs a user
# gesture + playsinline).
# ---------------------------------------------------------------------------
_PAGE_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">
<title>PelakYab camera</title>
<style>
  :root { color-scheme: dark; }
  body { margin:0; background:#15171e; color:#e8eaf0; font-family:system-ui,sans-serif;
         display:flex; flex-direction:column; align-items:center; min-height:100vh; }
  h1 { font-size:1.1rem; margin:14px 0 4px; }
  #status { font-size:.95rem; padding:6px 14px; border-radius:20px; margin:6px;
            background:#2a2d39; }
  #status.ok { background:#1f7a3d; }
  #status.err { background:#9b2c2c; }
  video { width:94vw; max-width:520px; border-radius:12px; background:#000; margin-top:6px; }
  button { font-size:1.1rem; padding:14px 28px; margin:16px; border:0; border-radius:12px;
           background:#2d6cdf; color:#fff; }
  .hint { font-size:.82rem; color:#9aa0b0; max-width:90vw; text-align:center; line-height:1.4; }
</style>
</head>
<body>
  <h1>PelakYab — phone camera</h1>
  <div id="status">Tap “Start camera” to connect</div>
  <button id="startBtn">Start camera</button>
  <video id="v" autoplay playsinline muted></video>
  <p class="hint">Point the back camera at the licence plate. Keep this page open.
     You can lock nothing — just leave the screen on.</p>
<script>
const statusEl = document.getElementById('status');
const startBtn = document.getElementById('startBtn');
const video = document.getElementById('v');
const canvas = document.createElement('canvas');
const ctx = canvas.getContext('2d');
let ws = null, track = null, sending = false;

function setStatus(t, cls){ statusEl.textContent = t; statusEl.className = cls || ''; }

function connectWS(){
  ws = new WebSocket((location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws');
  ws.binaryType = 'arraybuffer';
  ws.onopen = () => { setStatus('Connected — streaming ✓','ok'); sending = true; pump(); };
  ws.onclose = () => { setStatus('Disconnected — reconnecting…','err'); sending = false;
                       setTimeout(connectWS, 1500); };
  ws.onerror = () => { setStatus('Connection error','err'); };
}

async function start(){
  startBtn.style.display = 'none';
  setStatus('Requesting camera…');
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: 'environment' },
               width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false });
    video.srcObject = stream;
    track = stream.getVideoTracks()[0];
    await video.play();
    connectWS();
  } catch (e) {
    setStatus('Camera blocked: ' + e.message, 'err');
    startBtn.style.display = '';
  }
}

function pump(){
  if (!sending) return;
  if (ws && ws.readyState === 1 && ws.bufferedAmount < 256*1024 &&
      video.videoWidth > 0) {
    const w = video.videoWidth, h = video.videoHeight;
    if (canvas.width !== w) { canvas.width = w; canvas.height = h; }
    ctx.drawImage(video, 0, 0, w, h);
    canvas.toBlob((b) => { if (b && ws && ws.readyState === 1)
        b.arrayBuffer().then(buf => ws.send(buf)); }, 'image/jpeg', 0.6);
  }
  setTimeout(() => requestAnimationFrame(pump), 70);   // ~12-14 fps
}

startBtn.onclick = start;
</script>
</body>
</html>
"""
