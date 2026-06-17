#!/usr/bin/env python
"""PelakYab — entry point.

    python main.py                 # launch the GUI (default)
    python main.py --headless      # run the pipeline with no GUI (prints events)
    python main.py --config x.yaml # use an alternate config
    python main.py --check         # verify env + models + camera, then exit
"""
from __future__ import annotations

import argparse
import sys

from pelakyab.config import load_config


def cmd_headless(cfg) -> int:
    from pelakyab.pipeline import Pipeline

    def on_event(ev):
        p = ev.plate
        print(f"[NEW] {p.display_en:18s} | {p.plate_type:12s} | "
              f"{p.province}/{p.city} | conf={p.confidence:.2f}")

    pipe = Pipeline(cfg)
    print("Running headless. Ctrl+C to stop.")
    pipe.run(on_event=on_event)
    return 0


def cmd_check(cfg) -> int:
    import os
    ok = True
    print("PelakYab environment check")
    print("-" * 40)

    # deps
    for mod in ("cv2", "numpy", "yaml", "requests"):
        try:
            __import__(mod)
            print(f"  [ok]  {mod}")
        except Exception as e:
            ok = False
            print(f"  [!!]  {mod}: {e}")
    for mod in ("ultralytics", "PySide6", "PIL", "arabic_reshaper", "bidi"):
        try:
            __import__(mod)
            print(f"  [ok]  {mod}")
        except Exception as e:
            print(f"  [warn] {mod} missing ({e})")

    # torch / cuda
    try:
        import torch
        print(f"  [ok]  torch {torch.__version__}  cuda={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"        device: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"  [warn] torch: {e}")

    # models
    for label, path in (("plate_model", cfg.detection.plate_model),
                        ("char_model", cfg.detection.char_model)):
        exists = os.path.exists(path)
        ok = ok and exists
        print(f"  [{'ok' if exists else '!!'}]  {label}: {path}")

    # font
    print(f"  [{'ok' if os.path.exists(cfg.gui.persian_font) else 'warn'}]  "
          f"persian font: {cfg.gui.persian_font}")

    # camera (quick, non-fatal)
    if str(cfg.camera.source).strip().lower() == "browser":
        for mod in ("aiohttp", "qrcode", "cryptography"):
            try:
                __import__(mod)
                print(f"  [ok]  {mod}")
            except Exception as e:
                ok = False
                print(f"  [!!]  {mod}: {e}  (pip install -r requirements.txt)")
        try:
            from pelakyab.camera import BrowserCameraStream
            browser = getattr(cfg.camera, "browser", None)
            port = getattr(browser, "port", 8443)
            tunnel = getattr(browser, "tunnel", False)
            cam = BrowserCameraStream(port=port, tunnel=tunnel)
            if tunnel:
                print("  camera: browser via internet link (cloudflared) "
                      "— public URL is generated at launch")
            else:
                cam.detect_host_ip()
                print(f"  camera: browser (same-Wi-Fi)  ->  {cam.connect_url}")
        except Exception as e:
            print(f"  [warn] browser camera: {e}")
    else:
        print(f"  camera source: {cfg.camera.source}")
    print("-" * 40)
    print("READY" if ok else "Some required items are missing (see above).")
    return 0 if ok else 1


def cmd_qr(cfg) -> int:
    """Print the phone-connect URL + an ASCII QR and save a printable PNG."""
    import time
    from pathlib import Path
    from pelakyab.camera import BrowserCameraStream
    if BrowserCameraStream is None:
        print("Browser camera deps missing. pip install -r requirements.txt")
        return 1
    browser = getattr(cfg.camera, "browser", None)
    port = getattr(browser, "port", 8443)
    tunnel = getattr(browser, "tunnel", False)
    cam = BrowserCameraStream(port=port, tunnel=tunnel)

    if tunnel:
        print("Starting secure internet link (cloudflared)…")
        cam.start()
        if not cam.wait_for_url(30):
            print("Could not establish the link:",
                  getattr(cam, "_tunnel_error", None) or "timed out")
            cam.stop()
            return 1
    else:
        cam.detect_host_ip()                  # resolve LAN IP for the printed URL

    print("\nPhone-connect URL:", cam.connect_url, "\n")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        print(cam.qr_terminal())
    except Exception:
        pass
    try:
        out = Path(__file__).resolve().parent / "connect_qr.png"
        cam.qr_image().save(out)
        print(f"Saved printable QR -> {out}")
    except Exception as e:
        print(f"(could not save PNG: {e})")

    if tunnel:
        print("\nKeep this running while you scan. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            cam.stop()
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="PelakYab — Iranian plate recognition")
    ap.add_argument("--config", default=None, help="path to config.yaml")
    ap.add_argument("--headless", action="store_true", help="run without GUI")
    ap.add_argument("--check", action="store_true", help="verify setup and exit")
    ap.add_argument("--qr", action="store_true",
                    help="print/save the phone-connect QR (browser camera) and exit")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)

    if args.check:
        return cmd_check(cfg)
    if args.qr:
        return cmd_qr(cfg)
    if args.headless:
        return cmd_headless(cfg)

    from pelakyab.gui import run_gui
    return run_gui(cfg)


if __name__ == "__main__":
    sys.exit(main())
