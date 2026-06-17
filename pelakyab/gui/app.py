"""PelakYab desktop GUI (PySide6) — two-tab dark interface.

  ┌ Toolbar: connection · fps · link · Rotate · Connect phone ─────────────┐
  ├ Tab "Live" ────────────────┬──────────────────────────────────────────┤
  │  live annotated video       │  DETECTED NOW card (plate, crop, fields)  │
  │                             │  + this-session feed                      │
  ├ Tab "History" ─────────────┴──────────────────────────────────────────┤
  │  search + table of stored plates (thumbnails)  │  DETAIL (car + fields) │
  └────────────────────────────────────────────────────────────────────────┘

A background ``PipelineWorker`` (QThread) runs the CV pipeline and emits Qt
signals; all widget updates happen on the GUI thread.
"""
from __future__ import annotations

import io
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from ..config import Config, ROOT, load_config
from ..pipeline import Pipeline
from ..storage import PlateStore
from ..data.plate_types import latin_letter

# Unicode bidi marks: force a base text direction so plate strings (mixed
# Latin digits + strong-RTL Persian letters) don't get their groups reordered.
RLM = chr(0x200F)   # right-to-left mark
LRM = chr(0x200E)   # left-to-right mark

# colour per plate category (also used for the live card accent)
CATEGORY_HEX = {
    "private": "#3ecf6a", "public": "#ffcc00", "police": "#2ecc71",
    "military": "#7f8fa6", "government": "#e74c3c", "diplomatic": "#3498db",
    "agricultural": "#16a085", "special": "#9b59b6", "temporary": "#bdc3c7",
    "unknown": "#95a5a6",
}

STYLESHEET = """
* { font-family: "Segoe UI", "Tahoma", sans-serif; }
QMainWindow, QWidget { background: #1e1f26; color: #e7e9ee; font-size: 13px; }
QToolBar { background: #15161b; border: 0; spacing: 8px; padding: 6px 8px; }
QToolBar QLabel { padding: 0 8px; }
QTabWidget::pane { border: 1px solid #2c2e38; border-radius: 8px; top: -1px; }
QTabBar::tab { background: #23252f; color: #aab1c4; padding: 9px 22px;
               margin-right: 3px; border-top-left-radius: 8px;
               border-top-right-radius: 8px; font-weight: 600; }
QTabBar::tab:selected { background: #2d6cdf; color: #ffffff; }
QTabBar::tab:hover:!selected { background: #2a2d39; }
QTableWidget { background: #23252f; border: 1px solid #2c2e38; border-radius: 8px;
               gridline-color: #2c2e38; selection-background-color: #2d6cdf;
               selection-color: #fff; }
QHeaderView::section { background: #15161b; color: #8b93a7; padding: 8px;
                       border: 0; border-bottom: 1px solid #2c2e38; }
QPushButton { background: #2d6cdf; color: #fff; border: 0; padding: 7px 14px;
              border-radius: 7px; font-weight: 600; }
QPushButton:hover { background: #3b7af0; }
QPushButton:pressed { background: #245bc0; }
QPushButton#ghost { background: #2a2d39; color: #cfd5e3; }
QPushButton#ghost:hover { background: #343847; }
QComboBox, QLineEdit { background: #23252f; border: 1px solid #343847;
                       padding: 6px 8px; border-radius: 7px; }
QComboBox::drop-down { border: 0; }
QFrame#card { background: #23252f; border: 1px solid #2c2e38; border-radius: 12px; }
QLabel#video { background: #000; border: 1px solid #2c2e38; border-radius: 10px;
               color: #555; }
QLabel#plateBig { font-size: 30px; font-weight: 800; }
QLabel#sectionTitle { color: #8b93a7; font-size: 12px; font-weight: 700;
                      letter-spacing: 1px; }
QTextEdit { background: #23252f; border: 1px solid #2c2e38; border-radius: 8px; }
QListWidget { background: #23252f; border: 1px solid #2c2e38; border-radius: 8px; }
QStatusBar { background: #15161b; color: #8b93a7; }
QSplitter::handle { background: #15161b; }
"""


# --------------------------------------------------------------------------- #
def bgr_to_qimage(frame: np.ndarray) -> QtGui.QImage:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    return QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888).copy()


def resolve_image(path: str | None) -> Path | None:
    if not path:
        return None
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    return p if p.exists() else None


# --------------------------------------------------------------------------- #
#  worker thread
# --------------------------------------------------------------------------- #
class PipelineWorker(QtCore.QThread):
    frameReady = QtCore.Signal(QtGui.QImage)
    plateEvent = QtCore.Signal(dict, bool, QtGui.QImage)   # record, is_new, crop
    stats = QtCore.Signal(dict)
    status = QtCore.Signal(str)
    failed = QtCore.Signal(str)

    def __init__(self, cfg: Config, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.pipeline = Pipeline(cfg)
        self._running = True
        self._min_interval = 1.0 / max(1, cfg.gui.max_live_fps)

    def run(self) -> None:
        try:
            self.status.emit("Connecting to camera…")
            self.pipeline.start_camera()
            self.status.emit("Loading models (first run can take a while)…")
            self.pipeline.warmup()
            self.status.emit("Running")
        except Exception as exc:
            self.failed.emit(f"Startup failed: {exc}")
            return

        last, fps = 0.0, 0.0
        while self._running:
            now = time.perf_counter()
            if now - last < self._min_interval:
                time.sleep(0.003)
                continue
            dt = now - last
            last = now
            try:
                result = self.pipeline.process_once(draw=True)
            except Exception as exc:
                self.status.emit(f"Frame error: {exc}")
                time.sleep(0.1)
                continue
            if result is None:
                self.stats.emit({"connected": self.pipeline.camera.connected})
                time.sleep(0.05)
                continue

            fps = 0.8 * fps + 0.2 * (1.0 / dt if dt > 0 else 0.0)
            self.frameReady.emit(bgr_to_qimage(result.frame))
            for ev in result.events:
                crop = bgr_to_qimage(ev.plate_crop) if ev.plate_crop is not None \
                    else QtGui.QImage()
                self.plateEvent.emit(ev.record, ev.is_new_event, crop)
            self.stats.emit({
                "connected": self.pipeline.camera.connected,
                "n_plates": result.n_plates,
                "fps": round(fps, 1), "infer_ms": round(result.infer_ms, 1),
            })
        self.pipeline.stop()

    @QtCore.Slot(int)
    def rotate_by(self, delta: int) -> int:
        return self.pipeline.camera.rotate_by(delta)

    def stop(self) -> None:
        self._running = False
        self.wait(4000)


# --------------------------------------------------------------------------- #
#  small helpers
# --------------------------------------------------------------------------- #
def _section(title: str) -> QtWidgets.QLabel:
    lbl = QtWidgets.QLabel(title.upper())
    lbl.setObjectName("sectionTitle")
    return lbl


def prov_city(rec: dict) -> str:
    s = rec.get("province", "")
    if rec.get("city"):
        s += f" / {rec['city']}"
    return s


def plate_en_str(rec: dict) -> str:
    """All-ASCII EN plate built from structured fields (e.g. '43 ein 971 - 51').

    Built from fields rather than the stored string so older records (which may
    have saved the Persian letter) also render with no RTL character.
    """
    if rec.get("left") and rec.get("region_code"):
        return (f"{rec['left']} {latin_letter(rec.get('letter',''))} "
                f"{rec.get('serial','')} - {rec['region_code']}")
    return rec.get("plate_en", "")


def plate_fa_str(rec: dict) -> str:
    """Persian plate in physical left-to-right order (two digits LEFT, region
    code RIGHT). Each token is prefixed with an LRM so the bidi algorithm keeps
    the groups in plate order while each Persian word still shapes RTL.

    Built from structured fields so records stored with the old (un-marked)
    string also render correctly.
    """
    if rec.get("left") and rec.get("region_code"):
        parts = [rec.get("left", ""), rec.get("letter", ""),
                 rec.get("serial", ""), "ایران", rec.get("region_code", "")]
        return " ".join(LRM + p for p in parts if p)
    fa = rec.get("plate_fa", "")
    return (LRM + fa) if fa else ""


# --------------------------------------------------------------------------- #
#  LIVE tab
# --------------------------------------------------------------------------- #
class LiveTab(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(10)

        self.video = QtWidgets.QLabel("Waiting for camera…")
        self.video.setObjectName("video")
        self.video.setAlignment(QtCore.Qt.AlignCenter)
        self.video.setMinimumSize(640, 480)

        # ---- "detected now" card ----
        card = QtWidgets.QFrame(); card.setObjectName("card")
        card.setFixedWidth(360)
        cl = QtWidgets.QVBoxLayout(card)
        cl.setContentsMargins(16, 16, 16, 16); cl.setSpacing(10)

        cl.addWidget(_section("Detected now"))
        self.plate_big = QtWidgets.QLabel("—")
        self.plate_big.setObjectName("plateBig")
        self.plate_big.setAlignment(QtCore.Qt.AlignCenter)
        self.plate_big.setLayoutDirection(QtCore.Qt.LeftToRight)
        cl.addWidget(self.plate_big)

        self.plate_crop = QtWidgets.QLabel()
        self.plate_crop.setAlignment(QtCore.Qt.AlignCenter)
        self.plate_crop.setFixedHeight(70)
        self.plate_crop.setStyleSheet("background:#15161b; border-radius:6px;")
        cl.addWidget(self.plate_crop)

        self.fields = QtWidgets.QLabel("Point the camera at a plate.")
        self.fields.setWordWrap(True)
        self.fields.setTextFormat(QtCore.Qt.RichText)
        cl.addWidget(self.fields)

        cl.addSpacing(6)
        cl.addWidget(_section("This session"))
        self.feed = QtWidgets.QListWidget()
        cl.addWidget(self.feed, stretch=1)

        lay.addWidget(self.video, stretch=1)
        lay.addWidget(card, stretch=0)

    @QtCore.Slot(QtGui.QImage)
    def set_frame(self, img: QtGui.QImage) -> None:
        self.video.setPixmap(QtGui.QPixmap.fromImage(img).scaled(
            self.video.size(), QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation))

    def show_event(self, rec: dict, crop: QtGui.QImage, is_new: bool) -> None:
        color = CATEGORY_HEX.get(rec.get("category", "unknown"), "#95a5a6")
        self.plate_big.setText(plate_fa_str(rec) or rec.get("plate_en", "—"))
        self.plate_big.setStyleSheet(f"color:{color};")
        if crop and not crop.isNull():
            self.plate_crop.setPixmap(QtGui.QPixmap.fromImage(crop).scaledToHeight(
                60, QtCore.Qt.SmoothTransformation))
        amb = " ⚠" if rec.get("ambiguous_region") else ""
        self.fields.setText(
            f"<b><span dir='ltr'>{plate_en_str(rec)}</span></b><br>"
            f"<span style='color:{color}'>● {rec.get('type','')}</span> "
            f"· {rec.get('type_fa','')}<br>"
            f"{prov_city(rec)}{amb}<br>"
            f"<span style='color:#8b93a7'>color {rec.get('color','')} · "
            f"conf {rec.get('best_confidence',0)} · "
            f"{(rec.get('last_seen','') or '').replace('T',' ')}</span>")
        if is_new:
            it = QtWidgets.QListWidgetItem(
                LRM + f"{(rec.get('last_seen','') or '').replace('T',' ')}   "
                f"{plate_en_str(rec)}   ·  {rec.get('type','')}  ·  "
                f"{rec.get('province','')}")
            it.setForeground(QtGui.QColor(color))
            self.feed.insertItem(0, it)
            while self.feed.count() > 100:
                self.feed.takeItem(self.feed.count() - 1)


# --------------------------------------------------------------------------- #
#  HISTORY tab
# --------------------------------------------------------------------------- #
class HistoryTab(QtWidgets.QWidget):
    COLS = ["", "Plate", "Type", "Province / City", "Last seen", "Seen"]

    def __init__(self, cfg: Config, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.thumb = cfg.gui.thumbnail_size
        self._row_for_key: dict[str, int] = {}

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10); outer.setSpacing(8)

        # top bar: search + count + refresh
        top = QtWidgets.QHBoxLayout()
        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText("Search plate, province, type…")
        self.search.textChanged.connect(self._apply_filter)
        self.count = QtWidgets.QLabel("0 plates")
        self.count.setStyleSheet("color:#8b93a7;")
        refresh = QtWidgets.QPushButton("Refresh"); refresh.setObjectName("ghost")
        refresh.clicked.connect(self.reload)
        top.addWidget(self.search, 1); top.addWidget(self.count)
        top.addWidget(refresh)
        outer.addLayout(top)

        # table + detail
        self.table = QtWidgets.QTableWidget(0, len(self.COLS))
        self.table.setHorizontalHeaderLabels(self.COLS)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setDefaultSectionSize(self.thumb + 6)
        self.table.setColumnWidth(0, self.thumb + 6)
        self.table.setIconSize(QtCore.QSize(self.thumb, self.thumb))
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        hdr.setSectionResizeMode(3, QtWidgets.QHeaderView.Stretch)
        self.table.itemSelectionChanged.connect(self._on_select)

        self.detail = DetailPanel(self.thumb)

        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        split.addWidget(self.table)
        split.addWidget(self.detail)
        split.setSizes([720, 430])
        outer.addWidget(split, 1)

        self.reload()

    # ---- data ----
    def _fresh_store(self) -> PlateStore:
        return PlateStore(json_path=self.cfg.storage.json_path,
                          images_dir=self.cfg.storage.images_dir)

    def reload(self) -> None:
        self.table.setRowCount(0)
        self._row_for_key.clear()
        for rec in self._fresh_store().all_records():
            self.upsert(rec, prepend=False)
        self.count.setText(f"{len(self._row_for_key)} plates")

    def _thumb_icon(self, rec: dict):
        s = rec.get("sightings", [])
        path = resolve_image(s[-1].get("plate_image")) if s else None
        if not path and s:
            path = resolve_image(s[-1].get("image"))
        if not path:
            return None
        return QtGui.QIcon(QtGui.QPixmap(str(path)).scaled(
            self.thumb, self.thumb, QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation))

    def upsert(self, rec: dict, prepend: bool = True) -> None:
        key = rec.get("key")
        plate_en = plate_en_str(rec)
        # Persian plate in physical plate order (two digits LEFT, region code
        # RIGHT). plate_fa_str prefixes each token with an LRM so the bidi
        # algorithm can't reorder the groups around the strong-RTL letter/ایران.
        plate_disp = plate_fa_str(rec) or plate_en
        values = ["", plate_disp, rec.get("type", ""),
                  prov_city(rec), (rec.get("last_seen", "") or "").replace("T", " "),
                  str(rec.get("count", 0))]
        if key in self._row_for_key:
            row = self._row_for_key[key]
            for c, v in enumerate(values):
                if c:
                    self.table.item(row, c).setText(v)
        else:
            row = 0 if prepend else self.table.rowCount()
            self.table.insertRow(row)
            if prepend:
                self._row_for_key = {k: r + 1 for k, r in self._row_for_key.items()}
            color = CATEGORY_HEX.get(rec.get("category", "unknown"), "#95a5a6")
            for c, v in enumerate(values):
                item = QtWidgets.QTableWidgetItem(v)
                item.setData(QtCore.Qt.UserRole, key)
                if c == 1:
                    f = item.font(); f.setBold(True); item.setFont(f)
                    item.setTextAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
                    item.setData(QtCore.Qt.UserRole + 1, plate_en)  # keep EN searchable
                if c == 2:
                    item.setForeground(QtGui.QColor(color))
                self.table.setItem(row, c, item)
            self._row_for_key[key] = row
        icon = self._thumb_icon(rec)
        if icon is not None:
            self.table.item(row, 0).setIcon(icon)
        self.count.setText(f"{len(self._row_for_key)} plates")

    def _apply_filter(self, text: str) -> None:
        t = text.strip().lower()
        for row in range(self.table.rowCount()):
            hay = " ".join(self.table.item(row, c).text().lower()
                           for c in range(1, len(self.COLS)))
            en = self.table.item(row, 1).data(QtCore.Qt.UserRole + 1) or ""
            self.table.setRowHidden(row, t not in (hay + " " + str(en).lower()))

    def _on_select(self) -> None:
        items = self.table.selectedItems()
        if not items:
            return
        rec = self._fresh_store().get(items[0].data(QtCore.Qt.UserRole))
        if rec:
            self.detail.show_record(rec)


# --------------------------------------------------------------------------- #
#  detail panel (history)
# --------------------------------------------------------------------------- #
class DetailPanel(QtWidgets.QFrame):
    def __init__(self, thumb_size: int = 96, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(14, 14, 14, 14); lay.setSpacing(10)

        lay.addWidget(_section("Car photo"))
        self.car_img = QtWidgets.QLabel("Select a plate")
        self.car_img.setAlignment(QtCore.Qt.AlignCenter)
        self.car_img.setMinimumHeight(220)
        self.car_img.setStyleSheet("background:#15161b; color:#667; border-radius:8px;")
        lay.addWidget(self.car_img, 3)

        self.plate_img = QtWidgets.QLabel()
        self.plate_img.setAlignment(QtCore.Qt.AlignCenter)
        self.plate_img.setFixedHeight(64)
        self.plate_img.setStyleSheet("background:#15161b; border-radius:6px;")
        lay.addWidget(self.plate_img)

        self.info = QtWidgets.QTextEdit(); self.info.setReadOnly(True)
        self.info.setMaximumHeight(220)
        lay.addWidget(self.info, 2)

    def show_record(self, rec: dict) -> None:
        s = rec.get("sightings", [])
        last = s[-1] if s else {}
        car = resolve_image(last.get("image"))
        plate = resolve_image(last.get("plate_image"))
        if car:
            self.car_img.setPixmap(QtGui.QPixmap(str(car)).scaled(
                self.car_img.width(), self.car_img.height(),
                QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        else:
            self.car_img.setText("(no car photo)")
        if plate:
            self.plate_img.setPixmap(QtGui.QPixmap(str(plate)).scaledToHeight(
                56, QtCore.Qt.SmoothTransformation))
        else:
            self.plate_img.clear()

        color = CATEGORY_HEX.get(rec.get("category", "unknown"), "#95a5a6")
        amb = "  ⚠ ambiguous region" if rec.get("ambiguous_region") else ""
        cities = ", ".join(rec.get("cities", []))
        self.info.setHtml(f"""
        <table cellspacing=5>
        <tr><td>Plate</td><td dir="ltr" style="font-size:17px;color:{color}">
            <b>{plate_fa_str(rec)}</b></td></tr>
        <tr><td>Plate (EN)</td><td dir="ltr"><b>{plate_en_str(rec)}</b></td></tr>
        <tr><td>Type</td><td>{rec.get('type','')} — {rec.get('type_fa','')}</td></tr>
        <tr><td>Color</td><td>{rec.get('color','')}</td></tr>
        <tr><td>Province</td><td>{rec.get('province','')} ({rec.get('province_fa','')}){amb}</td></tr>
        <tr><td>City</td><td>{rec.get('city','')}</td></tr>
        <tr><td>Other cities</td><td>{cities}</td></tr>
        <tr><td>Region code</td><td>{rec.get('region_code','')}</td></tr>
        <tr><td>First seen</td><td>{rec.get('first_seen','')}</td></tr>
        <tr><td>Last seen</td><td>{rec.get('last_seen','')}</td></tr>
        <tr><td>Times seen</td><td>{rec.get('count',0)}</td></tr>
        <tr><td>Best conf.</td><td>{rec.get('best_confidence',0)}</td></tr>
        </table>""")


# --------------------------------------------------------------------------- #
#  main window
# --------------------------------------------------------------------------- #
class HotspotDialog(QtWidgets.QDialog):
    """Startup gate: make sure the phone and PC share a network BEFORE we read
    the IP / start the server / build the QR (the PC's IP changes once it joins
    the hotspot, so this must happen first)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Connect to the same network")
        self.setModal(True)
        self.setMinimumWidth(460)
        root = QtWidgets.QVBoxLayout(self)
        root.setSpacing(12)

        title = QtWidgets.QLabel("📶  Put your phone and PC on the same network")
        title.setStyleSheet("font-size:16px; font-weight:600;")
        title.setAlignment(QtCore.Qt.AlignCenter)
        root.addWidget(title)

        steps = QtWidgets.QLabel(
            "Do this <b>before</b> the connection link is created:<br><br>"
            "1.  Turn ON your phone's <b>Wi-Fi hotspot</b>.<br>"
            "2.  Connect <b>this PC</b> to that hotspot.<br>"
            "&nbsp;&nbsp;&nbsp;&nbsp;(Or just make sure the phone and PC are on the "
            "same Wi-Fi.)<br><br>"
            "When you're connected, click <b>Done</b> — only then is the page and "
            "QR code generated, using the current network's address.")
        steps.setWordWrap(True)
        steps.setStyleSheet("color:#c5cad6; font-size:13px;")
        root.addWidget(steps)

        btns = QtWidgets.QHBoxLayout()
        btns.addStretch(1)
        done = QtWidgets.QPushButton("Done")
        done.setDefault(True)
        done.setMinimumWidth(120)
        done.clicked.connect(self.accept)
        btns.addWidget(done)
        root.addLayout(btns)

    # closing the window (X / Esc) counts as Done — the prompt was acknowledged
    def reject(self) -> None:
        self.accept()


class QrConnectDialog(QtWidgets.QDialog):
    """Shows the QR code + URL + steps for connecting a phone camera, and
    live-updates a 'establishing / waiting / connected' status. The QR is
    rendered lazily because in tunnel mode the public URL takes a few seconds."""

    def __init__(self, camera, parent=None):
        super().__init__(parent)
        self.camera = camera
        self.setWindowTitle("Connect phone camera")
        self.setMinimumWidth(420)
        self._elapsed = 0.0
        self._qr_rendered = False
        tunnel = bool(getattr(camera, "tunnel", False))
        self._was_tunnel = tunnel

        root = QtWidgets.QVBoxLayout(self)
        root.setSpacing(10)

        title = QtWidgets.QLabel("📱  Scan to connect your phone camera")
        title.setStyleSheet("font-size:16px; font-weight:600;")
        title.setAlignment(QtCore.Qt.AlignCenter)
        root.addWidget(title)

        self.qr_label = QtWidgets.QLabel("…")
        self.qr_label.setAlignment(QtCore.Qt.AlignCenter)
        self.qr_label.setMinimumSize(300, 300)
        root.addWidget(self.qr_label)

        self.url_label = QtWidgets.QLabel("")
        self.url_label.setAlignment(QtCore.Qt.AlignCenter)
        self.url_label.setWordWrap(True)
        self.url_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.url_label.setStyleSheet("font-family:Consolas,monospace; color:#9fc3ff;")
        root.addWidget(self.url_label)

        if tunnel:
            steps_txt = (
                "1.  Open your phone's <b>Camera</b> app, point it at the QR.<br>"
                "2.  Tap the link, then tap <b>Start camera → Allow</b>.<br>"
                "<span style='color:#8b93a7'>Works on any Wi-Fi or mobile data — "
                "the PC just needs internet (e.g. your phone's hotspot).</span>")
        else:
            steps_txt = (
                "1.  Phone <b>hotspot ON</b>; make sure this laptop is joined to "
                "it (then relaunch if the address looks wrong).<br>"
                "2.  Open your phone's <b>Camera</b>, point it at the QR, tap the "
                "link. If it warns “not secure”, tap <b>Advanced → Proceed</b>.<br>"
                "3.  Tap <b>Start camera → Allow</b>.<br>"
                "<span style='color:#8b93a7'>Can't connect the first time? "
                "Run <b>allow_phone_camera.bat</b> once (opens the firewall).</span>")
        steps = QtWidgets.QLabel(steps_txt)
        steps.setWordWrap(True)
        steps.setStyleSheet("color:#c5cad6; font-size:12.5px;")
        root.addWidget(steps)

        self.status = QtWidgets.QLabel("")
        self.status.setAlignment(QtCore.Qt.AlignCenter)
        root.addWidget(self.status)

        btns = QtWidgets.QHBoxLayout()
        btns.addStretch(1)
        self.btn = QtWidgets.QPushButton("Hide")
        self.btn.clicked.connect(self.accept)
        btns.addWidget(self.btn)
        root.addLayout(btns)

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._poll)
        self._timer.start(600)
        self._poll()

    def _set_status(self, text: str, color: str) -> None:
        self.status.setText(text)
        self.status.setStyleSheet(f"font-size:14px; color:{color}; padding:6px;")

    def _render_qr(self, url: str) -> None:
        try:
            buf = io.BytesIO()
            self.camera.qr_image().save(buf, format="PNG")
            pix = QtGui.QPixmap()
            pix.loadFromData(buf.getvalue(), "PNG")
            self.qr_label.setPixmap(pix.scaled(
                300, 300, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
            self.url_label.setText(url)
            self._qr_rendered = True
        except Exception as exc:
            self.qr_label.setText(f"(QR unavailable: {exc})")

    def _poll(self) -> None:
        self._elapsed += 0.6
        cam = self.camera
        url = cam.connect_url
        if url and not self._qr_rendered:
            self._render_qr(url)

        fell_back = self._was_tunnel and not getattr(cam, "tunnel", False)
        if cam.connected:
            self._set_status("●  Connected — streaming ✓", "#3ecf6a")
            self.btn.setText("Done")
        elif url is None:
            if getattr(cam, "tunnel_status", "") == "error":
                self._set_status("⚠  Couldn't start the internet link "
                                 "(see console)", "#e74c3c")
            else:
                self._set_status("●  Establishing secure link… "
                                 "(a few seconds)", "#e0a93b")
        elif fell_back:
            self._set_status("⚠  Internet link blocked on this network — using "
                             "same-Wi-Fi (one-time “proceed” tap)", "#e0a93b")
        elif (not getattr(cam, "relay", False)
              and not getattr(cam, "server_bound", True) and self._elapsed > 3):
            self._set_status("⚠  Server didn't start — check the port", "#e74c3c")
        else:
            self._set_status("●  Waiting for your phone…", "#e0a93b")

    def closeEvent(self, event) -> None:
        self._timer.stop()
        event.accept()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self._browser_mode = str(getattr(cfg.camera, "source", "")).strip().lower() == "browser"
        self.setWindowTitle(cfg.gui.window_title)
        self.resize(1300, 800)
        self._rotation = getattr(cfg.camera, "rotate", 0)

        self._build_toolbar()

        self.live = LiveTab()
        self.history = HistoryTab(cfg)
        tabs = QtWidgets.QTabWidget()
        tabs.addTab(self.live, "  ●  Live  ")
        tabs.addTab(self.history, "  ▦  History  ")
        self.setCentralWidget(tabs)
        self.statusBar().showMessage("Starting…")

        self.worker = PipelineWorker(cfg)
        self.worker.frameReady.connect(self.live.set_frame)
        self.worker.plateEvent.connect(self._on_plate)
        self.worker.stats.connect(self._on_stats)
        self.worker.status.connect(self.statusBar().showMessage)
        self.worker.failed.connect(self._on_failed)

        self._qr_dialog = None
        self._camera_started = False
        browser = getattr(cfg.camera, "browser", None)
        self._relay_mode = bool(getattr(browser, "relay_url", None))

        if self._browser_mode and not self._relay_mode:
            # Gate everything behind the hotspot prompt: no server, no IP/URL, no
            # QR, no model load happens until the user confirms the network.
            self.statusBar().showMessage("Waiting — connect phone & PC to the same network…")
            QtCore.QTimer.singleShot(250, self._prompt_hotspot)
        else:
            self.worker.start()
            self._camera_started = True
            if self._browser_mode:
                QtCore.QTimer.singleShot(300, self._show_qr_dialog)

    def _prompt_hotspot(self) -> None:
        """Block until the user has joined the shared network and clicks Done —
        only then do we resolve the IP, start the server, and build the QR."""
        if self._camera_started:
            return
        HotspotDialog(self).exec()
        self._begin_camera()

    def _begin_camera(self) -> None:
        if self._camera_started:
            return
        self._camera_started = True
        self.statusBar().showMessage("Starting camera server…")
        self.worker.start()                       # now resolves IP + serves + QR
        QtCore.QTimer.singleShot(400, self._show_qr_dialog)

    def _show_qr_dialog(self) -> None:
        if not self._camera_started:
            self._prompt_hotspot()
            return
        cam = getattr(self.worker.pipeline, "camera", None)
        if cam is None or not hasattr(cam, "qr_image"):
            return
        if self._qr_dialog is not None:
            self._qr_dialog.close()
        self._qr_dialog = QrConnectDialog(cam, self)
        self._qr_dialog.show()

    # ---- toolbar ----
    def _build_toolbar(self) -> None:
        tb = self.addToolBar("Main")
        tb.setMovable(False)
        tb.setIconSize(QtCore.QSize(16, 16))

        self.lbl_conn = QtWidgets.QLabel("●  offline")
        self.lbl_fps = QtWidgets.QLabel("fps —")
        for w in (self.lbl_conn, self.lbl_fps):
            tb.addWidget(w)

        self.lbl_link = None
        if self._browser_mode:
            self.lbl_link = QtWidgets.QLabel("⏳ linking…")
            self.lbl_link.setStyleSheet("color:#e0a93b; padding:0 8px;")
            self.lbl_link.setToolTip("How the phone is connecting")
            tb.addWidget(self.lbl_link)

        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                             QtWidgets.QSizePolicy.Preferred)
        tb.addWidget(spacer)

        rl = QtWidgets.QPushButton("⟲"); rl.setObjectName("ghost")
        rl.setToolTip("Rotate left 90°")
        rl.clicked.connect(lambda: self._rotate(-90))
        rr = QtWidgets.QPushButton("⟳"); rr.setObjectName("ghost")
        rr.setToolTip("Rotate right 90°")
        rr.clicked.connect(lambda: self._rotate(90))
        self.lbl_rot = QtWidgets.QLabel(f"{self._rotation}°")
        tb.addWidget(QtWidgets.QLabel("Rotate "))
        tb.addWidget(rl); tb.addWidget(rr); tb.addWidget(self.lbl_rot)

        if self._browser_mode:
            phone = QtWidgets.QPushButton("📱 Connect phone")
            phone.setObjectName("ghost")
            phone.setToolTip("Show the QR code to connect your phone camera")
            phone.clicked.connect(self._show_qr_dialog)
            tb.addWidget(phone)

            # The timer fires once the event loop runs, by which point the
            # worker (created later in __init__) exists.
            self._link_timer = QtCore.QTimer(self)
            self._link_timer.timeout.connect(self._update_link_indicator)
            self._link_timer.start(1000)

    def _update_link_indicator(self) -> None:
        """Reflect how the phone is connecting: public internet link, still
        establishing, or fallen back to same-Wi-Fi."""
        if self.lbl_link is None:
            return
        worker = getattr(self, "worker", None)
        cam = getattr(getattr(worker, "pipeline", None), "camera", None)
        status = getattr(cam, "tunnel_status", None)
        if status == "ready":
            text, color = "🌐 internet link ✓", "#3ecf6a"
        elif status == "starting":
            text, color = "⏳ linking…", "#e0a93b"
        else:  # 'lan' — configured for same-Wi-Fi, or fell back from a blocked tunnel
            blocked = bool(getattr(cam, "_tunnel_error", None))
            text = "📶 same-Wi-Fi (link blocked)" if blocked else "📶 same-Wi-Fi"
            color = "#e0a93b"
        self.lbl_link.setText(text)
        self.lbl_link.setStyleSheet(f"color:{color}; padding:0 8px;")

    def _rotate(self, delta: int) -> None:
        self._rotation = self.worker.rotate_by(delta)
        self.lbl_rot.setText(f"{self._rotation}°")

    # ---- signals ----
    @QtCore.Slot(dict, bool, QtGui.QImage)
    def _on_plate(self, rec: dict, is_new: bool, crop: QtGui.QImage) -> None:
        self.live.show_event(rec, crop, is_new)
        self.history.upsert(rec, prepend=True)

    @QtCore.Slot(dict)
    def _on_stats(self, s: dict) -> None:
        connected = s.get("connected", False)
        self.lbl_conn.setText("●  online" if connected else "●  offline")
        self.lbl_conn.setStyleSheet(
            f"color:{'#3ecf6a' if connected else '#e74c3c'}; padding:0 8px;")
        if "fps" in s:
            self.lbl_fps.setText(f"fps {s['fps']:.0f} · {s.get('infer_ms',0):.0f}ms")

    @QtCore.Slot(str)
    def _on_failed(self, msg: str) -> None:
        QtWidgets.QMessageBox.critical(self, "PelakYab", msg)
        self.statusBar().showMessage(msg)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        try:
            self.worker.stop()
        finally:
            event.accept()


def run_gui(cfg: Config | None = None) -> int:
    cfg = cfg or load_config()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLESHEET)
    win = MainWindow(cfg)
    win.show()

    # In browser mode, also surface the connect URL + an ASCII QR in the console.
    if getattr(win, "_browser_mode", False):
        cam = getattr(win.worker.pipeline, "camera", None)
        if cam is not None and hasattr(cam, "qr_terminal"):
            print("\n  Phone camera — scan this QR, or open:", cam.connect_url, "\n")
            try:                                  # block-char QR needs a UTF-8 console
                sys.stdout.reconfigure(encoding="utf-8", errors="replace")
                print(cam.qr_terminal())
            except Exception:
                pass

    return app.exec()
