# PelakYab · پلاک‌یاب

**Real-time Iranian (Persian) license-plate recognition** from a phone camera,
with full plate **type / province / city** decoding, JSON storage, and a desktop
GUI that shows each car's photo, plate, and the date/time it was seen.

> *PelakYab* (پلاک‌یاب) = "plate finder".
---

## How it works

```
 ┌─────────────┐   Wi-Fi/MJPEG    ┌──────────────────────────────────────────┐
 │  Android     │ ───────────────► │  PC (your GPU)                            │
 │  phone cam   │                  │                                           │
 │  (browser)   │                  │  temporal voting → stable plate reads     │
 └─────────────┘   HTTP            │  YOLO plate detector                      │
                                   │  YOLO character recognizer (8 glyphs)     │
                                   │  color classifier → plate type            │
                                   │  letter → type   │  region code → province│
                                   │  JSON store  +  saved car/plate images    │
                                   │  PySide6 GUI (live view + history + detail)│
                                   └──────────────────────────────────────────┘
```

The phone is just the camera. All detection/recognition runs on your PC so you
can use large, accurate models and a real GPU.

---

## Features

- **Two-stage YOLO pipeline** — plate detection → per-character detection,
  robust to the fixed 8-glyph Iranian layout.
- **Full decoding** — Persian letter → plate *type* (private, taxi, public,
  police, IRGC, army, government, diplomatic, disabled, …); region code →
  *province* and *city*; background color confirms/disambiguates the type.
- **Temporal voting** — reads are voted across frames so a plate held in view is
  logged once, with a dedicated CNN re-scoring the noisy letter glyph.
- **De-duplication** — the same car seen repeatedly updates one record (bumps
  `count`, `last_seen`) instead of spamming new rows.
- **JSON storage** — `data/plates.json`, plus saved wide car photos and plate
  crops under `data/images/<plate>/`.
- **Desktop GUI** — live annotated video, a searchable table of all detected
  plates with thumbnails, and a detail panel with the car photo + every field.
- **Pluggable models** — point `config.yaml` at any Iranian YOLO weights; class
  labels are normalized so it works regardless of naming.

---

## 1. Install

```powershell
# clone the repo — the pretrained model weights are included, so it
# runs out of the box with NO training required
git clone https://github.com/arianaariaei/PelakYab.git
cd PelakYab

# (recommended) create a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# install the CUDA build of PyTorch FIRST (match your CUDA), e.g. CUDA 12.4:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# then the rest
pip install -r requirements.txt
```

Verify:

```powershell
python main.py --check
```

## 2. Models — included in this repo

The pretrained weight files are **committed to this repository** (under `models/`),
so a fresh clone runs without downloading or training anything. They are based on
the [ANPR-YOLOv8](https://github.com/barzansaeedpour/ANPR-YOLOv8) Iranian-plate models:

| file | role |
|------|------|
| `models/plate_detector.pt`  | finds the plate rectangle in a frame |
| `models/char_recognizer.pt` | detects + classifies the plate glyphs (digits + Persian letters) |
| `models/char_classifier.pt` | second-stage CNN that re-scores the noisy letter glyph |

They are wired to `config.yaml` and ready to run. Their classes are already
canonical Persian glyphs / ASCII digits, so no normalization edits are needed.

**Optional — higher production accuracy (train/fine-tune on your GPU):**

```powershell
python scripts/download_models.py --base yolov8s      # cache a base to fine-tune
python scripts/train_plate_detector.py  --data path\to\plate_data.yaml  --device 0
python scripts/train_char_recognizer.py --data path\to\char_data.yaml   --device 0
```

Datasets:
- **IR-LPR** — 20,967 Iranian car images with plate **and** per-character boxes:
  <https://github.com/mut-deep/IR-LPR>
- **Iranis** — ~83k Persian plate-character crops (28 classes) to balance rare
  letters.

> After training the char model, print its class names
> (`print(YOLO('models/char_recognizer.pt').names)`). If they aren't already
> canonical Persian glyphs/ascii digits, add the mapping to
> `LETTER_NORMALIZATION` / `DIGIT_NORMALIZATION` in
> [`pelakyab/data/plate_types.py`](pelakyab/data/plate_types.py).


## 3. Connect the phone (no app)

With `camera.source: "browser"` (the default), the PC shows a **QR code**; scan
it with the phone's normal camera and the phone streams its camera in through the
**browser** — nothing to install. Works on Android and iPhone. Run
`python main.py` and the QR appears in the window (and console; `python main.py
--qr` shows it too). 

### Same Wi-Fi — offline (`camera.browser.tunnel: false`)

Phone and laptop on one network (e.g. the phone's hotspot), fully offline. Uses a
self-signed certificate, so there's a little one-time setup:

- One-time **Advanced → Proceed** tap on the phone (the cert isn't "official").
- Double-click **`allow_phone_camera.bat`** once (opens the firewall port).


## 4. Run

```powershell
python main.py                 # GUI (default)
python main.py --headless      # no GUI; prints each new plate
python main.py --check         # environment / model / camera check
python main.py --qr            # print/save the phone-connect QR (browser mode)
python scripts/test_image.py some_car.jpg --save out.jpg   # one still image
python scripts/selftest.py     # pure-Python decode test (no GPU needed)
```

---

## Configuration (`config.yaml`)

Key knobs:

- `camera.source` — `"browser"` (QR/phone), or `"0"` for a webcam.
- `recognition.stabilize` / `two_stage_letter` — temporal voting + CNN letter re-scoring.
- `detection.device` — `cuda:0` or `cpu`; `plate_imgsz` large (960) helps catch
  small/distant plates.
- `storage.dedup_cooldown` — seconds before the same plate logs a new sighting.
- `gui.persian_font` — any Persian-capable `.ttf` (default Windows Tahoma) used
  to draw plate text on the live frame.

## Data format

`data/plates.json` is keyed by normalized plate, e.g.:

```json
{
  "12ب345-11": {
    "plate_en": "12 ب 345 - 11",
    "plate_fa": "12 ب 345 ایران 11",
    "letter": "ب", "type": "Private", "type_fa": "شخصی", "color": "white",
    "region_code": "11", "province": "Tehran", "city": "Tehran",
    "first_seen": "2026-06-07T21:14:03", "last_seen": "2026-06-07T21:48:10",
    "count": 4, "best_confidence": 0.93,
    "sightings": [
      {"time": "2026-06-07T21:14:03", "confidence": 0.91,
       "image": "data/images/12ب345-11/20260607_211403_full.jpg",
       "plate_image": "data/images/12ب345-11/20260607_211403_plate.jpg"}
    ]
  }
}
```

## Project layout

```
PelakYab/
├── main.py                     # entry point (GUI / headless / check)
├── config.yaml                 # all settings
├── requirements.txt
├── models/                     # put plate_detector.pt + char_recognizer.pt here
├── data/                       # created at runtime: plates.json + images/
├── pelakyab/
│   ├── config.py               # yaml loader
│   ├── camera/ip_webcam.py     # IP Webcam stream backend
│   ├── camera/browser_cam.py   # app-free phone camera (QR + WebSocket)
│   ├── stabilizer.py           # vote plate reads across frames
│   ├── utils/preprocess.py     # optional CLAHE / deskew
│   ├── utils/draw.py           # Persian (RTL) text on frames
│   ├── detection/
│   │   ├── plate_detector.py   # YOLO stage 1
│   │   ├── char_recognizer.py  # YOLO stage 2 (+ label normalization)
│   │   └── color_classifier.py # HSV background-color → type
│   ├── data/
│   │   ├── provinces.py        # region code → province/city table
│   │   ├── plate_types.py      # letter/color → type + label normalization
│   │   └── plate_parser.py     # tokens → structured Plate
│   ├── storage/store.py        # JSON store + image saving + de-dup
│   ├── pipeline.py             # orchestrates everything
│   └── gui/app.py              # PySide6 GUI
└── scripts/
    ├── download_models.py
    ├── train_plate_detector.py
    ├── train_char_recognizer.py
    ├── test_image.py
    └── selftest.py
```

## Accuracy tips

- Mount the camera so plates are roughly frontal and ≥ ~80 px wide.
- Keep `plate_imgsz` ≥ 960 for distant plates; raise to 1280 if needed.
- Fine-tune both models on IR-LPR for your camera/lighting — this is the single
  biggest accuracy win.
- For motorcycle / 2-row plates, set `two_row=True` when constructing
  `CharRecognizer`.

## Notes / caveats

- The province/city table (`pelakyab/data/provinces.py`) is compiled from the
  Ghabzino guide + the common public NAJA list. A few codes are reused across
  provinces in different listings (notably `32`, and the Tehran/Alborz split
  codes `21/38/68/78`); these are flagged `ambiguous` and easy to edit.
- This is for **authorized** use (your own gate/lot/research). Respect local law
  and privacy when recording vehicles.

## Credits / sources

- Plate types & province/city codes: Ghabzino Iranian plate guide.
- Datasets/models: IR-LPR, Iranis, ANPR-YOLOv8.
- Detection: Ultralytics YOLO.
```
