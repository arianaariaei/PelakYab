# models/

These two YOLO weight files are **already downloaded** and wired to
`../config.yaml`:

- `plate_detector.pt`  — detects the license-plate region in a frame
- `char_recognizer.pt` — detects + classifies the plate glyphs

Source: pretrained Iranian models from the ANPR-YOLOv8 project
(<https://github.com/barzansaeedpour/ANPR-YOLOv8>), fetched via
`../scripts/download_models.py` guidance. The `.pt` files themselves are not
committed to git (see `../.gitignore`); re-download or replace them with your
own fine-tuned weights any time.

To push accuracy further, fine-tune on IR-LPR/Iranis with
`../scripts/train_plate_detector.py` and `../scripts/train_char_recognizer.py`
and drop the resulting `best.pt` files here under the same names.
