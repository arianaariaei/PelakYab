"""Temporal plate stabilizer — vote a plate's reading across frames.

The detector/recognizer run per frame and jitter, especially on the LETTER.
Without smoothing the pipeline stores a new (often misread) car on every frame.

This tracks each plate across frames and:
  * groups reads of the same DIGITS together (even across brief look-aways), so a
    plate isn't re-logged every time it re-enters view;
  * majority-votes the DIGITS over a short recent window (confidence-weighted);
  * votes the LETTER over the plate's WHOLE lifetime (the letter is the noisiest
    glyph, so it needs the longest memory to settle);
  * emits one stable, voted plate per car.
"""
from __future__ import annotations

import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Optional

from .data.plate_types import LETTER_TYPE


def _digit_sig(tokens) -> str:
    return "".join(t for t in tokens if t.isdigit())


def _letter_of(tokens, confs):
    for i, t in enumerate(tokens):
        if t in LETTER_TYPE:
            return t, (confs[i] if i < len(confs) and confs[i] else 0.5)
    return None, 0.0


@dataclass
class _Track:
    cx: float
    cy: float
    last_ts: float
    sig: str = ""
    reads: deque = field(default_factory=deque)        # (ts, tokens, confs)
    letter_votes: Counter = field(default_factory=Counter)
    color: Optional[str] = None
    crop=None
    conf: float = 0.0
    emitted_key: Optional[str] = None
    emitted_ts: float = 0.0


class PlateStabilizer:
    def __init__(self, window: float = 2.0, min_votes: int = 4,
                 match_dist: float = 0.12, emit_interval: float = 2.0,
                 retain: float = 20.0):
        self.window = window               # recent secs for DIGIT consensus
        self.min_votes = min_votes         # recent reads needed to emit
        self.match_dist = match_dist       # position match radius (frac of diag)
        self.emit_interval = emit_interval # min secs between re-emits of a key
        self.retain = retain               # keep a track this long after last seen
        self._tracks: list[_Track] = []

    def update(self, reads: list[dict], frame_shape, now: float | None = None) -> list[dict]:
        """Feed this frame's reads; return the plates confirmed right now.

        read: {cx, cy, tokens, confs, color, crop, conf}
        confirmed: {tokens, color, crop, conf}
        """
        now = time.monotonic() if now is None else now
        h, w = frame_shape[:2]
        diag = (w * w + h * h) ** 0.5 or 1.0

        for r in reads:
            sig = _digit_sig(r["tokens"])
            tr = self._match(r["cx"], r["cy"], diag, sig)
            if tr is None:
                tr = _Track(cx=r["cx"], cy=r["cy"], last_ts=now, sig=sig)
                self._tracks.append(tr)
            tr.cx, tr.cy, tr.last_ts = r["cx"], r["cy"], now
            if sig:
                tr.sig = sig
            tr.color, tr.crop, tr.conf = r.get("color"), r.get("crop"), r.get("conf", 0.0)
            confs = list(r.get("confs") or [])
            tr.reads.append((now, list(r["tokens"]), confs))
            lt, lc = _letter_of(r["tokens"], confs)
            if lt:
                tr.letter_votes[lt] += max(lc, 0.05)

        confirmed, alive = [], []
        for tr in self._tracks:
            while tr.reads and now - tr.reads[0][0] > self.window:
                tr.reads.popleft()
            if now - tr.last_ts > self.retain:
                continue                                  # track expired
            alive.append(tr)
            c = self._maybe_confirm(tr, now)
            if c is not None:
                confirmed.append(c)
        self._tracks = alive
        return confirmed

    def reset(self) -> None:
        self._tracks = []

    # ------------------------------------------------------------- internals
    def _match(self, cx: float, cy: float, diag: float, sig: str) -> Optional[_Track]:
        if sig:                                           # same plate digits -> same track
            for tr in self._tracks:
                if tr.sig and tr.sig == sig:
                    return tr
        best, best_d = None, self.match_dist * diag       # else nearest in view
        for tr in self._tracks:
            d = ((tr.cx - cx) ** 2 + (tr.cy - cy) ** 2) ** 0.5
            if d < best_d:
                best, best_d = tr, d
        return best

    def _consensus(self, tr: _Track) -> tuple[list[str], int]:
        reads = tr.reads
        lengths = Counter(len(tok) for _, tok, _ in reads)
        length = lengths.most_common(1)[0][0]
        same = [(tok, cf) for _, tok, cf in reads if len(tok) == length]
        out = []
        for i in range(length):
            votes: Counter = Counter()
            for tok, cf in same:
                votes[tok[i]] += cf[i] if i < len(cf) and cf[i] else 1.0
            out.append(votes.most_common(1)[0][0])
        # Override the letter slot with the lifetime letter vote (most robust).
        if tr.letter_votes:
            best_letter = tr.letter_votes.most_common(1)[0][0]
            for i, tok in enumerate(out):
                if tok in LETTER_TYPE:
                    out[i] = best_letter
                    break
        return out, len(same)

    def _maybe_confirm(self, tr: _Track, now: float) -> Optional[dict]:
        if not tr.reads:
            return None
        tokens, n = self._consensus(tr)
        if n < self.min_votes:
            return None
        key = "".join(tokens)
        if tr.emitted_key == key and (now - tr.emitted_ts) < self.emit_interval:
            return None
        tr.emitted_key, tr.emitted_ts = key, now
        return {"tokens": tokens, "color": tr.color, "crop": tr.crop, "conf": tr.conf}
