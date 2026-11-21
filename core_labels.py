"""
core_labels.py
==============
Turn SCAN Appendix B core-log PDFs (Cambridge Carbonates template, 1:50) into facies labels vs depth.

For each page image (scanned 300 dpi, or vector PDF rendered at 300 dpi):
  1. find the data area and the column boundaries (long vertical black lines)
  2. read the depth labels in the "Core depth m" column with Tesseract and fit depth = a + b * row
  3. per pixel row: lithology from the fill colour of the Lithology column,
     Dunham texture from the right edge of the "Carbonate texture" profile
  4. combine into facies classes and write intervals (top, base, facies) per well

Facies classes (FACIES):
  Lst_mud     limestone, mudstone or wackestone texture
  Lst_grain   limestone, packstone / grainstone / boundstone texture
  Lst_breccia limestone breccia (karst / collapse)
  Dolomite    dolomite or calcareous dolomite
  Sandstone   sandstone (rare in the Dinantian)
  Shale       mudstone / shale

Usage:
  python core_labels.py --pdf-dir "data/SCAN Dinantian Facies analysis and diagenetic evolution - Appendix B" \
                        --out data/core_facies.csv --qc-dir data/core_qc
Needs: tesseract (system), pytesseract, opencv-python, pymupdf, numpy, pandas.
On Windows install Tesseract from https://github.com/UB-Mannheim/tesseract/wiki and, if it is not on PATH,
set pytesseract.pytesseract.tesseract_cmd (see --tesseract).
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

TEXTURES = ["Mudstone", "Wackestone", "Packstone", "Grainstone", "Boundstone", "Breccia"]
LITH_COLOURS = {                       # RGB of the legend fills
    "Limestone": (72, 192, 240),
    "Dolomite": (144, 192, 48),
    "CalcDolomite": (0, 168, 120),
    "Sandstone": (240, 240, 96),
    "Shale": (150, 141, 0),            # mudstone / shale (olive)
    "None": (245, 245, 245),           # white: no core recovered / not described
}


# ----------------------------------------------------------------------------- page image
def page_image(pdf: Path) -> np.ndarray:
    """RGB image of the first page at ~300 dpi (embedded scan if the page is one image)."""
    import fitz
    doc = fitz.open(str(pdf))
    page = doc[0]
    imgs = page.get_images(full=True)
    if len(imgs) == 1 and not page.get_text().strip():
        pix = fitz.Pixmap(doc, imgs[0][0])
    else:
        pix = page.get_pixmap(dpi=300)
    if pix.n - pix.alpha >= 4:
        pix = fitz.Pixmap(fitz.csRGB, pix)
    a = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
    return a[:, :, :3].copy()


def _groups(xs, gap=3):
    out = []
    for x in xs:
        if out and x - out[-1][-1] <= gap:
            out[-1].append(x)
        else:
            out.append([x])
    return [int(np.mean(g)) for g in out]


def layout(gray):
    """Data rows (y0, y1) and column boundaries."""
    H, W = gray.shape
    dark = gray < 110
    rows = _groups(np.where(dark[:, : int(W * 0.6)].mean(1) > 0.8)[0], gap=6) + [H - 1]
    best = None
    for ya, yb in zip(rows[:-1], rows[1:]):          # data band = the band with the most column lines
        if yb - ya < 200:
            continue
        cols = _groups(np.where(dark[ya + 3:yb - 3].mean(0) > 0.85)[0])
        if best is None or len(cols) > len(best[2]):
            best = (ya + 3, yb - 3, cols)
    return best


# ----------------------------------------------------------------------------- depth
def depth_fit(gray, y0, y1, x0, x1):
    """Fit depth = a + b*y from OCR'd depth labels (robust to OCR errors)."""
    import pytesseract
    pts = []
    for ys in range(y0, y1, 2500):                   # tiles: Tesseract fails on very tall images
        ye = min(ys + 2600, y1)
        strip = cv2.resize(gray[ys:ye, x0 + 2:x1 - 2], None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        d = pytesseract.image_to_data(strip, config="--psm 11", output_type=pytesseract.Output.DICT)
        for t, top, h in zip(d["text"], d["top"], d["height"]):
            m = re.search(r"(\d{3,4}(?:\.\d+)?)", t.replace(",", "."))
            if m:
                pts.append((ys + (top + h / 2) / 2, float(m.group(1))))
    pts = np.array(sorted(set(pts)))
    if len(pts) < 2:
        raise ValueError("fewer than two depth labels read")
    best, best_n = None, 0
    for i in range(len(pts)):                       # RANSAC over label pairs
        for j in range(i + 1, len(pts)):
            dy, dz = pts[j, 0] - pts[i, 0], pts[j, 1] - pts[i, 1]
            if dz <= 0 or not 50 < dy / dz < 350:    # px per unit: ~240 for metres, ~72 for feet (1:50, 300 dpi)
                continue
            b = dz / dy
            a = pts[i, 1] - b * pts[i, 0]
            n = int((np.abs(a + b * pts[:, 0] - pts[:, 1]) < 0.15).sum())
            if n > best_n:
                best, best_n = (a, b), n
    if best is None or best_n < 2:
        raise ValueError(f"no consistent depth labels: {pts.tolist()}")
    inl = np.abs(best[0] + best[1] * pts[:, 0] - pts[:, 1]) < 0.15
    b, a = np.polyfit(pts[inl, 0], pts[inl, 1], 1)
    if 1 / b < 120:                                  # labels in feet -> metres
        a, b = a * 0.3048, b * 0.3048
    return a, b, int(inl.sum())


# ----------------------------------------------------------------------------- columns
def lithology_rows(rgb, y0, y1, x0, x1):
    names = list(LITH_COLOURS)
    ref = np.array([LITH_COLOURS[n] for n in names], float)
    reg = rgb[y0:y1, x0 + 8:x1 - 8].astype(float)
    out = []
    for row in reg:
        px = row[row.sum(1) > 250]                   # drop black symbol pixels
        if len(px) < 0.3 * len(row):
            out.append(None)
            continue
        c = np.median(px, 0)
        dist = np.sqrt(((ref - c) ** 2).sum(1))
        out.append(names[int(np.argmin(dist))] if dist.min() < 60 else None)
    # rolling mode over ~6 cm removes bed-boundary lines and symbol rows
    codes = pd.Series([names.index(o) if o else -1 for o in out])
    mode = codes.rolling(15, center=True, min_periods=1).apply(lambda v: np.bincount((v + 1).astype(int)).argmax() - 1)
    return [names[int(m)] if m >= 0 else None for m in mode]


def texture_rows(rgb, gray, y0, y1, x0, x1):
    """Dunham texture per row from the right edge of the texture profile."""
    reg = gray[y0:y1, x0:x1]
    # grey dashed guide lines separate the texture classes
    guide = _groups(np.where(((reg > 90) & (reg < 215)).mean(0) > 0.25)[0])
    w = x1 - x0
    guide = [g for g in guide if 0.3 * w < g < 0.97 * w][:5]
    if len(guide) < 5:                               # fall back to the template proportions
        guide = [int(f * w) for f in (0.373, 0.499, 0.623, 0.749, 0.876)]
    r, gch = rgb[y0:y1, x0:x1, 0].astype(int), rgb[y0:y1, x0:x1, 1].astype(int)
    black = (reg < 100) & ~(r - gch > 60)            # ignore red thin-section squares
    black[:, w - 4:] = False                         # column border
    full = black.mean(1) > 0.5                       # horizontal grid lines
    edge = np.array([np.where(b)[0].max() if b.any() and not f else -1 for b, f in zip(black, full)])
    s = pd.Series(np.where(edge < 0, np.nan, edge)).ffill().bfill()
    edge = s.rolling(9, center=True, min_periods=1).median().to_numpy()
    tex = np.searchsorted(guide, edge)               # 0..5
    return [TEXTURES[min(int(t), 5)] for t in tex]


def facies(lith, tex):
    if lith in (None, "None"):
        return None
    if lith in ("Dolomite", "CalcDolomite"):
        return "Dolomite"
    if lith in ("Sandstone", "Shale"):
        return lith
    if tex == "Breccia":
        return "Lst_breccia"
    if tex in ("Mudstone", "Wackestone"):
        return "Lst_mud"
    return "Lst_grain"


# ----------------------------------------------------------------------------- per file
def process(pdf: Path, qc_dir: Path | None = None):
    rgb = page_image(pdf)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    y0, y1, cols = layout(gray)
    if len(cols) < 6:  # noqa
        raise ValueError(f"unexpected layout, columns {cols[:8]}")
    dep, lit, tex = (cols[1], cols[2]), (cols[3], cols[4]), (cols[4], cols[5])
    a, b, n = depth_fit(gray, y0, y1, *dep)
    L = lithology_rows(rgb, y0, y1, *lit)
    T = texture_rows(rgb, gray, y0, y1, *tex)
    rows = pd.DataFrame({"y": np.arange(y0, y1), "depth": a + b * np.arange(y0, y1), "lith": L, "texture": T})
    rows["FACIES"] = [facies(l, t) for l, t in zip(rows.lith, rows.texture)]
    info = dict(file=pdf.name, depth_labels=n, m_per_px=b, top=rows.depth.min(), base=rows.depth.max())
    if qc_dir is not None:
        qc(rgb, rows, y0, y1, lit, tex, qc_dir / (pdf.stem + "_qc.png"))
    return rows, info


def qc(rgb, rows, y0, y1, lit, tex, path):
    """Side-by-side check: scanned columns next to the extracted facies."""
    colours = {"Lst_mud": (72, 192, 240), "Lst_grain": (0, 90, 200), "Lst_breccia": (230, 120, 40),
               "Dolomite": (144, 192, 48), "Sandstone": (240, 220, 60), "Shale": (150, 141, 0),
               None: (255, 255, 255)}
    crop = rgb[y0:y1, lit[0]:tex[1]]
    bar = np.array([colours[f if isinstance(f, str) else None] for f in rows.FACIES], np.uint8)[:, None, :].repeat(80, 1)
    img = np.hstack([crop, np.full((len(bar), 10, 3), 255, np.uint8), bar])
    step = max(1, len(img) // 3000)
    cv2.imwrite(str(path), cv2.cvtColor(img[::step, ::1], cv2.COLOR_RGB2BGR))


def to_intervals(rows, well):
    r = rows.dropna(subset=["FACIES"]).copy()
    gap = r.depth.diff().abs() > 0.02                # uncored / undescribed gap breaks an interval
    r["blk"] = ((r.FACIES != r.FACIES.shift()) | gap).cumsum()
    iv = r.groupby("blk").agg(top=("depth", "min"), base=("depth", "max"), FACIES=("FACIES", "first"))
    iv = iv[iv.base - iv.top >= 0.05]                # drop slivers < 5 cm
    iv.insert(0, "WELL", well)
    return iv.reset_index(drop=True)


def well_name(stem):
    return re.match(r"([A-Z0-9]+-\d+(?:-S\d)?)", stem.replace(" ", "")).group(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf-dir", required=True)
    ap.add_argument("--out", default="data/core_facies.csv")
    ap.add_argument("--qc-dir", default="data/core_qc")
    ap.add_argument("--skip", nargs="*", default=["Core 7"], help="file-name parts to skip (O18-01 Core 7 is Silurian)")
    ap.add_argument("--tesseract", default=None, help="path to tesseract.exe if not on PATH")
    a = ap.parse_args()
    if a.tesseract:
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = a.tesseract
    qc_dir = Path(a.qc_dir); qc_dir.mkdir(parents=True, exist_ok=True)
    ivs, infos = [], []
    for pdf in sorted(Path(a.pdf_dir).glob("*.pdf")):
        if any(s in pdf.name for s in a.skip):
            continue
        try:
            rows, info = process(pdf, qc_dir)
            ivs.append(to_intervals(rows, well_name(pdf.stem)))
            info["status"] = "ok"
        except Exception as e:                      # noqa: BLE001
            info = dict(file=pdf.name, status=f"skipped: {e}")
        infos.append(info)
        print(info)
    out = pd.concat(ivs, ignore_index=True)
    out.round(3).to_csv(a.out, index=False)
    pd.DataFrame(infos).to_csv(Path(a.out).with_name("core_facies_log.csv"), index=False)
    out["thick"] = out.base - out.top
    print(out.groupby(["WELL", "FACIES"]).thick.sum().round(1).unstack(fill_value=0))


if __name__ == "__main__":
    main()
