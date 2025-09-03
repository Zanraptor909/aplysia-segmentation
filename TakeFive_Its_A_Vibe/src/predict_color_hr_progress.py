# predict_color_hr_progress.py
# Safer mask conversion, high-recall, smooth edges, safe merge.
# Cleaned for fewer IDE warnings (PyCharm/Pylance).

from ultralytics import YOLO
import cv2
import numpy as np
import math
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple

# ===== Paths =====
RUN_DIR    = Path(r"C:\TakeFive_Its_A_Vibe\runs\aplysia_seg2") # Change Model Used
MODEL_BEST = RUN_DIR / "weights" / "best.pt"
MODEL_LAST = RUN_DIR / "weights" / "last.pt"
IMG        = Path(r"C:\TakeFive_Its_A_Vibe\test_image\test_image.JPG") # Change Test Image Used
SAVE_DIR   = Path(r"C:\TakeFive_Its_A_Vibe\test_image\predict_out")

# ===== Inference (high recall, CPU, retina masks) =====
IMG_SIZE       = 704
CONF_MIN       = 0.08
IOU_NMS        = 0.40
MAX_DET        = 1600
RETINA_MASKS   = True
DEVICE         = "cpu"
HALF           = False

# ===== Soft color prior & splitting =====
H_LOW1, H_HIGH1 = 0, 10
H_LOW2, H_HIGH2 = 170, 180
S_MIN, V_MIN    = 28, 22
RED_KEEP_QUANT  = 0.42
MIN_KEEP_RATIO  = 0.35

SPLIT_IF_AREA_RATIO = 1.45
MIN_PEAK_REL        = 0.36
SEP_FACTOR          = 0.58
BG_DILATE_ITERS     = 3
MIN_SEG_AREA_PCT    = 0.00018

# ===== Smoothness & safe merge =====
SMOOTH_REFINE_ITER  = 2
BLUR_SIGMA          = 1.4
SIMPLIFY_EPS_FRAC   = 0.003

MERGE_BOX_PAD_PX        = 2
MERGE_COVERAGE_MAXFRAC  = 0.20

# ---------- helpers ----------
def pick_model() -> str:
    if MODEL_BEST.exists(): return str(MODEL_BEST)
    if MODEL_LAST.exists():
        print("[info] best.pt not found, using last.pt")
        return str(MODEL_LAST)
    raise FileNotFoundError(f"No weights found in {RUN_DIR/'weights'}")

def redness_map_from_bgr(bgr_small: np.ndarray) -> np.ndarray:
    """Soft redness score in [0,1]; keep float32 contiguous."""
    hsv = cv2.cvtColor(bgr_small, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    # make boolean array explicitly numpy.bool_ before cast
    hue_gate = np.asarray((h >= H_LOW2) | (h <= H_HIGH1), dtype=np.bool_)
    sat_ok   = np.asarray(s >= S_MIN, dtype=np.bool_)
    val_ok   = np.asarray(v >= V_MIN, dtype=np.bool_)
    gate = np.asarray(hue_gate & sat_ok & val_ok, dtype=np.uint8).astype(np.float32)

    lab = cv2.cvtColor(bgr_small, cv2.COLOR_BGR2Lab)
    a = lab[:, :, 1].astype(np.float32)
    a_norm = cv2.normalize(a, None, 0.0, 1.0, cv2.NORM_MINMAX)
    soft = 0.5 + 0.5 * gate
    score = (a_norm * soft).astype(np.float32)
    score = cv2.GaussianBlur(score, (0, 0), 1.0)  # type: ignore[arg-type]
    return np.ascontiguousarray(score, dtype=np.float32)

def estimate_typical_radius_px(mask_list: List[np.ndarray]) -> float:
    if not mask_list: return 8.0
    areas = [cv2.countNonZero(m) for m in mask_list]
    if not areas: return 8.0
    med = float(np.median(areas))
    smalls = [a for a in areas if a <= med] or areas
    typ_area = float(np.median(smalls))
    return max(math.sqrt(max(typ_area, 1.0) / math.pi), 4.0)

def local_maxima_peaks(dist: np.ndarray, rel_thresh: float, min_sep_px: int) -> List[Tuple[int, int]]:
    if float(dist.max()) <= 0: return []
    d = cv2.GaussianBlur(dist, (0, 0), 1.0)  # type: ignore[arg-type]
    _, thr = cv2.threshold(d, rel_thresh * float(d.max()), 255, cv2.THRESH_BINARY)
    thr = thr.astype(np.uint8)
    dmax = cv2.dilate(d, np.ones((3, 3), np.uint8))
    peaks_mask = np.asarray((d == dmax) & (thr > 0), dtype=np.uint8) * 255
    nlabels, _, _, centers = cv2.connectedComponentsWithStats(peaks_mask)
    pts: List[Tuple[int, int]] = []
    for k in range(1, nlabels):
        cx, cy = centers[k]
        pts.append((int(round(float(cx))), int(round(float(cy)))))
    kept: List[Tuple[int, int]] = []
    for p in sorted(pts, key=lambda xy: d[xy[1], xy[0]], reverse=True):
        if all((p[0]-q[0])**2 + (p[1]-q[1])**2 >= (min_sep_px**2) for q in kept):
            kept.append(p)
    return kept

def watershed_split(mask8: np.ndarray, min_sep_px: int, rel_thresh: float) -> List[np.ndarray]:
    dist = cv2.distanceTransform(mask8, cv2.DIST_L2, 5).astype(np.float32)
    peaks = local_maxima_peaks(dist, rel_thresh, min_sep_px)
    if len(peaks) < 2:
        c = find_main_contour(mask8)
        return [c] if c is not None else []
    markers = np.zeros(mask8.shape, dtype=np.int32)
    for idx, (x, y) in enumerate(peaks, start=1):
        cv2.circle(markers, (int(x), int(y)), 1, int(idx), -1)
    sure_bg = cv2.dilate(mask8, np.ones((3, 3), np.uint8), iterations=BG_DILATE_ITERS)
    unknown = cv2.subtract(sure_bg, (mask8 > 0).astype(np.uint8) * 255)
    markers[unknown > 0] = 0
    topo = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    topo_rgb = cv2.cvtColor(255 - topo, cv2.COLOR_GRAY2BGR)
    cv2.watershed(topo_rgb, markers)  # type: ignore[arg-type]
    segs: List[np.ndarray] = []
    for lbl in range(1, int(markers.max()) + 1):
        seg = (markers == lbl).astype(np.uint8) * 255
        c = find_main_contour(seg)
        if c is not None:
            segs.append(c.astype(np.int32))
    return segs or ([find_main_contour(mask8)] if find_main_contour(mask8) is not None else [])

def find_main_contour(mask8: np.ndarray) -> np.ndarray | None:
    cnts, _ = cv2.findContours(mask8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    return max(cnts, key=lambda c: cv2.contourArea(c.astype(np.float32)))

def smooth_mask(mask8: np.ndarray) -> np.ndarray:
    m = np.ascontiguousarray(mask8, dtype=np.uint8)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, ker, iterations=SMOOTH_REFINE_ITER)
    # fill holes
    h, w = m.shape
    ff = m.copy()
    cv2.floodFill(ff, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 255)
    m = cv2.bitwise_or(m, cv2.bitwise_not(ff))
    m = cv2.GaussianBlur(m, (0, 0), BLUR_SIGMA)  # type: ignore[arg-type]
    _, m = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
    return np.ascontiguousarray(m, dtype=np.uint8)

def simplify_contour(c: np.ndarray, eps_frac: float = SIMPLIFY_EPS_FRAC) -> np.ndarray:
    peri = cv2.arcLength(c, True)
    eps = max(1.0, eps_frac * float(peri))
    return cv2.approxPolyDP(c, eps, True)

def scale_contours(contours: List[np.ndarray], sx: float, sy: float) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for c in contours:
        xy = c.reshape(-1, 2).astype(np.float32)
        xy[:, 0] *= sx
        xy[:, 1] *= sy
        out.append(np.ascontiguousarray(xy.reshape(-1, 1, 2).astype(np.int32)))
    return out

def draw_polys(img_bgr: np.ndarray, polys: List[np.ndarray], color: Tuple[int, int, int] = (0, 255, 255),
               fill: bool = False, alpha: float = 0.35, thickness: int = 2) -> np.ndarray:
    out = img_bgr.copy()
    if fill:
        overlay = img_bgr.copy()
        for pts in polys:
            cv2.fillPoly(overlay, [pts], color)
        out = cv2.addWeighted(overlay, alpha, img_bgr, 1 - alpha, 0)
    else:
        for pts in polys:
            cv2.polylines(out, [pts], True, color, thickness)
    return out

# -------- SAFE MERGE (bbox overlap only) ----------
def _expanded_rect(rect: Tuple[int, int, int, int], pad: int, width: int, height: int) -> Tuple[int, int, int, int]:
    x, y, w, h = rect
    x2, y2 = x + w, y + h
    x = max(0, x - pad); y = max(0, y - pad)
    x2 = min(width,  x2 + pad); y2 = min(height, y2 + pad)
    return (x, y, x2, y2)

def _rects_overlap(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1)

def merge_nearby_contours_safe(contours: List[np.ndarray], canvas_hw: Tuple[int, int], pad_px: int,
                               coverage_guard: float = 0.20) -> List[np.ndarray]:
    if not contours: return []
    height, width = canvas_hw
    rects = [cv2.boundingRect(c) for c in contours]
    ex = [_expanded_rect(r, pad_px, width, height) for r in rects]

    # union-find
    parent = list(range(len(ex)))
    def find_root(a: int) -> int:
        k = a
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k
    def unite(a: int, b: int) -> None:
        ra, rb = find_root(a), find_root(b)
        if ra != rb:
            parent[ra] = rb

    for ia in range(len(ex)):
        for ib in range(ia + 1, len(ex)):
            if _rects_overlap(ex[ia], ex[ib]):
                unite(ia, ib)

    groups: dict[int, List[int]] = {}
    for idx, _ in enumerate(ex):
        r = find_root(idx)
        groups.setdefault(r, []).append(idx)

    merged: List[np.ndarray] = []
    total_area = 0.0
    for g_idxs in groups.values():
        layer = np.zeros((height, width), np.uint8)
        for gi in g_idxs:
            cv2.drawContours(layer, [contours[gi]], -1, 255, -1)  # type: ignore[arg-type]
        cnts, _ = cv2.findContours(layer, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            cmax = max(cnts, key=lambda cc: cv2.contourArea(cc.astype(np.float32)))
            merged.append(cmax.astype(np.int32))
            total_area += float(cv2.contourArea(cmax.astype(np.float32)))

    if (total_area / float(width * height)) > coverage_guard:
        return contours
    return merged

def main() -> None:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    phases = ["Load model", "Run pass1", "Prepare masks", "Color map", "Refine/split/merge", "Render/Save"]
    pbar = tqdm(total=len(phases), desc="Starting", unit="phase")

    # 1) Load
    pbar.set_description(phases[0])
    model = YOLO(pick_model()); pbar.update(1)

    # 2) Pass 1
    pbar.set_description(phases[1])
    rs = model.predict(
        source=str(IMG), imgsz=IMG_SIZE, conf=CONF_MIN, iou=IOU_NMS, max_det=MAX_DET,
        augment=False, retina_masks=RETINA_MASKS, device=DEVICE, half=HALF,
        save=False, verbose=False, show_labels=False, show_conf=False
    )
    r = rs[0]; pbar.update(1)

    # 3) Prepare masks (safe conversion)
    pbar.set_description(phases[2])
    orig = cv2.imread(str(IMG));  assert orig is not None, f"Could not read {IMG}"
    height, width = orig.shape[:2]
    if r.masks is None or getattr(r.masks, "data", None) is None:
        pbar.close(); print("No masks returned."); return

    m = r.masks.data
    if hasattr(m, "detach"): m = m.detach()
    if hasattr(m, "cpu"):    m = m.cpu()
    m = np.array(m, dtype=np.float32, copy=True)          # own memory
    masks: List[np.ndarray] = [np.ascontiguousarray((mi > 0.5).astype(np.uint8) * 255) for mi in m]
    if not masks:
        pbar.close(); print("No masks returned."); return

    mh, mw = masks[0].shape[:2]
    sx, sy = width / float(mw), height / float(mh)
    areas = [cv2.countNonZero(x) for x in masks]
    med_area = float(np.median(areas)) if areas else 0.0
    typ_radius = estimate_typical_radius_px(masks)
    min_sep_px = max(3, int(SEP_FACTOR * typ_radius))
    pbar.update(1)

    # 4) Color map
    pbar.set_description("Color map")
    small = cv2.resize(orig, (mw, mh), interpolation=cv2.INTER_AREA)
    red_score = redness_map_from_bgr(small)
    pbar.update(1)

    # 5) Refine + split + merge
    pbar.set_description(phases[4])
    segs_model: List[np.ndarray] = []
    for idx in tqdm(range(len(masks)), desc="Masks", unit="mask", leave=False):
        mask8 = masks[idx]
        area0 = areas[idx]
        if area0 == 0:
            continue

        m_bool = (mask8 > 0)
        scores_in = red_score[m_bool]
        if scores_in.size > 20 and float(scores_in.max()) > 0:
            thr = float(np.quantile(scores_in, RED_KEEP_QUANT))
            refined = np.where((m_bool) & (red_score >= thr), 255, 0).astype(np.uint8)
            if cv2.countNonZero(refined) < MIN_KEEP_RATIO * area0:
                refined = mask8
        else:
            refined = mask8

        refined = smooth_mask(refined)

        area_ref = cv2.countNonZero(refined)
        if med_area > 0 and area_ref > SPLIT_IF_AREA_RATIO * med_area:
            pieces = watershed_split(refined, min_sep_px=min_sep_px, rel_thresh=MIN_PEAK_REL)
        else:
            c = find_main_contour(refined)
            pieces = [c] if c is not None else []

        for pc in pieces:
            a = float(cv2.contourArea(pc.astype(np.float32)))
            if (a / (mw * mh)) >= (MIN_SEG_AREA_PCT / (sx * sy)):
                segs_model.append(pc)

    merged_model = merge_nearby_contours_safe(segs_model, (mh, mw), MERGE_BOX_PAD_PX,
                                              coverage_guard=MERGE_COVERAGE_MAXFRAC)

    # Scale & smooth outlines
    contours_all = scale_contours(merged_model, sx, sy)
    contours_all = [simplify_contour(c) for c in contours_all]

    # 6) Render/Save
    pbar.set_description(phases[5])
    filled   = draw_polys(orig, contours_all, fill=True,  alpha=0.35)
    outlined = draw_polys(orig, contours_all, fill=False, thickness=2)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    out_filled  = SAVE_DIR / f"{IMG.stem}_filled_HR_pb.png"
    out_outline = SAVE_DIR / f"{IMG.stem}_outline_HR_pb.png"
    cv2.imwrite(str(out_filled), filled)
    cv2.imwrite(str(out_outline), outlined)
    pbar.update(1); pbar.close()

    print(f"Saved (filled masks)  -> {out_filled}")
    print(f"Saved (outline only)  -> {out_outline}")
    print(f"Final count: {len(contours_all)}")

if __name__ == "__main__":
    main()
