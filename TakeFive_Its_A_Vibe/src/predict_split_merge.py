# predict_split_merge.py — with progress bar
from ultralytics import YOLO
import cv2, numpy as np, math, os
from typing import List, Tuple
from tqdm import tqdm  # pip install tqdm

# ---------- paths ----------
MODEL    = r"C:\TakeFive_Its_A_Vibe\runs\aplysia_seg2\weights\best.pt"
IMG      = r"C:\TakeFive_Its_A_Vibe\test_image\test_image.JPG"
SAVE_DIR = r"C:\TakeFive_Its_A_Vibe\test_image\predict_out"

# ---------- render ----------
COLOR = (0, 255, 255)   # yellow
ALPHA = 0.35
THICK = 2

# ---------- inference (12GB VRAM) ----------
IMG_SIZE     = 960      # ↑ bigger than before for finer masks
CONF_MIN     = 0.07
IOU_NMS      = 0.40
MAX_DET      = 2500
RETINA_MASKS = True
DEVICE       = 0        # GPU
HALF         = True     # FP16 on GPU

# ---------- color prior & splitting ----------
S_MIN, V_MIN      = 28, 22
RED_KEEP_QUANT    = 0.45
MIN_KEEP_RATIO    = 0.30

SPLIT_IF_AREA_RATIO = 1.45
MIN_PEAK_REL        = 0.38
SEP_FACTOR          = 0.60
BG_DILATE_ITERS     = 3
MIN_SEG_AREA_PCT    = 0.00018

# ---------- helpers ----------
def redness_map_from_bgr(bgr_small: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr_small, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    gate = (((h <= 10) | (h >= 170)) & (s >= S_MIN) & (v >= V_MIN)).astype(np.float32)
    lab = cv2.cvtColor(bgr_small, cv2.COLOR_BGR2Lab)
    a = lab[:, :, 1].astype(np.float32)
    a_norm = cv2.normalize(a, None, 0.0, 1.0, cv2.NORM_MINMAX)
    score = (0.5 + 0.5 * gate) * a_norm
    return cv2.GaussianBlur(score.astype(np.float32), (0, 0), 1.0)

def estimate_typical_radius_px(mask_list: List[np.ndarray]) -> float:
    if not mask_list: return 8.0
    areas = [cv2.countNonZero(m) for m in mask_list]
    if not areas: return 8.0
    med = float(np.median(areas))
    smalls = [a for a in areas if a <= med] or areas
    typ_area = float(np.median(smalls))
    return max(math.sqrt(max(typ_area, 1.0) / math.pi), 4.0)

def local_maxima_peaks(dist: np.ndarray, rel_thresh: float, min_sep_px: int) -> List[Tuple[int,int]]:
    if float(dist.max()) <= 0: return []
    d = cv2.GaussianBlur(dist, (0,0), 1.0)
    _, thr = cv2.threshold(d, rel_thresh * float(d.max()), 255, cv2.THRESH_BINARY)
    thr = thr.astype(np.uint8)
    dmax = cv2.dilate(d, np.ones((3,3), np.uint8))
    peaks_mask = ((d == dmax) & (thr > 0)).astype(np.uint8) * 255
    n, _, _, cents = cv2.connectedComponentsWithStats(peaks_mask)
    pts = [(int(round(cx)), int(round(cy))) for i,(cx,cy) in enumerate(cents) if i != 0]
    kept: List[Tuple[int,int]] = []
    for p in sorted(pts, key=lambda xy: d[xy[1], xy[0]], reverse=True):
        if all((p[0]-q[0])**2 + (p[1]-q[1])**2 >= min_sep_px**2 for q in kept):
            kept.append(p)
    return kept

def watershed_split(mask8: np.ndarray, min_sep_px: int, rel_thresh: float) -> List[np.ndarray]:
    dist = cv2.distanceTransform(mask8, cv2.DIST_L2, 5).astype(np.float32)
    peaks = local_maxima_peaks(dist, rel_thresh, min_sep_px)
    if len(peaks) < 2:
        c = find_main_contour(mask8)
        return [c] if c is not None else []
    markers = np.zeros(mask8.shape, dtype=np.int32)
    for i, (x, y) in enumerate(peaks, start=1):
        cv2.circle(markers, (x, y), 1, i, -1)
    sure_bg = cv2.dilate(mask8, np.ones((3,3), np.uint8), iterations=BG_DILATE_ITERS)
    unknown = cv2.subtract(sure_bg, mask8)
    markers[unknown > 0] = 0
    topo = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.watershed(cv2.cvtColor(255 - topo, cv2.COLOR_GRAY2BGR), markers)
    out: List[np.ndarray] = []
    for lbl in range(1, markers.max() + 1):
        seg = (markers == lbl).astype(np.uint8) * 255
        c = find_main_contour(seg)
        if c is not None: out.append(c.astype(np.int32))
    return out

def find_main_contour(mask8: np.ndarray) -> np.ndarray | None:
    cnts, _ = cv2.findContours(mask8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    return max(cnts, key=lambda c: cv2.contourArea(c.astype(np.float32)))

def simplify(c: np.ndarray, eps_frac: float = 0.003) -> np.ndarray:
    peri = cv2.arcLength(c, True)
    eps  = max(1.0, eps_frac * float(peri))
    return cv2.approxPolyDP(c, eps, True)

def dedup_by_iou(contours: List[np.ndarray], hw: Tuple[int,int], iou_thr: float = 0.6) -> List[np.ndarray]:
    """Remove near-duplicates by mask IoU, keeping larger area."""
    if not contours: return []
    h, w = hw
    keep: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    for c in sorted(contours, key=lambda cc: cv2.contourArea(cc.astype(np.float32)), reverse=True):
        m = np.zeros((h, w), np.uint8); cv2.drawContours(m, [c], -1, 255, -1)
        dup = False
        for mm in masks:
            inter = np.logical_and(mm > 0, m > 0).sum()
            union = np.logical_or(mm > 0, m > 0).sum()
            if union > 0 and (inter / union) > iou_thr:
                dup = True; break
        if not dup:
            keep.append(c); masks.append(m)
    return keep

def masks_to_contours_split(r) -> List[np.ndarray]:
    """Refine each mask + split clumps; return contours at ORIGINAL image scale.
    Progress bar shows per-mask processing."""
    if r.masks is None or getattr(r.masks, "data", None) is None: return []
    m = r.masks.data
    if hasattr(m, "detach"): m = m.detach()
    if hasattr(m, "cpu"):    m = m.cpu()
    m = np.array(m, dtype=np.float32, copy=True)    # [N,h,w]

    H, W = r.orig_shape
    mh, mw = m.shape[1:3]
    sx, sy = W / float(mw), H / float(mh)

    small = cv2.resize(r.orig_img, (mw, mh), interpolation=cv2.INTER_AREA)
    red_score = redness_map_from_bgr(small)

    masks_bin = [(mi > 0.5).astype(np.uint8) * 255 for mi in m]
    areas = [cv2.countNonZero(x) for x in masks_bin]
    med_area = float(np.median(areas)) if areas else 0.0
    typ_radius = estimate_typical_radius_px(masks_bin)
    min_sep_px = max(3, int(SEP_FACTOR * typ_radius))

    model_contours: List[np.ndarray] = []
    for mask8, a0 in tqdm(list(zip(masks_bin, areas)), desc="Processing masks", unit="mask"):
        if a0 == 0:
            continue

        # soft color refine + fallback if too much trimmed
        where = mask8 > 0
        sc = red_score[where]
        if sc.size > 20 and float(sc.max()) > 0:
            thr = float(np.quantile(sc, RED_KEEP_QUANT))
            refined = np.zeros_like(mask8); refined[where & (red_score >= thr)] = 255
            if cv2.countNonZero(refined) < MIN_KEEP_RATIO * a0:
                refined = mask8
        else:
            refined = mask8

        # smooth & fill holes
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, ker, iterations=2)
        ff = refined.copy()
        cv2.floodFill(ff, np.zeros((ff.shape[0]+2, ff.shape[1]+2), np.uint8), (0,0), 255)
        refined = refined | cv2.bitwise_not(ff)
        refined = cv2.GaussianBlur(refined, (0,0), 1.4)
        _, refined = cv2.threshold(refined, 127, 255, cv2.THRESH_BINARY)

        # split large unions
        if med_area > 0 and cv2.countNonZero(refined) > SPLIT_IF_AREA_RATIO * med_area:
            segs = watershed_split(refined, min_sep_px=min_sep_px, rel_thresh=MIN_PEAK_REL)
        else:
            c = find_main_contour(refined); segs = [c] if c is not None else []

        for c in segs:
            a_model = float(cv2.contourArea(c.astype(np.float32)))
            a_img   = a_model * sx * sy
            if a_img / (W * H) >= MIN_SEG_AREA_PCT:
                model_contours.append(c)

    # deduplicate (model scale), then scale up
    model_contours = dedup_by_iou(model_contours, (mh, mw), iou_thr=0.6)
    out: List[np.ndarray] = []
    for c in model_contours:
        xy = c.reshape(-1, 2).astype(np.float32)
        xy[:, 0] *= sx; xy[:, 1] *= sy
        out.append(simplify(xy.reshape(-1,1,2).astype(np.int32)))
    return out

def draw_and_save(orig: np.ndarray, polys: List[np.ndarray]) -> Tuple[str,str]:
    base = os.path.splitext(os.path.basename(IMG))[0]
    out_filled  = os.path.join(SAVE_DIR, f"{base}_filled.png")
    out_outline = os.path.join(SAVE_DIR, f"{base}_outline.png")

    filled = orig.copy()
    if polys:
        overlay = orig.copy()
        cv2.fillPoly(overlay, polys, COLOR)
        filled = cv2.addWeighted(overlay, ALPHA, orig, 1-ALPHA, 0)

    outlined = orig.copy()
    if polys:
        cv2.polylines(outlined, polys, True, COLOR, THICK)

    cv2.imwrite(out_filled, filled)
    cv2.imwrite(out_outline, outlined)
    return out_filled, out_outline

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    tqdm.write("Loading model and running inference…")
    model = YOLO(MODEL)
    results = model.predict(
        source=IMG, imgsz=IMG_SIZE, conf=CONF_MIN, iou=IOU_NMS, max_det=MAX_DET,
        retina_masks=RETINA_MASKS, device=DEVICE, half=HALF,
        save=False, verbose=False, show_labels=False, show_conf=False
    )
    r = results[0]

    # Raw count as returned by YOLO
    raw_count = (len(r.boxes) if r.boxes is not None
                 else (0 if r.masks is None else r.masks.data.shape[0]))

    tqdm.write("Post-processing masks…")
    polys = masks_to_contours_split(r)

    out_filled, out_outline = draw_and_save(r.orig_img.copy(), polys)
    tqdm.write(f"Saved filled  -> {out_filled}")
    tqdm.write(f"Saved outline -> {out_outline}")
    tqdm.write(f"Raw YOLO instances: {raw_count}")
    tqdm.write(f"Post-processed count: {len(polys)}")

if __name__ == "__main__":
    main()
