from ultralytics import YOLO
import cv2
import numpy as np
import os
import time
from collections import deque

# =========================
# -------- CONFIG ---------
# =========================
# Source toggle
USE_WEBCAM = True            # True = webcam feed, False = single image
WEBCAM_ID  = 0               # which webcam to open when USE_WEBCAM=True

# Paths
MODEL     = r"C:\TakeFive_Its_A_Vibe\runs\aplysia_seg2\weights\best.pt"   # trained weights
IMG       = r"C:\TakeFive_Its_A_Vibe\test_image\test_image_5.JPG"         # image path
SAVE_DIR  = r"C:\TakeFive_Its_A_Vibe\test_image\predict_out"              # output folder

# Inference knobs (tune these to reduce false positives)
IMG_SIZE      = 960          # 640/960/1280 typical; bigger → finer masks
CONF_MIN      = 0.30         # ↑ increase to be stricter (0.35–0.50 if too many slugs)
IOU_NMS       = 0.45         # higher merges more, lower keeps more overlaps
MAX_DET       = 2000         # per-image cap
RETINA_MASKS  = True         # higher-res masks (slower/more VRAM)
DEVICE        = 0            # 0 for first GPU, "cpu" to force CPU
HALF          = True         # FP16 on GPU (ignored on CPU)

# Rendering
SAVE_FILLED   = True         # save filled overlay on image mode
SAVE_OUTLINE  = True         # save outline-only on image mode
COLOR         = (0, 255, 255)  # BGR yellow
ALPHA         = 0.35           # fill opacity
THICK         = 2              # outline thickness
SHOW_HUD      = True           # draw text HUD (count/settings) on preview

# Webcam averaging
SHOW_AVG      = True           # show an average on webcam
USE_EMA       = True           # True = EMA, False = rolling average
EMA_ALPHA     = 0.20           # 0..1 (higher = more responsive, less smooth)
AVG_WINDOW    = 30             # rolling average window if USE_EMA=False

# =========================
# ------- HELPERS ---------
# =========================
def masks_to_polys(r):
    """Return list of OpenCV polygons (Nx1x2 int32) for each mask."""
    polys = []
    if r.masks is None:
        return polys

    # Prefer Ultralytics' polygon extraction if available
    if getattr(r.masks, "xy", None) is not None and len(r.masks.xy):
        for arr in r.masks.xy:
            pts = np.asarray(arr, dtype=np.int32).reshape(-1, 1, 2)
            polys.append(pts)
        return polys

    # Fallback: derive polygons from binary masks
    m = r.masks.data
    if hasattr(m, "cpu"):
        m = m.cpu().numpy()
    for m_i in m:
        mask = (m_i > 0.5).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            polys.append(c.astype(np.int32))
    return polys

def predict_on(model, frame_or_path):
    """Run YOLO with configured settings on a frame (np.ndarray) or path (str)."""
    results = model(
        frame_or_path,
        imgsz=IMG_SIZE,
        conf=CONF_MIN,
        iou=IOU_NMS,
        max_det=MAX_DET,
        device=DEVICE,
        half=(HALF if DEVICE != "cpu" else False),
        retina_masks=RETINA_MASKS,
        verbose=False,
    )
    return results[0]

def render_frame(r, polys=None, draw_hud=True, window_title="Prediction",
                 avg_value=None, avg_label="Avg"):
    """
    Render polygons (filled + outline) onto original image, add HUD, and show one window.
    If 'polys' is provided, it is used directly (no recompute). Returns (frame, count, (conf_min, conf_max)).
    """
    orig = r.orig_img.copy()
    polys = polys if polys is not None else masks_to_polys(r)

    # Draw filled
    if polys:
        overlay = orig.copy()
        cv2.fillPoly(overlay, polys, COLOR)
        orig = cv2.addWeighted(overlay, ALPHA, orig, 1 - ALPHA, 0)

    # Draw outlines
    if polys:
        cv2.polylines(orig, polys, isClosed=True, color=COLOR, thickness=THICK)

    # Count + conf range
    num_instances = len(polys)
    conf_min = conf_max = None
    if r.boxes is not None and r.boxes.conf is not None and len(r.boxes) > 0:
        confs = r.boxes.conf.tolist()
        conf_min, conf_max = (min(confs), max(confs)) if confs else (None, None)

    if draw_hud:
        y = 28
        # Current count
        cv2.putText(orig, f"Count: {num_instances}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        y += 26
        # Average (if provided)
        if avg_value is not None and SHOW_AVG:
            cv2.putText(orig, f"{avg_label}: {avg_value:.2f}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 3)
            cv2.putText(orig, f"{avg_label}: {avg_value:.2f}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 1)
            y += 24
        # Confidence range
        if conf_min is not None:
            cv2.putText(orig, f"Conf: {conf_min:.2f}-{conf_max:.2f}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2)
            y += 24
        # Settings line
        cv2.putText(orig, f"Settings: conf={CONF_MIN:.2f} iou={IOU_NMS:.2f} img={IMG_SIZE}",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2)
        y += 22
        dev_txt = f"dev={DEVICE} fp16={HALF and DEVICE!='cpu'} retina={RETINA_MASKS}"
        cv2.putText(orig, dev_txt, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2)

    # Show exactly ONE window (whatever title the caller chose)
    cv2.imshow(window_title, orig)
    return orig, num_instances, (conf_min, conf_max)

# =========================
# --------- MAIN ----------
# =========================
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    model = YOLO(MODEL)

    if USE_WEBCAM:
        cap = cv2.VideoCapture(WEBCAM_ID)
        if not cap.isOpened():
            print(f"❌ Could not open webcam {WEBCAM_ID}")
            return
        print("🎥 Webcam mode: press 'q' to quit, 's' to save current frame")

        # Averages
        counts = deque(maxlen=max(1, int(AVG_WINDOW))) if not USE_EMA else None
        ema_val = None

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Inference
            r = predict_on(model, frame)

            # Compute polys and count WITHOUT showing any extra window
            polys = masks_to_polys(r)
            count = len(polys)

            # Averages (EMA or rolling)
            if SHOW_AVG:
                if USE_EMA:
                    if ema_val is None:
                        ema_val = float(count)
                    else:
                        ema_val = EMA_ALPHA * float(count) + (1.0 - EMA_ALPHA) * ema_val
                    avg_val, avg_label = ema_val, "EMA"
                else:
                    counts.append(count)
                    avg_val = (sum(counts) / len(counts)) if counts else 0.0
                    avg_label = f"Avg({len(counts)}/{counts.maxlen})"
            else:
                avg_val, avg_label = None, "Avg"

            # Render + show exactly once
            rendered, count, (cmin, cmax) = render_frame(
                r, polys=polys, draw_hud=SHOW_HUD, window_title="Webcam",
                avg_value=avg_val, avg_label=avg_label
            )

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                ts = time.strftime("%Y%m%d-%H%M%S")
                suffix = f"_count{count}"
                if avg_val is not None:
                    suffix += f"_{avg_label.lower()}{avg_val:.2f}"
                out_path = os.path.join(SAVE_DIR, f"webcam_{ts}{suffix}.png")
                cv2.imwrite(out_path, rendered)
                print(f"Saved snapshot -> {out_path}")

        cap.release()
        cv2.destroyAllWindows()

    else:
        r = predict_on(model, IMG)
        # Single image: render once, one window
        rendered, count, (cmin, cmax) = render_frame(
            r, draw_hud=SHOW_HUD, window_title="Image"
        )

        base = os.path.splitext(os.path.basename(IMG))[0]

        if SAVE_FILLED:
            out_filled = os.path.join(SAVE_DIR, f"{base}_filled.png")
            cv2.imwrite(out_filled, rendered)
            print(f"Saved filled  -> {out_filled}")

        if SAVE_OUTLINE:
            # Outline-only version
            orig = r.orig_img.copy()
            polys = masks_to_polys(r)
            if polys:
                cv2.polylines(orig, polys, isClosed=True, color=COLOR, thickness=THICK)
            out_outline = os.path.join(SAVE_DIR, f"{base}_outline.png")
            cv2.imwrite(out_outline, orig)
            print(f"Saved outline -> {out_outline}")

        print(f"Detected {count} objects")
        if cmin is not None:
            print(f"Confidence range: {cmin:.3f} – {cmax:.3f}")

        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
