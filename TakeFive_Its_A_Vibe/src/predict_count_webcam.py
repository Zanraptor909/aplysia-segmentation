# predict_count_webcam.py
from ultralytics import YOLO
import cv2
import numpy as np
import os, time, sys, platform
from collections import deque

# =========================
# -------- CONFIG ---------
# =========================
# Source toggle
USE_WEBCAM = True            # True = webcam feed, False = single image
WEBCAM_ID  = 0               # preferred webcam index to try first (0=laptop, 1=USB)

# Paths
MODEL     = r"C:\aplysia-segmentation\TakeFive_Its_A_Vibe\runs\seg_v8s\weights\best.pt"  # trained weights
IMG       = r"C:\aplysia-segmentation\TakeFive_Its_A_Vibe\test_image\test_image.JPG"     # image path
SAVE_DIR  = r"C:\aplysia-segmentation\TakeFive_Its_A_Vibe\test_image\predict_out"        # output folder

# Inference knobs
IMG_SIZE      = 960          # 640/960/1280 typical; bigger → finer masks
CONF_MIN      = 0.30         # ↑ increase to be stricter
IOU_NMS       = 0.45
MAX_DET       = 2000
RETINA_MASKS  = True
DEVICE        = 0            # 0 for first GPU, or "cpu"
HALF          = True         # FP16 on GPU (ignored when CPU)

# Rendering
SAVE_FILLED   = True         # for image mode
SAVE_OUTLINE  = True         # for image mode
COLOR         = (0, 255, 255)  # BGR yellow
ALPHA         = 0.35
THICK         = 2
SHOW_HUD      = True

# Webcam averaging
SHOW_AVG      = True
USE_EMA       = True         # True = EMA, False = rolling avg
EMA_ALPHA     = 0.20
AVG_WINDOW    = 30           # rolling window if USE_EMA=False

# Camera handling
CANDIDATE_IDS = [WEBCAM_ID, 1, 0, 2, 3, 4]  # order to probe/cycle through


# =========================
# ------- HELPERS ---------
# =========================
def masks_to_polys(r):
    """Return list of OpenCV polygons (Nx1x2 int32) for each mask at original image scale."""
    polys = []
    if r.masks is None:
        return polys

    # Prefer Ultralytics polygons if available (already scaled to original image)
    if getattr(r.masks, "xy", None) is not None and len(r.masks.xy):
        for arr in r.masks.xy:
            pts = np.asarray(arr, dtype=np.int32).reshape(-1, 1, 2)
            polys.append(pts)
        return polys

    # Fallback: derive polys from binary masks (scale handled by r.plot normally; we handle directly)
    m = r.masks.data
    if hasattr(m, "cpu"):
        m = m.cpu().numpy()
    for m_i in m:
        mask = (m_i > 0.5).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            polys.append(c.astype(np.int32))
    return polys


def predict_on(model, frame_or_path, device_override=None):
    """Run YOLO with configured settings on a frame (np.ndarray) or path (str)."""
    dev = DEVICE if device_override is None else device_override
    return model.predict(
        source=frame_or_path,
        imgsz=IMG_SIZE,
        conf=CONF_MIN,
        iou=IOU_NMS,
        max_det=MAX_DET,
        device=dev,
        half=(HALF and dev != "cpu"),
        retina_masks=RETINA_MASKS,
        verbose=False,
        save=False,
        show=False,
        show_labels=False,
        show_conf=False,
    )[0]


def render_frame(r, polys=None, draw_hud=True, window_title="Prediction",
                 avg_value=None, avg_label="Avg"):
    """
    Render polygons (filled + outline) onto original image, add HUD, and show one window.
    Returns (frame, count, (conf_min, conf_max)).
    """
    orig = r.orig_img.copy()
    polys = polys if polys is not None else masks_to_polys(r)

    # Filled overlay
    if polys:
        overlay = orig.copy()
        cv2.fillPoly(overlay, polys, COLOR)
        orig = cv2.addWeighted(overlay, ALPHA, orig, 1 - ALPHA, 0)

    # Outlines
    if polys:
        cv2.polylines(orig, polys, isClosed=True, color=COLOR, thickness=THICK)

    # Count + conf range
    num_instances = len(polys)
    conf_min = conf_max = None
    if r.boxes is not None and r.boxes.conf is not None and len(r.boxes) > 0:
        confs = r.boxes.conf.tolist()
        if confs:
            conf_min, conf_max = min(confs), max(confs)

    if draw_hud:
        y = 28
        # Current count
        cv2.putText(orig, f"Count: {num_instances}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        y += 26
        # Average (if provided)
        if avg_value is not None and SHOW_AVG:
            text = f"{avg_label}: {avg_value:.2f}"
            cv2.putText(orig, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 3)
            cv2.putText(orig, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 1)
            y += 24
        # Confidence range
        if conf_min is not None:
            cv2.putText(orig, f"Conf: {conf_min:.2f}-{conf_max:.2f}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2)
            y += 24
        # Settings
        cv2.putText(orig, f"Settings: conf={CONF_MIN:.2f} iou={IOU_NMS:.2f} img={IMG_SIZE}",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2)
        y += 22
        dev_txt = f"dev={DEVICE} fp16={HALF and DEVICE!='cpu'} retina={RETINA_MASKS}"
        cv2.putText(orig, dev_txt, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2)

    # Show exactly ONE window
    cv2.imshow(window_title, orig)
    return orig, num_instances, (conf_min, conf_max)


def open_camera(cam_id):
    """Open a camera index with a reliable backend on Windows."""
    backend = cv2.CAP_DSHOW if platform.system() == "Windows" else 0
    cap = cv2.VideoCapture(cam_id, backend)
    # Hint preferred format (drivers may ignore)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS,          30)
    return cap


def find_working_camera():
    """Try candidate indices; return (cap, id) if one works and delivers frames."""
    for cid in CANDIDATE_IDS:
        cap = open_camera(cid)
        if cap.isOpened():
            ok1, _ = cap.read()
            ok2, _ = cap.read()
            if ok1 or ok2:
                print(f"🎥 Using camera index {cid}")
                return cap, cid
            cap.release()
    return None, None


# =========================
# --------- MAIN ----------
# =========================
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    model = YOLO(MODEL)

    if USE_WEBCAM:
        cap, current_id = find_working_camera()
        if cap is None:
            print(f"❌ No working camera found from candidates {CANDIDATE_IDS}")
            cv2.namedWindow("Webcam")
            blank = np.zeros((240, 680, 3), np.uint8)
            cv2.putText(blank, "No camera available. Check privacy settings / close other apps.",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Webcam", blank); cv2.waitKey(3000); cv2.destroyAllWindows()
            return

        print("✅ Webcam mode: 'q' quit, 's' save frame, 'c' cycle camera")
        counts = deque(maxlen=max(1, int(AVG_WINDOW))) if not USE_EMA else None
        ema_val = None
        forced_cpu = False   # if CUDA fails once, flip this to keep UI alive

        while True:
            ok, frame = cap.read()
            if not ok:
                # quick retry
                time.sleep(0.02)
                ok, frame = cap.read()
                if not ok:
                    blank = np.zeros((320, 640, 3), np.uint8)
                    cv2.putText(blank, f"Camera {current_id}: no frames (press c to switch, q to quit)",
                                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.imshow("Webcam", blank)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue

            # Inference with CPU fallback on exception
            try:
                dev_override = "cpu" if forced_cpu else None
                r = predict_on(model, frame, device_override=dev_override)
            except Exception as e:
                forced_cpu = True
                err = frame.copy() if isinstance(frame, np.ndarray) else np.zeros((320, 640, 3), np.uint8)
                cv2.putText(err, "Inference error; switching to CPU…",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                msg = str(e)
                cv2.putText(err, msg[:80] + ("…" if len(msg) > 80 else ""),
                            (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
                cv2.imshow("Webcam", err)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            polys = masks_to_polys(r)
            count = len(polys)

            # Averages
            if SHOW_AVG:
                if USE_EMA:
                    ema_val = float(count) if ema_val is None else (EMA_ALPHA * float(count) + (1.0 - EMA_ALPHA) * ema_val)
                    avg_val, avg_label = ema_val, "EMA"
                else:
                    counts.append(count)
                    avg_val = (sum(counts) / len(counts)) if counts else 0.0
                    avg_label = f"Avg({len(counts)}/{counts.maxlen})"
            else:
                avg_val, avg_label = None, "Avg"

            rendered, _, _ = render_frame(
                r, polys=polys, draw_hud=SHOW_HUD, window_title="Webcam",
                avg_value=avg_val, avg_label=avg_label
            )

            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            elif k == ord('s'):
                ts = time.strftime("%Y%m%d-%H%M%S")
                suffix = f"_count{count}"
                if avg_val is not None:
                    suffix += f"_{avg_label.lower()}{avg_val:.2f}"
                out_path = os.path.join(SAVE_DIR, f"webcam_{ts}{suffix}.png")
                cv2.imwrite(out_path, rendered); print(f"💾 Saved -> {out_path}")
            elif k == ord('c'):
                # cycle cameras
                try:
                    idx = CANDIDATE_IDS.index(current_id)
                    new_id = CANDIDATE_IDS[(idx + 1) % len(CANDIDATE_IDS)]
                except ValueError:
                    new_id = WEBCAM_ID
                print(f"🔁 Switching camera {current_id} → {new_id}")
                cap.release()
                cap = open_camera(new_id)
                if not cap.isOpened():
                    print(f"❌ Could not open camera {new_id}, reverting…")
                    cap = open_camera(current_id)
                    if not cap.isOpened():
                        print("❌ Lost camera and cannot reopen. Exiting.")
                        break
                else:
                    current_id = new_id

        cap.release()
        cv2.destroyAllWindows()

    else:
        # Single image path (one window)
        r = predict_on(model, IMG)
        rendered, count, (cmin, cmax) = render_frame(
            r, draw_hud=SHOW_HUD, window_title="Image"
        )

        base = os.path.splitext(os.path.basename(IMG))[0]
        if SAVE_FILLED:
            out_filled = os.path.join(SAVE_DIR, f"{base}_filled.png")
            cv2.imwrite(out_filled, rendered)
            print(f"Saved filled  -> {out_filled}")

        if SAVE_OUTLINE:
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
