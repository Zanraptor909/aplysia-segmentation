from ultralytics import YOLO
import cv2
import numpy as np
import os
from glob import glob
from tqdm import tqdm

# =========================
# ---- CONFIGURABLES ------
# =========================
MODEL     = r"C:\TakeFive_Its_A_Vibe\runs\seg_v8s\weights\best.pt"  # trained weights
IMG       = r"C:\TakeFive_Its_A_Vibe\test_image\test_image_2.JPG"        # image or folder
SAVE_DIR  = r"C:\TakeFive_Its_A_Vibe\test_image\predict_out"             # output folder

# Inference knobs (tune these to reduce false positives / “too many slugs”)
IMG_SIZE      = 1280        # larger → finer masks; 640/960/1280 are typical
CONF_MIN      = 0.60      # ↑ increase to be stricter (e.g., 0.25–0.40)
IOU_NMS       = 0.30       # lower can split overlaps more; higher merges more
MAX_DET       = 2000       # cap detections per image
RETINA_MASKS  = True       # higher-res masks at some cost
DEVICE        = 0          # 0 for first GPU, "cpu" for CPU
HALF          = True       # FP16 on GPU (saves VRAM; keep False on CPU)

# Rendering controls
SAVE_FILLED   = True
SAVE_OUTLINE  = True
COLOR         = (0, 255, 255)   # BGR: yellow
ALPHA         = 0.35            # fill opacity
THICK         = 2               # outline thickness

# Input collection
IMG_EXTS      = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# =========================
# ====== FUNCTIONS ========
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

def collect_images(path):
    """Return a list of image paths from a single file or a directory."""
    if os.path.isdir(path):
        imgs = []
        for ext in IMG_EXTS:
            imgs.extend(glob(os.path.join(path, f"*{ext}")))
        imgs.sort()
        return imgs
    else:
        return [path]

def run_inference(model, img_path):
    """Run YOLO on one image with configured settings and return (r, orig)."""
    results = model(
        img_path,
        imgsz=IMG_SIZE,
        conf=CONF_MIN,
        iou=IOU_NMS,
        max_det=MAX_DET,
        device=DEVICE,
        half=HALF if DEVICE != "cpu" else False,
        retina_masks=RETINA_MASKS,
        verbose=False,
    )
    r = results[0]
    orig = r.orig_img.copy()
    return r, orig

def render_outputs(r, orig, base_name):
    """Render and save filled/outlined outputs. Return info dictionary."""
    polys = masks_to_polys(r)

    out_paths = {}
    if SAVE_FILLED:
        filled = orig.copy()
        if polys:
            overlay = orig.copy()
            cv2.fillPoly(overlay, polys, COLOR)
            filled = cv2.addWeighted(overlay, ALPHA, orig, 1 - ALPHA, 0)
        out_filled = os.path.join(SAVE_DIR, f"{base_name}_filled.png")
        cv2.imwrite(out_filled, filled)
        out_paths["filled"] = out_filled

    if SAVE_OUTLINE:
        outlined = orig.copy()
        if polys:
            cv2.polylines(outlined, polys, isClosed=True, color=COLOR, thickness=THICK)
        out_outline = os.path.join(SAVE_DIR, f"{base_name}_outline.png")
        cv2.imwrite(out_outline, outlined)
        out_paths["outline"] = out_outline

    # Count + confidences
    num_instances = 0
    conf_min = conf_max = None
    names = []
    confs = []

    if r.boxes is not None:
        num_instances = len(r.boxes)
        confs = r.boxes.conf.tolist()
        cls_ids = r.boxes.cls.tolist()
        names = [r.names[int(c)] for c in cls_ids]
        if confs:
            conf_min, conf_max = min(confs), max(confs)
    elif r.masks is not None and hasattr(r.masks, "data"):
        num_instances = r.masks.data.shape[0]

    return {
        "num_instances": num_instances,
        "conf_min": conf_min,
        "conf_max": conf_max,
        "names": names,
        "confs": confs,
        "out_paths": out_paths,
    }

# =========================
# ========= MAIN ==========
# =========================
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Load once
    model = YOLO(MODEL)

    images = collect_images(IMG)
    if not images:
        print(f"No images found at: {IMG}")
        return

    # Progress bar over images
    for img_path in tqdm(images, desc="Processing images", unit="img"):
        try:
            r, orig = run_inference(model, img_path)
            base = os.path.splitext(os.path.basename(img_path))[0]
            info = render_outputs(r, orig, base)

            # Console summary for each image
            print(f"\nImage: {img_path}")
            for k, v in info["out_paths"].items():
                print(f"Saved {k:7} -> {v}")
            print(f"Detected {info['num_instances']} objects")

            if info["confs"]:
                print("Classes:", info["names"])
                print(
                    f"Confidence range: {info['conf_min']:.3f} – {info['conf_max']:.3f}"
                )
                print("Confidences:", [round(c, 3) for c in info["confs"]])

        except Exception as e:
            print(f"[WARN] Failed on {img_path}: {e}")

if __name__ == "__main__":
    main()
