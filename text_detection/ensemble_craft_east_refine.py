import os
import glob
import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

# ---------------- CONFIG ----------------
IMG_DIR = "test-images/ch4_test_images"
CRAFT_DIR = "outputs/craft"
EAST_DIR = "outputs/east"
OUT_DIR = "outputs/ensemble_refine"

IOU_THRESH = 0.25

os.makedirs(OUT_DIR, exist_ok=True)

# ---------------- UTILS ----------------
def polygon_iou(p1, p2):
    try:
        a, b = Polygon(p1), Polygon(p2)
        if not a.is_valid or not b.is_valid:
            return 0.0
        return a.intersection(b).area / a.union(b).area
    except:
        return 0.0

def read_boxes(txt_path):
    polys, scores = [], []

    if not os.path.exists(txt_path):
        print(f"⚠️ Missing: {txt_path}")
        return polys, scores

    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split(",")

            # Must have at least 8 numbers (polygon)
            if len(parts) < 8:
                continue

            try:
                coords = list(map(float, parts[:8]))
                poly = np.array(coords).reshape(4, 2).tolist()

                # Score is optional
                if len(parts) >= 9:
                    score = float(parts[8])
                else:
                    score = 1.0

                polys.append(poly)
                scores.append(score)

            except Exception:
                continue

    return polys, scores


def merge_polygons(polys):
    """Merge multiple polygons → bounding polygon"""
    merged = unary_union([Polygon(p) for p in polys])
    return np.array(merged.minimum_rotated_rectangle.exterior.coords)[:-1]

# ---------------- MAIN ----------------
images = glob.glob(os.path.join(IMG_DIR, "*.jpg"))

for img_path in images:
    name = os.path.splitext(os.path.basename(img_path))[0]

    craft_polys, _ = read_boxes(f"{CRAFT_DIR}/{name}_craft_boxes.txt")
    east_polys, east_scores = read_boxes(f"{EAST_DIR}/{name}_east.txt")

    used_east = [False] * len(east_polys)
    final_polys = []

    for c_poly in craft_polys:
        matched = False

        for i, e_poly in enumerate(east_polys):
            if used_east[i]:
                continue

            if polygon_iou(c_poly, e_poly) >= IOU_THRESH:
                final_polys.append(e_poly)   # ← EAST refinement
                used_east[i] = True
                matched = True
                break

        if not matched:
            final_polys.append(c_poly)      # ← keep CRAFT box

    # Add remaining EAST boxes (missed by CRAFT)
    for i, e_poly in enumerate(east_polys):
        if not used_east[i]:
            final_polys.append(e_poly)

    # ---------------- SAVE ----------------
    out_txt = f"{OUT_DIR}/{name}_ensemble.txt"
    with open(out_txt, "w") as f:
        for poly in final_polys:
            coords = ",".join([f"{int(x)},{int(y)}" for x, y in poly])
            f.write(f"{coords},1.0000\n")

    # ---------------- VIS ----------------
    img = cv2.imread(img_path)
    for poly in final_polys:
        pts = np.array(poly, int)
        cv2.polylines(img, [pts], True, (0, 255, 0), 2)

    cv2.imwrite(f"{OUT_DIR}/{name}_ensemble.jpg", img)

print("✅ CRAFT-for-Recall + EAST-for-Refinement completed")
print(f"Results saved to {OUT_DIR}/")


# =========================================================
# EVALUATION: Precision / Recall / F1 (ICDAR-style, IoU-based)
# =========================================================

GT_DIR = "test-images/Challenge4_Test_Task1_GT"

def safe_polygon(poly):
    try:
        p = Polygon(poly)
        if not p.is_valid or p.area == 0:
            return None
        return p
    except:
        return None

def polygon_iou_safe(p1, p2):
    a, b = safe_polygon(p1), safe_polygon(p2)
    if a is None or b is None:
        return 0.0
    return a.intersection(b).area / a.union(b).area

def read_gt_boxes(gt_path):
    polys = []
    with open(gt_path, "r", encoding="utf-8-sig") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 8:
                continue
            # Skip "don't care" regions
            if len(parts) > 8 and parts[8].strip() == "###":
                continue
            coords = list(map(float, parts[:8]))
            poly = np.array(coords).reshape(4, 2).tolist()
            polys.append(poly)
    return polys

def read_pred_boxes(pred_path):
    polys = []
    if not os.path.exists(pred_path):
        return polys
    with open(pred_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 8:
                continue
            coords = list(map(float, parts[:8]))
            poly = np.array(coords).reshape(4, 2).tolist()
            polys.append(poly)
    return polys

def evaluate_image(gt_polys, pred_polys):
    matched_gt = set()
    tp, fp = 0, 0

    for pred in pred_polys:
        best_iou = 0
        best_gt = -1
        for i, gt in enumerate(gt_polys):
            if i in matched_gt:
                continue
            iou = polygon_iou_safe(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt = i
        if best_iou >= IOU_THRESH and best_gt != -1:
            tp += 1
            matched_gt.add(best_gt)
        else:
            fp += 1

    fn = len(gt_polys) - len(matched_gt)
    return tp, fp, fn

# ---------------- RUN EVALUATION ----------------
gt_files = glob.glob(os.path.join(GT_DIR, "*.txt"))

agg_tp = agg_fp = agg_fn = 0
images_eval = 0

for gt_path in gt_files:
    fname = os.path.basename(gt_path)
    img_name = fname[3:].replace(".txt", ".jpg")

    pred_path = os.path.join(
        OUT_DIR,
        img_name.replace(".jpg", "_ensemble.txt")
    )

    if not os.path.exists(pred_path):
        continue

    gt_polys = read_gt_boxes(gt_path)
    pred_polys = read_pred_boxes(pred_path)

    tp, fp, fn = evaluate_image(gt_polys, pred_polys)

    agg_tp += tp
    agg_fp += fp
    agg_fn += fn
    images_eval += 1

    if images_eval % 50 == 0:
        print(f"Evaluated {images_eval} images...")

# ---------------- METRICS ----------------
precision = agg_tp / (agg_tp + agg_fp) if (agg_tp + agg_fp) else 0
recall = agg_tp / (agg_tp + agg_fn) if (agg_tp + agg_fn) else 0
f1 = (2 * precision * recall / (precision + recall)
      if (precision + recall) else 0)

print("\n========== ENSEMBLE PERFORMANCE ==========")
print(f"Images Evaluated : {images_eval}")
print(f"True Positives  : {agg_tp}")
print(f"False Positives : {agg_fp}")
print(f"False Negatives : {agg_fn}")
print("------------------------------------------")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")
print("==========================================")
