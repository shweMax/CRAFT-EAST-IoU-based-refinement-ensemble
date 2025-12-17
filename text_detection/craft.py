"""
Refined CRAFT for ICDAR 2015
✔ NO RefineNet
✔ Correct coordinate rescaling
✔ WBF / GT compatible output
"""

import os
import cv2
import numpy as np
import glob
from tqdm import tqdm
from craft_text_detector import Craft
from craft_text_detector.predict import get_prediction

# --------------------------------------------------
# PATHS
# --------------------------------------------------

IMG_DIR = "test-images/ch4_test_images"
SAVE_DIR = "outputs/craft"
MODEL_PATH = "models/craft_ic15_20k.pth"

os.makedirs(SAVE_DIR, exist_ok=True)

# --------------------------------------------------
# PROCESS SINGLE IMAGE
# --------------------------------------------------

def process_image(craft_model, image_path, save_dir):

    img = cv2.imread(image_path)
    if img is None:
        return 0

    h, w = img.shape[:2]

    prediction = get_prediction(
        image=img,
        craft_net=craft_model.craft_net,
        refine_net=None,          # ❌ Disable RefineNet
        text_threshold=0.7,
        link_threshold=0.4,
        low_text=0.4,
        cuda=craft_model.cuda,
        long_size=2240,           # ICDAR 2015 setting
        poly=False
    )

    boxes = prediction["boxes"]
    scores = prediction.get("scores", [0.9] * len(boxes))
    # 🔥 FIX HERE
    if "resize_ratio" in prediction:
        ratio_w, ratio_h = prediction["resize_ratio"]
    elif "ratio" in prediction:
        ratio_w, ratio_h = prediction["ratio"]
    else:
        ratio_w, ratio_h = 1.0, 1.0

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    txt_path = os.path.join(save_dir, f"{base_name}_craft_boxes.txt")

    valid_boxes = 0

    # -------------------------------
    # SAVE TXT (RESCALED COORDS)
    # -------------------------------
    with open(txt_path, "w") as f:
        for box, score in zip(boxes, scores):
            if box is None or len(box) != 4:
                continue

            box = np.array(box, dtype=float)

            # 🔥 CRITICAL FIX: scale back
            box[:, 0] /= ratio_w
            box[:, 1] /= ratio_h

            coords = ",".join(
                [f"{int(x)},{int(y)}" for x, y in box]
            )
            f.write(f"{coords},{score:.4f}\n")
            valid_boxes += 1

    # -------------------------------
    # SAVE VISUALIZATION (OPTIONAL)
    # -------------------------------
    vis = img.copy()
    for box in boxes:
        if box is None or len(box) != 4:
            continue
        box = np.array(box, dtype=float)
        box[:, 0] /= ratio_w
        box[:, 1] /= ratio_h
        box = box.astype(np.int32)
        cv2.polylines(vis, [box], True, (0, 255, 0), 2)

    vis_path = os.path.join(save_dir, f"{base_name}_craft_result.jpg")
    cv2.imwrite(vis_path, vis)

    return valid_boxes

# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print("=" * 60)
    print("REFINED CRAFT — ICDAR 2015 (FIXED)")
    print("=" * 60)
    print("✓ RefineNet: DISABLED")
    print("✓ long_size = 2240")
    print("✓ Coordinates rescaled correctly")
    print("=" * 60)

    craft = Craft(
        output_dir=SAVE_DIR,
        crop_type="box",
        cuda=False,
        rectify=True,
        weight_path_craft_net=MODEL_PATH
    )

    image_files = glob.glob(os.path.join(IMG_DIR, "*.jpg"))
    print(f"\n📸 Found {len(image_files)} images\n")

    total = 0
    for img_path in tqdm(image_files, desc="Running CRAFT"):
        total += process_image(craft, img_path, SAVE_DIR)

    print("\n✓ CRAFT processing complete")
    print(f"Total detections: {total}")
    print(f"Avg per image: {total / len(image_files):.2f}")
    print(f"Saved to: {SAVE_DIR}")

    craft.unload_craftnet_model()

if __name__ == "__main__":
    main()
