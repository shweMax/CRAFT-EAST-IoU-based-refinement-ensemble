#!/usr/bin/env python
# coding: utf-8

import os
import glob
import cv2
import math
import numpy as np

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

MODEL_PATH = 'models/frozen_east_text_detection.pb'
IMAGE_DIR = 'test-images/ch4_test_images'
OUTPUT_DIR = 'outputs/east'

CONF_THRESHOLD = 0.9
NMS_THRESHOLD = 0.2
MAX_SIDE = 1280

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------

def resize_image(image, max_side=1280):
    h, w = image.shape[:2]
    ratio = min(max_side / max(h, w), 1.0)
    new_w = int(w * ratio)
    new_h = int(h * ratio)
    new_w = new_w - new_w % 32
    new_h = new_h - new_h % 32
    resized = cv2.resize(image, (new_w, new_h))
    return resized, w / new_w, h / new_h


def decode(scores, geometry, score_thresh):
    detections = []
    confidences = []

    height, width = scores.shape[2:4]

    for y in range(height):
        for x in range(width):
            score = scores[0, 0, y, x]
            if score < score_thresh:
                continue

            offset_x, offset_y = x * 4.0, y * 4.0
            angle = geometry[0, 4, y, x]
            cos = math.cos(angle)
            sin = math.sin(angle)

            h = geometry[0, 0, y, x] + geometry[0, 2, y, x]
            w = geometry[0, 1, y, x] + geometry[0, 3, y, x]

            end_x = int(offset_x + cos * geometry[0, 1, y, x] + sin * geometry[0, 2, y, x])
            end_y = int(offset_y - sin * geometry[0, 1, y, x] + cos * geometry[0, 2, y, x])
            start_x = int(end_x - w)
            start_y = int(end_y - h)

            detections.append((start_x, start_y, end_x, end_y))
            confidences.append(float(score))

    return detections, confidences


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

print("[INFO] Loading EAST model...")
net = cv2.dnn.readNet(MODEL_PATH)

image_paths = glob.glob(os.path.join(IMAGE_DIR, "*"))

for img_path in image_paths:
    image = cv2.imread(img_path)
    if image is None:
        continue

    resized, rW, rH = resize_image(image, MAX_SIDE)

    blob = cv2.dnn.blobFromImage(
        resized, 1.0, resized.shape[1::-1],
        (123.68, 116.78, 103.94), swapRB=True, crop=False
    )

    net.setInput(blob)
    scores, geometry = net.forward([
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"
    ])

    boxes, confidences = decode(scores, geometry, CONF_THRESHOLD)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)

    base = os.path.splitext(os.path.basename(img_path))[0]
    txt_path = os.path.join(OUTPUT_DIR, base + "_east.txt")

    with open(txt_path, "w") as f:
        if len(indices) > 0:
            for i in indices.flatten():
                x1, y1, x2, y2 = boxes[i]
                x1 = int(x1 * rW)
                y1 = int(y1 * rH)
                x2 = int(x2 * rW)
                y2 = int(y2 * rH)
                score = confidences[i]
                f.write(f"{x1},{y1},{x2},{y2},{score:.4f}\n")

    print(f"[EAST] Saved: {txt_path}")

print("✅ EAST completed")
