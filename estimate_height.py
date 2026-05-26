"""
Photograph-Based Pediatric Height Estimation Pipeline
=====================================================
"""

import os
import csv
import warnings
import numpy as np
import pandas as pd
import cv2
from PIL import Image, ImageOps
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
)

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIG
# ============================================================================

INPUT_DIR = "Data"
CSV_PATH = "Data.csv"
OUTPUT_CSV = "results.csv"
CAMERA_HEIGHT_CM = 70.0
MIDPX_FALLBACK_FRAC = 0.484
HEAD_BAND = 12

# ============================================================================
# LOAD MODELS
# ============================================================================

print("Loading models...")

from lang_sam import LangSAM
import mediapipe as mp

segmenter = LangSAM()
pose = mp.solutions.pose.Pose(static_image_mode=True)

print("Models loaded.")

# ============================================================================
# HELPERS
# ============================================================================


def load_image(path):
    image = Image.open(path)
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image, np.array(image)


# ============================================================================
# SEGMENTATION
# ============================================================================


def langsam_predict(image, prompt):
    try:
        results = segmenter.predict([image], [prompt])
    except Exception:
        return None, None

    if not results or not results[0]:
        return None, None

    r = results[0]
    masks = r.get("masks", None)
    boxes = r.get("boxes", None)

    if masks is None or boxes is None:
        return None, None

    masks = np.array(masks)
    boxes = np.array(boxes)

    if masks.ndim < 3 or masks.shape[0] == 0:
        return None, None

    return masks, boxes


def rank_candidates(masks, boxes, width, height):
    center_x = width / 2.0
    total_px = width * height

    box_cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
    box_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    dist = np.abs(box_cx - center_x) / width
    area = box_area / total_px

    score = dist - 3.0 * area

    return np.argsort(score)


def detect_child_mask(image, image_np):
    prompts = ["child", "person", "human"]

    H, W = image_np.shape[:2]

    for prompt in prompts:
        masks, boxes = langsam_predict(image, prompt)

        if masks is None:
            continue

        ranked = rank_candidates(masks, boxes, W, H)

        for idx in ranked:
            mask = masks[idx].astype(bool)

            if mask.mean() < 0.01:
                continue

            return mask

    return None


# ============================================================================
# POST-IT DETECTION
# ============================================================================


def detect_postit(image):
    prompts = ["post-it note", "sticky note"]

    for prompt in prompts:
        masks, boxes = langsam_predict(image, prompt)

        if masks is None:
            continue

        for mask in masks:
            mask = mask.astype(bool)
            rows = np.where(mask.any(axis=1))[0]

            if len(rows) == 0:
                continue

            return float(rows[0])

    return None


# ============================================================================
# MEDIAPIPE HEAD COLUMN
# ============================================================================


def get_head_column(image_np, child_mask):
    rows = np.where(child_mask.any(axis=1))[0]
    cols = np.where(child_mask.any(axis=0))[0]

    if len(rows) == 0 or len(cols) == 0:
        return None

    y0, y1 = rows[0], rows[-1]
    x0, x1 = cols[0], cols[-1]

    crop = image_np[y0:y1, x0:x1]

    if crop.shape[0] < 50 or crop.shape[1] < 50:
        return None

    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    result = pose.process(crop_rgb)

    if not result.pose_landmarks:
        return None

    landmarks = result.pose_landmarks.landmark

    nose = landmarks[0]

    head_col = int(x0 + nose.x * crop.shape[1])

    return head_col


# ============================================================================
# BODY EXTENTS
# ============================================================================


def head_top_from_mask(mask, head_col):
    H, W = mask.shape

    col_lo = max(0, head_col - HEAD_BAND)
    col_hi = min(W - 1, head_col + HEAD_BAND)

    band = mask[:, col_lo:col_hi + 1]

    rows = np.where(band.any(axis=1))[0]

    if len(rows) == 0:
        return None

    return float(rows[0])



def body_extent(mask, head_col):
    topPx = head_top_from_mask(mask, head_col)

    rows = np.where(mask.any(axis=1))[0]

    if len(rows) == 0:
        return None, None

    botPx = float(rows[int(0.98 * len(rows))])

    return topPx, botPx


# ============================================================================
# HEIGHT GEOMETRY
# ============================================================================


def compute_height(topPx, botPx, midPx):
    P_p = botPx - topPx
    P_c = botPx - midPx

    if P_p <= 0 or P_c <= 0:
        return None, None

    height = (CAMERA_HEIGHT_CM * P_p) / P_c

    return height, P_c


# ============================================================================
# ENSEMBLE
# ============================================================================


def ensemble(front_h, front_pc, side_h, side_pc):
    values = []
    weights = []

    if front_h is not None:
        values.append(front_h)
        weights.append(front_pc)

    if side_h is not None:
        values.append(side_h)
        weights.append(side_pc)

    if not values:
        return None

    return float(np.average(values, weights=weights))


# ============================================================================
# SINGLE IMAGE PIPELINE
# ============================================================================


def process_image(path):
    if not os.path.exists(path):
        return None, None

    image, image_np = load_image(path)

    H = image_np.shape[0]

    midPx = detect_postit(image)

    if midPx is None:
        midPx = MIDPX_FALLBACK_FRAC * H

    child_mask = detect_child_mask(image, image_np)

    if child_mask is None:
        return None, None

    head_col = get_head_column(image_np, child_mask)

    if head_col is None:
        return None, None

    topPx, botPx = body_extent(child_mask, head_col)

    if topPx is None or botPx is None:
        return None, None

    return compute_height(topPx, botPx, midPx)


# ============================================================================
# EVALUATION
# ============================================================================


def regression_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    mape = np.mean(
        np.abs((np.array(y_true) - np.array(y_pred)) / np.array(y_true))
    ) * 100

    r, _ = pearsonr(y_true, y_pred)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "Pearson_r": r,
    }


# ============================================================================
# MAIN PIPELINE
# ============================================================================


def main():
    df = pd.read_csv(CSV_PATH)

    outputs = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        uuid = row["uuid"]

        front_path = os.path.join(INPUT_DIR, f"{uuid}_front.jpg")
        side_path = os.path.join(INPUT_DIR, f"{uuid}_side.jpg")

        front_h, front_pc = process_image(front_path)
        side_h, side_pc = process_image(side_path)

        final_h = ensemble(front_h, front_pc, side_h, side_pc)

        outputs.append({
            "uuid": uuid,
            "actual_height": row["height_cm"],
            "front_height": front_h,
            "side_height": side_h,
            "ensemble": final_h,
        })

    results = pd.DataFrame(outputs)
    results.to_csv(OUTPUT_CSV, index=False)

    valid = results.dropna(subset=["ensemble", "actual_height"])

    metrics = regression_metrics(
        valid["actual_height"],
        valid["ensemble"]
    )

    print("\nRegression Metrics")
    print("==================")

    for k, v in metrics.items():
        print(f"{k}: {v:.3f}")


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    main()
