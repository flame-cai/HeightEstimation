import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from scipy.stats import linregress, pearsonr
from sklearn.metrics import average_precision_score, cohen_kappa_score
from sklearn.metrics import confusion_matrix, roc_auc_score
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
from transformers import Sam2Model, Sam2Processor
import mediapipe as mp
from mediapipe.tasks import python as mpp
from mediapipe.tasks.python import vision as mpv

ROOT = Path(__file__).resolve().parent
CHILD = ["child", "person", "human"]
MARKER = ["post-it note", "sticky note"]
SCALE = ["weighing scale", "digital scale", "weight scale"]
CAM_HEIGHT_CM, SCALE_CM, VALID_CM = 70.0, 6.0, (40.0, 160.0)
HORIZON_FRACTION, HORIZON_SHRINK = 0.48, 0.25
RECUMBENT_ADJ_CM, RECUMBENT_MAX_DAYS = 0.7, 731
STUNTED_Z, AT_RISK_Z = -2.0, -1.0
BOOTSTRAP_N, SEED = 2000, 42

device = "cuda" if torch.cuda.is_available() else "cpu"
gd_proc = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
gd = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base")
gd = gd.to(device).eval()
sam_proc = Sam2Processor.from_pretrained("facebook/sam2.1-hiera-large")
sam = Sam2Model.from_pretrained("facebook/sam2.1-hiera-large").to(device).eval()
pose_options = mpv.PoseLandmarkerOptions(
    base_options=mpp.BaseOptions(model_asset_path=str(ROOT / "pose_landmarker_heavy.task")),
    running_mode=mpv.RunningMode.IMAGE)
pose = mpv.PoseLandmarker.create_from_options(pose_options)

data = pd.read_csv(ROOT / "Data.csv", encoding="utf-8-sig")
data["uuid"] = data.uuid.str.strip()
data = data.sort_values("uuid").reset_index(drop=True)

landmarks = {}
for view in ("front", "side"):
    records = []
    for uuid in data.uuid:
        photo = Image.open(ROOT / "Data" / f"{uuid}_{view}.jpg")
        image = ImageOps.exif_transpose(photo).convert("RGB")
        array = np.array(image)
        H, W = array.shape[:2]
        record = dict(ok=False, image_h=H, vertex=np.nan, feet=np.nan, marker=np.nan,
                      scale_overlap=0.0)

        seg = {}
        for prompt in CHILD + MARKER + SCALE:
            inputs = gd_proc(images=image, text=prompt + ".", return_tensors="pt").to(device)
            with torch.no_grad():
                found = gd_proc.post_process_grounded_object_detection(
                    gd(**inputs), inputs.input_ids, threshold=0.25, text_threshold=0.25,
                    target_sizes=[image.size[::-1]])[0]
            boxes = found["boxes"].cpu().numpy()
            if not len(boxes):
                seg[prompt] = (None, None, None)
                continue
            inputs = sam_proc(images=image, input_boxes=[boxes.tolist()], return_tensors="pt")
            inputs = inputs.to(device)
            with torch.no_grad():
                masks = sam(**inputs, multimask_output=False).pred_masks
            masks = sam_proc.post_process_masks(masks.cpu(), inputs["original_sizes"].cpu())
            masks = masks[0].numpy()
            masks = masks.reshape(-1, *masks.shape[-2:]) > 0
            seg[prompt] = (masks, boxes, found["scores"].cpu().numpy())

        best = None
        for prompt in MARKER:
            masks, _, scores = seg[prompt]
            if masks is None:
                continue
            for mask, score in zip(masks, scores):
                rows = np.flatnonzero(mask.any(axis=1))
                in_band = mask.any() and 0.3 * H <= rows.mean() <= 0.7 * H
                if in_band and mask.mean() <= 0.05 and (best is None or score > best[1]):
                    best = (float(rows[0]), float(score))
        record["marker"] = best[0] if best else np.nan

        child, child_box = None, None
        for prompt in CHILD:
            masks, boxes, _ = seg[prompt]
            if child is not None or masks is None:
                continue
            centre = (boxes[:, 0] + boxes[:, 2]) / 2
            area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) / (W * H)
            for i in np.argsort(np.abs(centre - W / 2) / W - 3 * area):
                if masks[i].mean() >= 0.01:
                    child, child_box = masks[i], boxes[i]
                    break
        if child is None or not child.any():
            records.append(record)
            continue

        rows = np.flatnonzero(child.any(axis=1))
        cols = np.flatnonzero(child.any(axis=0))
        y0, y1, x0, x1 = rows[0], rows[-1], cols[0], cols[-1]
        record["feet"] = float(rows[min(len(rows) - 1, int(0.99 * len(rows)))])
        lo = y0 + int(0.05 * (y1 - y0))
        hi = max(y0 + int(0.15 * (y1 - y0)), lo + 1)
        widths = [child[r].sum() for r in range(lo, hi) if child[r].any()]
        half_width = float(np.median(widths)) / 2 if widths else np.nan

        head_column = None
        crop = np.ascontiguousarray(array[y0:y1, x0:x1])
        if crop.shape[0] >= 50 and crop.shape[1] >= 50:
            try:
                frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop)
                people = pose.detect(frame).pose_landmarks
                if people:
                    head_column = int(x0 + people[0][0].x * crop.shape[1])
            except Exception:
                head_column = None
        if head_column is None:
            head = child[y0:y0 + max(1, int(0.20 * (y1 - y0)))]
            head_column = int(head.sum(axis=0).argmax()) if head.any() else (x0 + x1) // 2

        best_scale = (0.0, 0.0)
        for prompt in SCALE:
            _, boxes, scores = seg[prompt]
            if boxes is None:
                continue
            for box, score in zip(boxes, scores):
                shared = max(0.0, min(box[2], child_box[2]) - max(box[0], child_box[0]))
                overlap = shared / max(1e-6, child_box[2] - child_box[0])
                if score > best_scale[0]:
                    best_scale = (float(score), float(overlap))
        record["scale_overlap"] = best_scale[1]

        half_band = max(2, int(round(half_width))) if np.isfinite(half_width) else 12
        strip = child[:, max(0, head_column - half_band):min(W, head_column + half_band + 1)]
        strip_rows = np.flatnonzero(strip.any(axis=1))
        record["vertex"] = float(strip_rows[0]) if len(strip_rows) else float(rows[0])
        record["ok"] = True
        records.append(record)
    landmarks[view] = pd.DataFrame(records)
pose.close()
print(f"{len(data)} children, {2 * len(data)} photographs processed on {device}")

day = pd.to_datetime(data.timestamp).dt.date.astype(str)
session = day + " " + data.lat.round(3).astype(str)
on_scale = np.maximum(landmarks["front"].scale_overlap, landmarks["side"].scale_overlap) > 0.5
camera_height = np.where(on_scale, CAM_HEIGHT_CM - SCALE_CM, CAM_HEIGHT_CM)
view_heights = []
for view in ("front", "side"):
    f = landmarks[view]
    marker_fraction = pd.Series(f.marker / f.image_h)
    horizon = 0 * f.marker + marker_fraction.groupby(session).transform("median") * f.image_h
    horizon = (1 - HORIZON_SHRINK) * horizon + HORIZON_SHRINK * HORIZON_FRACTION * f.image_h
    stature, camera_px = f.feet - f.vertex, f.feet - horizon
    valid = f.ok & (stature > 0) & (camera_px > 0)
    height = np.where(valid, camera_height * stature / camera_px, np.nan)
    height = np.where((height > VALID_CM[0]) & (height < VALID_CM[1]), height, np.nan)
    view_heights.append(np.round(height, 2))
predicted = np.nanmean(np.vstack(view_heights), axis=0)
measured = data.height_cm_self_reported.to_numpy(float)

results = pd.DataFrame({"id": data.uuid, "age_days": data.age_days, "gender": data.gender,
                        "height_cm": measured, "predicted_height_cm": np.round(predicted, 2)})
results.to_csv(ROOT / "results.csv", index=False)
print(f"wrote results.csv ({len(data)} rows)")

sex = data.gender.astype(str).str.strip().str.upper().str[:1]
if not sex.isin(["M", "F"]).all():
    unknown = sorted(set(data.gender[~sex.isin(["M", "F"])]))
    sys.exit(f"unrecognised sex {unknown}; expected Male/Female")
who = {s: pd.read_excel(ROOT / "who_tables" / f"lhfa_{s}_expanded.xlsx").set_index("Day")
       for s in ("boys", "girls")}
lms = [who["boys" if s == "M" else "girls"].loc[int(round(a)), ["L", "M", "S"]]
       for s, a in zip(sex, data.age_days)]
L, M, S = np.array(lms, dtype=float).T
adjust = np.where(data.age_days < RECUMBENT_MAX_DAYS, RECUMBENT_ADJ_CM, 0.0)
z_measured = (((measured + adjust) / M) ** L - 1) / (L * S)
z_predicted = (((predicted + adjust) / M) ** L - 1) / (L * S)

ok = np.isfinite(predicted)
n = int(ok.sum())
meas, pred = measured[ok], predicted[ok]
z_meas, z_pred = z_measured[ok], z_predicted[ok]
err = pred - meas
truth = (z_meas < STUNTED_Z).astype(int)
guess = (z_pred < STUNTED_Z).astype(int)
tn, fp, fn, tp = confusion_matrix(truth, guess, labels=[0, 1]).ravel()

estimates = {
    "mae": np.abs(err).mean(),
    "rmse": np.sqrt((err ** 2).mean()),
    "bias": err.mean(),
    "sd": err.std(ddof=1),
    "r": pearsonr(pred, meas)[0],
    "ccc": 2 * np.cov(pred, meas, ddof=0)[0, 1]
           / (pred.var() + meas.var() + (pred.mean() - meas.mean()) ** 2),
    "sens": tp / (tp + fn) * 100,
    "spec": tn / (tn + fp) * 100,
    "ppv": tp / (tp + fp) * 100,
    "npv": tn / (tn + fn) * 100,
    "accuracy": (tp + tn) / n * 100,
    "f1": 2 * tp / (2 * tp + fp + fn) * 100,
    "kappa": cohen_kappa_score(truth, guess),
    "roc_auc": roc_auc_score(truth, -z_pred),
    "pr_auc": average_precision_score(truth, -z_pred),
}

rng = np.random.default_rng(SEED)
draws = {key: [] for key in estimates}
for _ in range(BOOTSTRAP_N):
    i = rng.choice(n, n, replace=True)
    e, p, m, t, g, zp = err[i], pred[i], meas[i], truth[i], guess[i], z_pred[i]
    for key in estimates:
        try:
            if key == "mae":
                value = np.abs(e).mean()
            elif key == "rmse":
                value = np.sqrt((e ** 2).mean())
            elif key == "bias":
                value = e.mean()
            elif key == "sd":
                value = e.std(ddof=1)
            elif key == "r":
                value = pearsonr(p, m)[0]
            elif key == "ccc":
                spread = p.var() + m.var() + (p.mean() - m.mean()) ** 2
                value = 2 * np.cov(p, m, ddof=0)[0, 1] / spread
            elif key == "sens":
                value = ((t == 1) & (g == 1)).sum() / max(1, (t == 1).sum()) * 100
            elif key == "spec":
                value = ((t == 0) & (g == 0)).sum() / max(1, (t == 0).sum()) * 100
            elif key == "ppv":
                value = ((t == 1) & (g == 1)).sum() / max(1, (g == 1).sum()) * 100
            elif key == "npv":
                value = ((t == 0) & (g == 0)).sum() / max(1, (g == 0).sum()) * 100
            elif key == "accuracy":
                value = (t == g).mean() * 100
            elif key == "f1":
                hits = ((t == 1) & (g == 1)).sum()
                misses = ((t == 0) & (g == 1)).sum() + ((t == 1) & (g == 0)).sum()
                value = 2 * hits / max(1, 2 * hits + misses) * 100
            elif key == "kappa":
                value = cohen_kappa_score(t, g)
            elif key == "roc_auc":
                value = roc_auc_score(t, -zp)
            else:
                value = average_precision_score(t, -zp)
        except Exception:
            continue
        if np.isfinite(value):
            draws[key].append(value)
ci = {key: np.percentile(values, [2.5, 97.5]) for key, values in draws.items()}

labels = {
    "mae": ("Mean absolute error, cm", ".2f"),
    "rmse": ("Root mean square error, cm", ".2f"),
    "bias": ("Mean signed error (bias), cm", "+.2f"),
    "sd": ("SD of differences, cm", ".2f"),
    "r": ("Pearson r", ".3f"),
    "ccc": ("Lin concordance correlation", ".3f"),
    "sens": ("Sensitivity, %", ".1f"),
    "spec": ("Specificity, %", ".1f"),
    "ppv": ("Positive predictive value, %", ".1f"),
    "npv": ("Negative predictive value, %", ".1f"),
    "accuracy": ("Accuracy, %", ".1f"),
    "f1": ("F1-score, %", ".1f"),
    "kappa": ("Cohen kappa", ".2f"),
    "roc_auc": ("ROC AUC", ".3f"),
    "pr_auc": ("PR AUC", ".3f"),
}

print(f"\nTABLE 1 (n={n}; 95% CI from {BOOTSTRAP_N} bootstrap resamples)")
for key, (label, fmt) in labels.items():
    low, high = ci[key]
    print(f"  {label:<31}{estimates[key]:{fmt}}  ({low:{fmt}} to {high:{fmt}})")
print(f"  {'TP / FN / FP / TN':<31}{tp} / {fn} / {fp} / {tn}")

fit = linregress((pred + meas) / 2, err)
low, high = np.percentile(err, [2.5, 97.5])
print("\nAGREEMENT")
print(f"  95% limits of agreement        {low:+.2f} to {high:+.2f} cm")
print(f"  proportional bias              slope {fit.slope:+.4f} cm/cm, P={fit.pvalue:.2f}")
print(f"  prevalence of stunting         {truth.mean() * 100:.1f}%")

band_meas = np.where(z_meas < STUNTED_Z, 0, np.where(z_meas < AT_RISK_Z, 1, 2))
band_pred = np.where(z_pred < STUNTED_Z, 0, np.where(z_pred < AT_RISK_Z, 1, 2))
shift = band_pred - band_meas
print("\n3 WHO BANDS (rows measured, columns estimated: stunted, at risk, normal)")
matrix = confusion_matrix(band_meas, band_pred, labels=[0, 1, 2])
for name, row in zip(["stunted", "at risk", "normal"], matrix):
    print(f"  {name:<12}" + "".join(f"{count:5d}" for count in row))
print(f"  accuracy                       {(band_meas == band_pred).mean() * 100:.1f}%")
quadratic_kappa = cohen_kappa_score(band_meas, band_pred, weights="quadratic")
print(f"  quadratic Cohen kappa          {quadratic_kappa:.2f}")
print(f"  1 band more severe             {int((shift == -1).sum())}")
print(f"  1 band less severe             {int((shift == 1).sum())}")
print(f"  2 bands apart                  {int((np.abs(shift) == 2).sum())}")
