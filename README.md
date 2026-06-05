# Estimating Pediatric Height From Monocular RGB Photographs for Child Nutrition Screening

Photograph-based pediatric standing height estimation using image segmentation, pose landmark detection, and geometric modeling.

---

## Overview

This repository accompanies the research letter **"Estimating Pediatric Height From Monocular RGB Photographs for Child Nutrition Screening"** and contains the implementation of a computer vision pipeline for estimating pediatric standing height from monocular RGB photographs.

The proposed framework combines image segmentation, pose landmark detection, and geometric modeling to estimate standing height using ordinary smartphone photographs. The method requires a reference object positioned at a known height above ground level and does not require specialized depth sensors, stereo cameras, or dedicated anthropometric equipment.

For each participant, front-view and side-view photographs are acquired using a fixed camera setup. A reference marker is positioned such that its upper edge is located at a known height above the ground plane. This known reference height is subsequently used for geometric height estimation.

In the study accompanying this repository, a post-it marker was used as the reference object and positioned with its upper edge exactly 70 cm above ground level.

The processing pipeline consists of five major stages:

1. Child segmentation
2. Reference marker detection
3. Head localization
4. Foot localization
5. Geometric height estimation and multi-view ensembling

---

## Methodology

### Child Segmentation

Child body regions are extracted using a segmentation framework combining Grounding DINO and the Segment Anything Model (SAM).

To improve robustness across image conditions, multiple prompts are evaluated sequentially:

```python
["child", "person", "human"]
```

For each prompt, the framework generates candidate segmentation masks and bounding boxes. Candidate detections are ranked according to both object size and proximity to the image center.

The ranking score is computed as:

[
Score =
\frac{|x_c - x_{center}|}{W}
----------------------------

3
\left(
\frac{A_{box}}{A_{image}}
\right)
]

where:

* (x_c) is the bounding-box center,
* (x_{center}) is the image center,
* (W) is image width,
* (A_{box}) is bounding-box area,
* (A_{image}) is total image area.

This ranking strategy favors large, centrally positioned detections that are likely to correspond to the photographed child.

Segmentation masks occupying less than 1% of the image area are discarded to reduce false detections. The highest-ranked valid mask is selected for subsequent processing.

---

### Reference Marker Detection

A reference marker serves as the geometric anchor required for monocular height estimation.

The marker is detected using the same segmentation framework with the prompts:

```python
["post-it note", "sticky note"]
```

For the selected marker mask, the uppermost foreground pixel row is extracted and defined as:

[
midPx
]

which represents the image row corresponding to the known reference height above ground level.

#### Fallback Strategy

When the marker cannot be detected, the reference row is approximated using a fixed fraction of image height:

[
midPx = 0.484 \times H_{image}
]

where (H_{image}) denotes image height in pixels.

This fallback value was empirically determined for the image acquisition setup used in the study.

---

### Head Localization

Accurate identification of the top of the head is critical for reliable anthropometric estimation.

The child segmentation mask is first cropped to its bounding region. Pose landmark detection is subsequently performed using the BlazePose model through the MediaPipe Pose framework.

The nose landmark is extracted and used to estimate the horizontal center of the head:

[
headCol
]

A narrow vertical band centered around this location is then examined within the segmentation mask.

The uppermost foreground pixel contained within this band is identified as:

[
topPx
]

Restricting the search to the facial region reduces susceptibility to segmentation artifacts, raised-arm postures, and background objects that might otherwise influence estimation of the upper body boundary.

---

### Foot Localization

The lower body boundary is estimated from the child segmentation mask.

Rather than selecting the absolute lowest foreground pixel, the algorithm uses the 98th percentile of foreground row indices:

[
botPx
]

This approach improves robustness against segmentation noise and isolated outlier pixels near the feet while preserving a stable estimate of the lower body boundary.

---

### Geometric Height Estimation

Height estimation is performed using a pinhole-camera geometric model.

For each image view, projected body height is computed as:

[
P_p = botPx - topPx
]

where (P_p) denotes projected body height in image space.

The projected distance between the reference row and the detected foot position is computed as:

[
P_c = botPx - midPx
]

Using the known reference height (H_c), height is estimated as:

[
H_i =
\frac{H_c \times P_p}{P_c}
]

where:

* (H_i) is the estimated height,
* (H_c) is the known reference height,
* (P_p) is projected body height,
* (P_c) is projected reference distance.

In the study accompanying this repository, (H_c = 70) cm.

Predictions associated with invalid geometric configurations ((P_p \le 0) or (P_c \le 0)) are discarded.

---

### Multi-View Ensembling

Independent height estimates are generated from front-view and side-view photographs.

The final height prediction is computed using a weighted ensemble:

[
H_{ensemble}
============

\frac{
\sum_i H_i P_{c,i}
}{
\sum_i P_{c,i}
}
]

where:

* (H_i) denotes the height estimate from image view (i),
* (P_{c,i}) denotes the corresponding projected reference distance.

This weighting strategy assigns greater influence to predictions associated with larger geometric reference distances and improves robustness against view-specific segmentation or landmark localization errors.

If only one valid prediction is available, that prediction is used directly.

---

## Evaluation

Predicted heights are compared against manually measured anthropometric heights.

The repository reports the following evaluation metrics:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* Mean Absolute Percentage Error (MAPE)
* Pearson Correlation Coefficient (r)

These metrics are computed using all participants with valid ensemble predictions.

---

## Assumptions and Limitations

The current implementation assumes:

* A fixed camera configuration.
* A reference marker positioned at a known height above ground level.
* Full visibility of the child body.
* Standing posture during image acquisition.
* Adequate image quality for segmentation and landmark detection.

Performance may degrade when these assumptions are violated. Additional validation is required under varying camera positions, lighting conditions, marker placements, and field deployment settings.

---

## References

1. Liu S, Zeng Z, Ren T, et al. **Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection.** *arXiv*. 2023. doi:10.48550/arXiv.2303.05499

2. Ravi N, Gabeur V, Hu YT, et al. **SAM 2: Segment Anything in Images and Videos.** *arXiv*. 2024. doi:10.48550/arXiv.2408.00714

3. Lugaresi C, Tang J, Nash H, et al. **MediaPipe: A Framework for Building Perception Pipelines.** *arXiv*. 2019. doi:10.48550/arXiv.1906.08172

4. Bazarevsky V, Grishchenko I, Raveendran K, et al. **BlazePose: On-Device Real-Time Body Pose Tracking.** *arXiv*. 2020. doi:10.48550/arXiv.2006.10204

---
