# Photograph-Based Pediatric Height Estimation

Monocular RGB image-based pediatric standing height estimation and LHFA nutritional risk screening using computer vision and geometric modeling techniques.

---

# Methodology

The proposed study developed a photograph-based pediatric height estimation framework using monocular RGB images and geometric modeling techniques. For each participant, front-view and side-view photographs were captured using a fixed camera setup. A post-it marker was pasted such that its upper edge was positioned exactly 70 cm above the ground level, thereby serving as the optical reference point for geometric calculations. All images were orientation-corrected using EXIF metadata and converted to RGB format before further processing.

Child body regions and post-it reference markers were extracted using a segmentation framework combining Grounding DINO and SAM. The segmentation pipeline generated multiple candidate detections for each object category. Candidate masks were ranked according to object size and proximity to the image center, and the highest-ranked valid segmentation mask was selected for subsequent processing. Child masks occupying less than 1% of the image area were rejected to reduce false detections.

The upper edge of the detected post-it marker was used to estimate the optical reference row (`midPx`). When the post-it marker could not be detected, the midpoint was approximated using a fixed fraction of the image height:

```math
midPx = 0.484 \times H_{image}
```

where `H_image` denotes the image height in pixels.

To improve robustness against posture variations and raised-arm artifacts, MediaPipe Pose and BlazePose were employed to estimate facial landmarks including the nose and ears. The detected child mask was first cropped to its bounding region before pose estimation was performed. The horizontal head center was estimated using the detected nose landmark, and the top of the head (`topPx`) was determined by vertically scanning a narrow column band around the estimated head center within the segmentation mask. This approach reduced the likelihood of raised hands or background artifacts being incorrectly identified as the upper body boundary.

The lower body boundary (`botPx`) was estimated from the lower extent of the segmentation mask. Instead of selecting the absolute lowest foreground pixel, the 98th percentile of foreground row indices was used to improve robustness against segmentation noise and isolated outlier pixels near the feet.

Height estimation was performed using a pinhole camera geometry model. For each image view, the projected body height in image space was computed as:

```math
P_p = botPx - topPx
```

and the projected distance between the optical reference row and the detected foot position was computed as:

```math
P_c = botPx - midPx
```

The raw height estimate for each image view was subsequently calculated as:

```math
H_i = \frac{H_c \times P_p}{P_c}
```

where:

- `H_c` = camera reference height (70 cm)
- `P_p` = projected body height in pixels
- `P_c` = projected reference distance in pixels

Separate raw height estimates were generated independently from front-view and side-view images. The final ensemble estimate was computed using a weighted average of both view-specific predictions:

```math
H_{ensemble} =
\frac{\sum_i H_i \times P_{c,i}}
{\sum_i P_{c,i}}
```

where `P_{c,i}` denotes the vertical pixel distance between the optical reference row and the detected foot position for each image view. This weighted ensemble strategy improved stability against viewpoint-specific segmentation errors and posture variations.

For nutritional screening analysis, the predicted heights were converted into Length/Height-for-Age (LHFA) z-scores using the WHO 2006 Child Growth Standards.

---

# References

```bibtex
@article{liu2023grounding,
  title={Grounding dino: Marrying dino with grounded pre-training for open-set object detection},
  author={Liu, Shilong and Zeng, Zhaoyang and Ren, Tianhe and others},
  journal={arXiv preprint arXiv:2303.05499},
  year={2023}
}
```

```bibtex
@article{ravi2024sam2,
  title={SAM 2: Segment Anything in Images and Videos},
  author={Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and others},
  journal={arXiv preprint arXiv:2408.00714},
  year={2024}
}
```

```bibtex
@article{lugaresi2019mediapipe,
  title={Mediapipe: A framework for building perception pipelines},
  author={Lugaresi, Camillo and Tang, Jiuqiang and Nash, Hadon and others},
  journal={arXiv preprint arXiv:1906.08172},
  year={2019}
}
```

```bibtex
@article{bazarevsky2020blazepose,
  title={Blazepose: On-device real-time body pose tracking},
  author={Bazarevsky, Valentin and Grishchenko, Ivan and Raveendran, Karthik and others},
  journal={arXiv preprint arXiv:2006.10204},
  year={2020}
}
```
