# Photograph-Based Pediatric Height Estimation

Monocular RGB image-based pediatric standing height estimation and LHFA nutritional risk screening using computer vision and geometric modeling techniques.

---


# Methodology

The proposed study developed a photograph-based height estimation framework using monocular RGB images and geometric modeling techniques to estimate pediatric standing height. For each participant, front-view and side-view photographs were captured using a fixed camera setup. A post-it marker was pasted such that its upper edge was positioned exactly 70 cm above the ground level, thereby serving as the optical reference point for geometric calculations.

Child body regions were extracted using a segmentation framework combining Grounding DINO and SAM. The segmentation pipeline identified the child body mask and the post-it reference marker within the image. Multiple candidate detections were ranked according to object size and proximity to the image center, and the most appropriate segmentation mask was selected for subsequent processing.

To improve robustness against posture variations and raised-arm artifacts, MediaPipe Pose and BlazePose were employed to identify facial landmarks including the nose and ears. The horizontal head center was estimated from these landmarks, and the top of the head (`topPx`) was determined by vertically scanning a narrow band around the head center within the segmentation mask. The lower body boundary (`botPx`) was obtained from the lower extent of the child mask using the 98th percentile of foreground rows to minimize segmentation noise and outlier effects.

The upper edge of the detected post-it marker was used to estimate the optical reference row (`midPx`). When the post-it marker could not be detected, the midpoint was approximated using a fixed fraction of the image height:

```math
midPx = 0.484 \times H_{image}
```

Height estimation was performed using a pinhole camera geometry model. The raw height estimate for each image view was calculated as:

```math
H_i = \frac{H_c \times P_p}{P_c}
```

where:

- `H_c` = camera reference height (70 cm)
- `P_p = botPx - topPx`
- `P_c = botPx - midPx`

Separate raw height estimates were generated independently from front-view and side-view images. The final ensemble estimate was computed using a weighted average of both view-specific predictions:

```math
H_{ensemble} =
\frac{\sum_i H_i \times P_{c,i}}
{\sum_i P_{c,i}}
```

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
