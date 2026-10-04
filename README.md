# HeightEstimation

Code for the JMIR AI research letter *Child Stunting Screening: Estimating Height From
Smartphone Photographs*. It estimates a child's standing height from a front and a side
photograph and classifies stunting with the WHO Child Growth Standards.

## Method

The camera lens and the upper edge of a paper marker on the wall are both 70 cm above
the floor, so the marker row is the horizon in the photograph. Height then follows from

```
H = camH × (botPx − topPx) / (botPx − midPx)
```

where `topPx`, `botPx`, and `midPx` are the rows of the top of the head, the feet, and
the horizon, and `camH` is 70 cm (64 cm when the child stands on the 6 cm weighing
scale). Grounding DINO and SAM 2.1 segment the child, marker, and scale; MediaPipe Pose
locates the head. The front-view and side-view estimates are averaged with equal weight.

## Usage

Requires Python 3.13. A GPU is used when available.

```bash
pip install -r requirements.txt
python Height.py
```

Place the inputs next to `Height.py`:

- `Data/<uuid>_front.jpg` and `Data/<uuid>_side.jpg`: one photograph pair per child
- `Data.csv`: columns `uuid`, `gender` (`Male` or `Female`), `age_days`,
  `height_cm_self_reported` (stadiometer height), `timestamp`, and `lat`
- `who_tables/`: WHO length/height-for-age LMS tables (included)
- `pose_landmarker_heavy.task`: MediaPipe pose model (included; Apache 2.0, Google)

The script writes `results.csv` (`id`, `age_days`, `gender`, `height_cm`,
`predicted_height_cm`) and prints the agreement and classification statistics reported
in the paper, with bootstrap 95% CIs.

## Data

The study photographs and measurements are confidential and are not included.
