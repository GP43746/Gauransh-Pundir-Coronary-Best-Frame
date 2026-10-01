# Coronary Angiography Best-Frame Selection

A classical computer-vision pipeline for selecting the most informative
frame from a short temporal window of coronary angiography images.

The system assigns a numerical score to every candidate frame and
selects exactly one frame as the best frame.

------------------------------------------------------------------------

## 1. Problem Statement

Given a short temporal window of coronary angiography frames, the
objective is to automatically select the single frame that provides the
most useful visualization of the coronary vessels.

The system must:

1.  Process a variable number of candidate frames.
2.  Produce a numerical score for every frame.
3.  Rank all candidate frames.
4.  Select exactly one best frame.
5.  Provide interpretable visualizations showing the scoring behavior.
6.  Use only classical computer-vision and statistical techniques.

The assignment specifically discourages treating this as a generic
image-quality problem. A frame can have strong global edges or contrast
while still providing poor visualization of the coronary vessels.
Therefore, the proposed method combines vessel-specific information,
general image quality, and temporal information.

------------------------------------------------------------------------

## 2. Approach Overview

The final pipeline consists of four main stages:

``` text
Input angiography frames
        │
        ▼
Percentile intensity normalization
        │
        ▼
Sato vesselness estimation
        │
        ▼
Vessel mask generation
        │
        ├───────────────────────┐
        ▼                       ▼
Vessel features          Image-quality features
        │                       │
        └───────────┬───────────┘
                    ▼
             Temporal features
                    │
                    ▼
          Robust feature normalization
                    │
                    ▼
          Three scoring branches
                    │
                    ▼
       Weighted final frame score
                    │
                    ▼
             Frame ranking
                    │
                    ▼
             Best frame
```

The final score is:

\[ S\_{`\text{final}`{=tex}} = 0.40S_V + 0.40S_Q + 0.20S_T \]

where (S_V) is the vessel-information score, (S_Q) is the image-quality
score, and (S_T) is the temporal score.

The frame with the highest final score is selected.

------------------------------------------------------------------------

## 3. Why Multiple Feature Groups?

A single image-quality metric is insufficient for this problem.

For example:

-   A sharp frame may not contain clearly visible vessels.
-   High global contrast may be caused by non-vessel structures.
-   Strong edges may come from borders or artifacts.
-   A frame may contain visible vessels but be substantially worse than
    a neighboring frame during contrast passage.

The final system therefore separates the problem into three
complementary branches.

### Vessel branch

Measures whether the extracted vessel structures are spatially
concentrated, locally distinguishable from their surroundings, and
associated with strong gradients.

### Quality branch

Measures general image information using Tenengrad sharpness and local
RMS contrast.

### Temporal branch

Measures whether the current frame represents an improvement relative to
neighboring or peak frames in the same candidate window.

This provides an interpretable decomposition rather than treating all
measurements as one undifferentiated score.

------------------------------------------------------------------------

# 4. Preprocessing

## 4.1 Grayscale conversion

Each input image is converted to grayscale.

The implementation accepts either an image path or a NumPy image array.

All subsequent processing is performed using floating-point grayscale
images.

------------------------------------------------------------------------

## 4.2 Percentile intensity normalization

Each frame is independently normalized using the 1st and 99th intensity
percentiles.

Let (I) be the grayscale image and let (p_1=P_1(I)) and
(p\_{99}=P\_{99}(I)).

\[ I_n = `\operatorname{clip}`{=tex} `\left`{=tex}(
`\frac{I-p_1}{p_{99}-p_1}`{=tex}, 0,1 `\right`{=tex}) \]

This reduces sensitivity to extreme intensity values while retaining the
relative intensity structure within each frame.

No learned preprocessing or neural representation is used.

------------------------------------------------------------------------

# 5. Vesselness Extraction

## 5.1 Sato vesselness

The normalized image is processed using the Sato vesselness filter.

The final configuration uses:

``` text
Scales:       [1, 2, 3]
black_ridges: True
```

The purpose of vesselness filtering is to emphasize elongated tubular
structures corresponding to vessels.

The response from the multiple scales is combined by the Sato
implementation.

The vesselness map is retained for both vessel-mask generation and
interpretability visualization.

------------------------------------------------------------------------

# 6. Vessel Mask Generation

A binary candidate vessel mask is generated from the Sato vesselness
response.

### Step 1: Percentile threshold

The vesselness map is thresholded at its 90th percentile:

\[ T_V=P\_{90}(V) \]

where (V) is the Sato vesselness map.

Pixels satisfying (V(x,y)`\geq `{=tex}T_V) are treated as candidate
vessel pixels.

## 6.1 Connected-component filtering

Connected components are extracted using 8-connectivity.

Components smaller than 50 pixels are removed.

This suppresses small isolated responses that are unlikely to represent
meaningful vessel structures.

## 6.2 Morphological closing

A morphological closing operation is applied using a disk-shaped
structuring element with radius 1.

The purpose is to connect small gaps in the candidate vessel mask
without introducing an aggressive morphological transformation.

The resulting mask is used for vessel-specific feature extraction.

------------------------------------------------------------------------

# 7. Vessel Features

Three vessel-specific features are used.

## 7.1 Largest Component Fraction

The largest connected vessel component is measured relative to the total
vessel-mask area.

Let (A_L) be the area of the largest connected component and (A_T) the
total vessel-mask area.

\[ F\_{LC}=`\frac{A_L}{A_T}`{=tex} \]

when (A_T\>0).

This feature captures whether the detected vessel structure forms a
spatially coherent dominant component rather than being distributed
primarily across small isolated responses.

## 7.2 Vessel Gradient Energy

Gradient magnitude is computed using Sobel derivatives.

For a vessel mask (M), the feature is the mean squared gradient
magnitude inside the detected vessel region:

\[ E_G = `\frac{1}{|M|}`{=tex}
`\sum`{=tex}\_{(x,y)`\in `{=tex}M}G(x,y)\^2 \]

where (G(x,y)) is the Sobel gradient magnitude.

Higher values indicate stronger local intensity transitions within the
detected vessel region.

## 7.3 Local Vessel-Background Contrast

For each detected vessel region, a local surrounding region is
constructed through morphological dilation.

The final feature measures the absolute difference between the mean
vessel intensity and the mean intensity of its local surrounding region:

\[ C\_{VB}=\|`\mu`{=tex}\_V-`\mu`{=tex}\_B\| \]

where (`\mu`{=tex}\_V) is the mean intensity in the vessel region and
(`\mu`{=tex}\_B) is the mean intensity in the surrounding local
background.

The dilation radius is 3 pixels.

This feature is intended to capture local vessel-to-background
distinguishability rather than relying solely on global image contrast.

------------------------------------------------------------------------

# 8. Image-Quality Features

Two general image-quality features are retained.

## 8.1 Tenengrad Sharpness

Tenengrad sharpness is computed from Sobel gradient magnitude over the
full image:

\[ T = `\frac{1}{N}`{=tex} `\sum`{=tex}\_{x,y}G(x,y)\^2 \]

where (G) is the Sobel gradient magnitude.

Higher values indicate stronger overall high-frequency structure.

Tenengrad is useful as a sharpness measure, but it is not used alone
because non-vessel edges can also increase its value.

## 8.2 Local RMS Contrast

Local contrast is estimated using the local standard deviation of image
intensity.

For a local neighborhood (W):

\[ C\_{RMS}(x,y)= `\sqrt{E[I^2]-E[I]^2}`{=tex} \]

The final feature is the mean local RMS contrast over the image.

The implementation uses a 15 × 15 window.

This captures local intensity variation rather than relying only on
global intensity statistics.

------------------------------------------------------------------------

# 9. Temporal Features

The candidate frames form a short temporal sequence. Therefore,
information about the position of a frame within the sequence can
provide additional information.

The temporal branch contains five features.

## 9.1 Sharpness Ratio to Previous Frame

For frame (t):

\[ R\_{S,P}(t)=`\frac{S(t)}{S(t-1)+\epsilon}`{=tex} \]

where (S(t)) is the Tenengrad sharpness.

A value above 1 indicates an increase in sharpness relative to the
previous frame.

The first frame has no previous frame and therefore receives a missing
value for this feature.

## 9.2 Contrast Ratio to Previous Frame

\[ R\_{C,P}(t)=`\frac{C(t)}{C(t-1)+\epsilon}`{=tex} \]

where (C(t)) is local RMS contrast.

This captures frame-to-frame improvement in local contrast.

## 9.3 Sharpness Peak Ratio

\[ R\_{S,`\text{peak}`{=tex}}(t)=
`\frac{S(t)}{\max_jS(j)+\epsilon}`{=tex} \]

This gives a measure of how close the frame is to the sharpness peak of
the current window.

## 9.4 Vessel Gradient Peak Ratio

\[ R\_{VG,`\text{peak}`{=tex}}(t)=
`\frac{E_G(t)}{\max_jE_G(j)+\epsilon}`{=tex} \]

This provides a temporal measure of whether the current frame approaches
the strongest vessel-gradient response in the candidate window.

## 9.5 Vessel Contrast Peak Ratio

\[ R\_{VC,`\text{peak}`{=tex}}(t)=
`\frac{C_{VB}(t)}{\max_jC_{VB}(j)+\epsilon}`{=tex} \]

This captures how close each frame is to the strongest local
vessel-background contrast observed in the window.

------------------------------------------------------------------------

# 10. Feature Normalization

The raw features have different numerical scales. Directly adding raw
features would therefore make the scoring dependent on arbitrary feature
units.

Each feature is normalized independently within the candidate window
using robust percentile min-max normalization.

For feature values (x):

\[ p_5=P_5(x) \]

\[ p\_{95}=P\_{95}(x) \]

and:

\[ x_n= `\operatorname{clip}`{=tex} `\left`{=tex}(
`\frac{x-p_5}{p_{95}-p_5}`{=tex}, 0,1 `\right`{=tex}) \]

The normalization is performed separately for each candidate window.

The direction of all retained features is:

``` text
Higher value = better
```

This makes the normalized features comparable while reducing the
influence of extreme observations.

For temporal features where the first frame has no previous-frame
measurement, the missing value is ignored when calculating the temporal
branch mean.

---

# 11. Branch Scores

After normalization, each feature branch is computed as the mean of its constituent normalized features.

## 11.1 Vessel Score

The vessel branch contains:

```text
largest_component_fraction
vessel_gradient_energy
local_vessel_background_contrast
```

Therefore:

\[
S_V =
\frac{
F_{LC,n}+E_{G,n}+C_{VB,n}
}{3}
\]

## 11.2 Quality Score

The quality branch contains:

```text
tenengrad
local_rms_contrast
```

Therefore:

\[
S_Q =
\frac{
T_n+C_{RMS,n}
}{2}
\]

## 11.3 Temporal Score

The temporal branch contains:

```text
sharpness_ratio_prev
contrast_ratio_prev
sharpness_peak_ratio
vessel_gradient_peak_ratio
vessel_contrast_peak_ratio
```

Therefore:

\[
S_T =
\operatorname{mean}
\left(
R_{S,P},
R_{C,P},
R_{S,\text{peak}},
R_{VG,\text{peak}},
R_{VC,\text{peak}}
\right)
\]

using the normalized versions and ignoring unavailable values.

---

# 12. Final Scoring Function

The final score combines the three branches:

\[
\boxed{
S_{\text{final}}
=
0.40S_V
+
0.40S_Q
+
0.20S_T
}
\]

The weights were selected from a predeclared branch-weight stability experiment rather than tuning the final score separately for each case.

The final interpretation is:

```text
40%  Vessel information
40%  Image quality
20%  Temporal context
```

The frame with the maximum final score is selected:

\[
f^*=\arg\max_f S_{\text{final}}(f)
\]

---

# 13. Why These Three Branches?

The three branches intentionally capture different aspects of frame quality.

| Branch | What it measures | Why it is needed |
|---|---|---|
| Vessel | Vessel structure and vessel/background distinction | Directly targets the clinical visual content |
| Quality | Sharpness and local contrast | Captures general image visibility |
| Temporal | Relative improvement and proximity to window peaks | Uses the temporal structure available in the candidate window |

This prevents the system from depending entirely on one type of image statistic.

---

# 14. Experimental Feature Selection

Several candidate features and processing approaches were investigated before locking the final pipeline.

The final feature set was selected based on:

1. Complementarity with other features.
2. Behavior across development cases.
3. Redundancy between features.
4. Stability of combined scoring.
5. Interpretability.
6. Avoiding unnecessary complexity.

### Vessel features investigated

Candidate vessel-related measurements included:

- largest component fraction
- vessel gradient energy
- local vessel-background contrast
- skeleton continuity

Skeleton continuity was not retained because it was highly redundant with largest-component fraction. Their correlation was approximately:

\[
\rho \approx 0.99
\]

Adding the redundant feature did not improve the combined scoring behavior.

### Quality features investigated

Candidate quality measurements included:

- Tenengrad
- local RMS contrast
- high-frequency energy
- gradient energy
- Laplacian variance

Tenengrad and local RMS contrast were retained because they provided complementary sharpness and local-contrast information.

### Artifact-related features

Blackhat-based measurements were investigated as potential artifact indicators.

Although blackhat energy showed some useful ranking behavior, it was not sufficiently established as a reliable representation of artifact burden in this dataset.

It was therefore not included in the final score.

This keeps the final model interpretable and avoids introducing a feature merely because it correlates with a subset of outcomes.

---

# 15. Temporal Information

Temporal information was explicitly investigated rather than treating every frame as an independent image.

Two types of temporal information were found useful:

### Relative improvement

Comparing a frame with the previous frame captures whether image quality is improving or deteriorating.

### Relative position within the candidate window

Comparing a frame against the window-level peak captures whether it is close to the strongest observed response.

Simple temporal similarity was not used as a final feature because similarity to neighboring frames alone does not necessarily indicate that a frame is diagnostically informative.

The final temporal branch therefore focuses on relative quality and vessel measurements.

---

# 16. Development-Set Evaluation

The development dataset contains:

```text
10 cases
198 candidate frames
```

The system does not assume that every window contains exactly 10 frames.

The implementation accepts a variable number of frames:

```python
from src.pipeline import select_best_frame

best_frame, scores = select_best_frame(window)
```

Each case is processed independently.

For every case the evaluation records:

- number of candidate frames
- ground-truth frame
- predicted frame
- ground-truth rank
- normalized rank
- Top-1 result
- Top-3 result
- Top-5 result

---

# 17. Evaluation Metrics

## 17.1 Ground-Truth Rank

The rank of the annotated best frame after sorting candidates by predicted score.

```text
Rank 1 = highest-scoring frame
Rank 2 = second-highest
...
```

Lower rank is better.

## 17.2 Top-1 Accuracy

The proportion of cases where the predicted frame exactly matches the ground-truth frame.

\[
Top1 =
\frac{
\#\{\text{cases where GT rank}=1\}
}{
N
}
\]

## 17.3 Top-3 Accuracy

The proportion of cases where the ground-truth frame is among the three highest-ranked frames.

\[
Top3 =
\frac{
\#\{\text{cases where GT rank}\leq3\}
}{
N
}
\]

## 17.4 Top-5 Accuracy

The proportion of cases where the ground-truth frame is among the five highest-ranked frames.

\[
Top5 =
\frac{
\#\{\text{cases where GT rank}\leq5\}
}{
N
}
\]

## 17.5 Normalized Rank

Because different windows can contain different numbers of frames, rank is additionally normalized.

For a window containing \(N\) frames and ground-truth rank \(r\):

\[
R_{\text{norm}}
=
1-
\frac{r-1}{N-1}
\]

This produces:

```text
1.0 → ground truth ranked first
0.0 → ground truth ranked last
```

---

# 18. Branch-Weight Stability

The final branch weights were examined using a leave-one-case-out analysis.

The candidate branch-weight configurations were evaluated without continuously tuning the weights to individual cases.

The selected configuration was:

```text
Vessel   = 0.40
Quality  = 0.40
Temporal = 0.20
```

This configuration produced the strongest overall stability among the tested predefined combinations.

Because the development set contains only 10 cases, this analysis should be interpreted as a stability check rather than as statistically conclusive evidence of optimal weights.

---

# 19. Failure Cases

The development cases demonstrate several important failure modes.

## Failure Case 1: Vessel structure is visible but the model prefers a neighboring frame

A frame may have strong vessel responses but another nearby frame can obtain a better combination of global quality and temporal measurements.

This can cause the selected frame to shift by a few frames from the annotated frame.

### Possible improvement

Introduce more explicit spatial vessel-quality measurements, such as:

- regional vessel visibility,
- vessel-background contrast over anatomically meaningful regions,
- branch-level continuity,
- temporal persistence of vessel structures.

## Failure Case 2: Strong global quality dominates vessel-specific information

A frame can have high sharpness or local contrast because of structures unrelated to the coronary vessels.

Since Tenengrad and local RMS contrast operate over the full image, they can sometimes favor such frames.

### Possible improvement

Use a more conservative field-of-view or anatomy-aware region proposal before computing quality features.

Another option would be to calculate quality statistics separately inside vessel candidate regions and non-vessel regions.

## Failure Case 3: Vesselness response does not perfectly represent true vessels

Sato vesselness can respond to structures that resemble vessels but are not clinically relevant coronary structures.

Consequently, a high vesselness-derived score does not guarantee that the detected structure corresponds to the desired coronary anatomy.

### Possible improvement

Use additional classical constraints such as:

- orientation consistency,
- vessel width consistency,
- branch continuity,
- local tubularness,
- connected-path analysis,
- spatial priors.

## Failure Case 4: The annotated frame may be close to several visually similar frames

Coronary angiography often contains sequences where several consecutive frames provide nearly equivalent visualization.

A deterministic algorithm may select a neighboring frame even though the evaluation protocol identifies another frame.

### Possible improvement

Evaluate temporal neighborhoods or allow an application-specific acceptable-frame tolerance when the evaluation protocol permits it.

The current implementation still performs the required exact single-frame selection.

---

# 20. Interpretability

The scoring system is intentionally decomposed into interpretable quantities.

For every frame:

```text
Vessel score
      +
Quality score
      +
Temporal score
      ↓
Final score
```

The pipeline also retains:

- normalized image
- Sato vesselness map
- vessel mask
- individual feature values
- normalized feature values
- branch scores
- final score

This makes it possible to inspect why a particular frame received its score.

The visualization module provides:

1. candidate-frame grids with scores,
2. score curves across the temporal window,
3. vesselness and vessel-mask visualizations.

---

# 21. Project Structure

```text
coronary-best-frame/
│
├── README.md
├── requirements.txt
├── run.py
│
├── configs/
│   └── default.yaml
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── vesselness.py
│   ├── vessel_mask.py
│   ├── vessel_features.py
│   ├── quality_features.py
│   ├── temporal_features.py
│   ├── normalization.py
│   ├── scoring.py
│   ├── pipeline.py
│   └── visualization.py
│
├── scripts/
│   ├── evaluate_dev.py
│   └── generate_visualizations.py
│
├── examples/
│   └── example_run.py
│
├── outputs/
│   └── dev_visualizations/
│
├── results/
│   ├── dev_predictions.csv
│   ├── dev_metrics.csv
│   └── case_*_scores.csv
│
└── report/
    └── technical_report.pdf
```

The dataset itself is intentionally kept outside the source repository.

---

# 22. Installation

Python 3.10+ is recommended.

Create and activate a virtual environment:

```bash
python -m venv .venv
```

### Windows

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Required packages:

```text
numpy
scipy
opencv-python
scikit-image
matplotlib
pandas
PyYAML
```

No deep-learning framework is required.

---

# 23. Basic Usage

The main interface is:

```python
from src.pipeline import select_best_frame

best_frame, scores = select_best_frame(window)
```

where `window` is a list of image paths or image arrays.

Example:

```python
from pathlib import Path

from src.pipeline import select_best_frame

image_dir = Path("path/to/case")

frames = sorted(
    str(p)
    for p in image_dir.iterdir()
    if p.suffix.lower() in {
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".tif",
        ".tiff",
    }
)

best_frame, scores = select_best_frame(frames)

print("Selected frame:", best_frame)

for item in scores:
    print(item)
```

The implementation does not assume a fixed number of frames.

---

# 24. Command-Line Usage

The provided `run.py` can process a directory of candidate frames.

Example:

```bash
python run.py --input path/to/frame_directory
```

An alternative configuration can be supplied with:

```bash
python run.py \
    --input path/to/frame_directory \
    --config configs/default.yaml
```

The default configuration is automatically loaded when `--config` is omitted.

---

# 25. Development-Set Evaluation

The complete development evaluation can be executed using:

```bash
python scripts/evaluate_dev.py
```

The evaluation script:

1. Loads the development metadata.
2. Finds all development cases.
3. Loads all candidate frames.
4. Runs the locked pipeline.
5. Selects the highest-scoring frame.
6. Computes the ground-truth rank.
7. Computes Top-1, Top-3 and Top-5 results.
8. Saves per-case scores.
9. Saves aggregate metrics.

Outputs are written to:

```text
results/
```

---

# 26. Visualization Generation

Run:

```bash
python scripts/generate_visualizations.py
```

The resulting visualizations are stored under:

```text
outputs/dev_visualizations/
```

Each case contains:

```text
frame_scores.png
score_curve.png
```

The frame-score visualization displays the candidate frames and their corresponding final scores.

The score curve displays how the final score changes across the candidate temporal window.

---

# 27. Configuration

The main configuration is stored in:

```text
configs/default.yaml
```

Important parameters include:

```yaml
vesselness:
  scales:
    - 1.0
    - 2.0
    - 3.0
  black_ridges: true
```

```yaml
vessel_mask:
  threshold_percentile: 90.0
  min_component_size: 50
  morphology:
    radius: 1
```

```yaml
quality_features:
  local_rms_contrast:
    window_size: 15
```

```yaml
normalization:
  lower_percentile: 5.0
  upper_percentile: 95.0
```

```yaml
scoring:
  branch_weights:
    vessel: 0.40
    quality: 0.40
    temporal: 0.20
```

The configuration separates algorithmic parameters from the implementation code.

---

# 28. Computational Complexity

Let:

- \(N\) = number of candidate frames
- \(H,W\) = image height and width
- \(K\) = number of Sato scales

The dominant operations are performed independently for each frame.

Approximate processing complexity is therefore proportional to:

\[
O(N \cdot K \cdot H \cdot W)
\]

for the vesselness computation, plus lower-order costs for:

- gradient computation,
- local statistics,
- connected components,
- morphology,
- feature normalization,
- ranking.

The pipeline therefore scales approximately linearly with the number of candidate frames for a fixed image size and number of vesselness scales.

No model training or GPU inference is required.

---

# 29. Reproducibility

The implementation is designed to be reproducible through:

- explicit configuration,
- deterministic feature definitions,
- fixed preprocessing parameters,
- fixed vesselness scales,
- fixed mask parameters,
- fixed feature normalization,
- fixed branch weights,
- modular source files.

The final algorithm does not learn parameters from the test image beyond the explicitly defined candidate-window feature normalization and temporal peak calculations.

---

# 30. Design Principles

The final system follows five main principles.

### 1. Vessel relevance

The algorithm should reward visibility of the target vessel structures rather than simply rewarding arbitrary image edges.

### 2. Complementary evidence

No individual feature is treated as sufficient.

### 3. Robust normalization

Features are normalized before combination so that numerical scale does not determine their contribution.

### 4. Temporal context

Neighboring frames and window-level peaks provide useful information unavailable from isolated-frame analysis.

### 5. Interpretability

Every final score can be decomposed into vessel, quality, and temporal contributions.

---

# 31. Limitations

The current system has several limitations.

### Limited development-set size

The development dataset contains only 10 cases. Consequently, the evaluation results should not be interpreted as a statistically definitive estimate of generalization performance.

### Hand-designed vessel representation

Sato vesselness is a classical tubular-structure detector but is not guaranteed to perfectly distinguish coronary vessels from all other structures.

### No explicit anatomy model

The pipeline does not contain an anatomical model of the coronary tree.

### No learned adaptation

The method intentionally avoids CNNs, transformers, learned embeddings, and other deep-learning approaches because the assignment requires a classical computer-vision solution.

### Exact-frame annotation sensitivity

When several consecutive frames are visually similar, selecting a neighboring frame can produce a low exact-match score even when the visual difference is small.

---

# 32. Potential Future Improvements

Several extensions could improve the system while preserving the classical-CV constraint.

## Spatially localized scoring

Divide the field of view into spatial regions and calculate vessel visibility separately.

## Vessel orientation consistency

Use structure tensors or Hessian-derived orientation information to measure whether detected vessel structures form coherent directional patterns.

## Vessel-path analysis

Construct graph representations of vessel candidates and evaluate path continuity.

## Multi-scale vessel analysis

Evaluate vessel visibility separately across multiple spatial scales rather than aggregating all vesselness responses immediately.

## Temporal persistence

Track vessel structures across consecutive frames using classical optical flow or feature correspondence.

## Candidate-region quality

Calculate sharpness and contrast inside vessel candidate regions rather than relying primarily on full-image measurements.

## Robust temporal ranking

Instead of using only immediate previous-frame ratios, model short temporal neighborhoods around each candidate frame.

---

# 33. Rejected Approaches

Several approaches were considered but not included in the final pipeline.

### Single sharpness metric

Rejected because sharpness alone does not guarantee good vessel visualization.

### Global contrast alone

Rejected because high global contrast can originate from irrelevant structures.

### Skeleton continuity

Not retained because it was highly redundant with largest-component fraction and did not improve the combined model sufficiently.

### Additional artifact penalty

Not included because the investigated artifact measurement did not provide sufficiently reliable evidence that it represented actual artifact burden across the dataset.

### More aggressive preprocessing

More complex preprocessing pipelines were not retained because the percentile-normalized input provided a simpler and more stable basis for the final pipeline.

### Deep learning

Not used because the assignment explicitly restricts the solution to classical computer vision.

---

# 34. Summary

The final system treats best-frame selection as a **multi-factor ranking problem**, rather than as simple image-quality maximization.

The selected frame is determined from:

\[
\boxed{
S_{\text{final}}
=
0.40S_V
+
0.40S_Q
+
0.20S_T
}
\]

where:

\[
S_V =
\operatorname{mean}
(
\text{largest component fraction},
\text{vessel gradient energy},
\text{local vessel-background contrast}
)
\]

\[
S_Q =
\operatorname{mean}
(
\text{Tenengrad},
\text{local RMS contrast}
)
\]

and:

\[
S_T =
\operatorname{mean}
(
\text{previous-frame sharpness ratio},
\text{previous-frame contrast ratio},
\text{sharpness peak ratio},
\text{vessel-gradient peak ratio},
\text{vessel-contrast peak ratio}
)
\]

after robust candidate-window normalization.

The resulting system is:

- Classical computer vision only
- Modular
- Configurable
- Interpretable
- Variable-window compatible
- Reproducible
- Explicitly vessel-aware
- Temporally aware
- Free of learned image representations

The central design choice is to treat a good angiography frame as one that provides **strong and coherent vessel information, sufficient local image quality, and favorable temporal context**, rather than assuming that the visually sharpest frame is automatically the best frame.
