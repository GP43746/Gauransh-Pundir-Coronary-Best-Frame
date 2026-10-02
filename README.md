# Coronary Angiography Best-Frame Selection

A classical computer-vision pipeline for selecting the most informative frame from a short temporal window of coronary angiography images.

The system assigns a numerical score to every candidate frame and selects exactly one frame as the best frame.

---

## 1. Problem Statement

Given a short temporal window of coronary angiography frames, the objective is to automatically select the single frame that provides the most useful visualization of the coronary vessels.

The system must:

1. Process a variable number of candidate frames.
2. Produce a numerical score for every frame.
3. Rank all candidate frames.
4. Select exactly one best frame.
5. Provide interpretable visualizations showing the scoring behavior.
6. Use only classical computer-vision and statistical techniques.

The assignment specifically discourages treating this as a generic image-quality problem. A frame can have strong global edges or contrast while still providing poor visualization of the coronary vessels. Therefore, the proposed method combines vessel-specific information, general image quality, and temporal information.

---

## 2. Approach Overview

The final pipeline consists of four main stages:

```text
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
Vessel features         Image-quality features
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

$$S_{\text{final}} = 0.40S_V + 0.40S_Q + 0.20S_T$$

where $S_V$ is the vessel-information score, $S_Q$ is the image-quality score, and $S_T$ is the temporal score.

The frame with the highest final score is selected.

---

## 3. Why Multiple Feature Groups?

A single image-quality metric is insufficient for this problem.

For example:
- A sharp frame may not contain clearly visible vessels.
- High global contrast may be caused by non-vessel structures.
- Strong edges may come from borders or artifacts.
- A frame may contain visible vessels but be substantially worse than a neighboring frame during contrast passage.

The final system therefore separates the problem into three complementary branches.

### Vessel branch
Measures whether the extracted vessel structures are spatially concentrated, locally distinguishable from their surroundings, and associated with strong gradients.

### Quality branch
Measures general image information using Tenengrad sharpness and local RMS contrast.

### Temporal branch
Measures whether the current frame represents an improvement relative to neighboring or peak frames in the same candidate window.

This provides an interpretable decomposition rather than treating all measurements as one undifferentiated score.

---

## 4. Preprocessing

### 4.1 Grayscale conversion
Each input image is converted to grayscale. The implementation accepts either an image path or a NumPy image array. All subsequent processing is performed using floating-point grayscale images.

### 4.2 Percentile intensity normalization
Each frame is independently normalized using the 1st and 99th intensity percentiles.

Let $I$ be the grayscale image, and let $p_1 = P_1(I)$ and $p_{99} = P_{99}(I)$.

$$I_n = \operatorname{clip}\left( \frac{I - p_1}{p_{99} - p_1}, 0, 1 \right)$$

This reduces sensitivity to extreme intensity values while retaining the relative intensity structure within each frame. No learned preprocessing or neural representation is used.

---

## 5. Vesselness Extraction

### 5.1 Sato vesselness
The normalized image is processed using the Sato vesselness filter.

The final configuration uses:
```yaml
scales: [1.0, 2.0, 3.0]
black_ridges: true
```

The purpose of vesselness filtering is to emphasize elongated tubular structures corresponding to vessels. The response from the multiple scales is combined by the Sato implementation. The vesselness map is retained for both vessel-mask generation and interpretability visualization.

---

## 6. Vessel Mask Generation

A binary candidate vessel mask is generated from the Sato vesselness response.

### Step 1: Percentile threshold
The vesselness map is thresholded at its 90th percentile:

$$T_V = P_{90}(V)$$

where $V$ is the Sato vesselness map. Pixels satisfying $V(x,y) \geq T_V$ are treated as candidate vessel pixels.

### 6.1 Connected-component filtering
Connected components are extracted using 8-connectivity. Components smaller than 50 pixels are removed. This suppresses small isolated responses that are unlikely to represent meaningful vessel structures.

### 6.2 Morphological closing
A morphological closing operation is applied using a disk-shaped structuring element with radius 1. The purpose is to connect small gaps in the candidate vessel mask without introducing an aggressive morphological transformation.

---

## 7. Vessel Features

Three vessel-specific features are used.

### 7.1 Largest Component Fraction
The largest connected vessel component is measured relative to the total vessel-mask area.

Let $A_L$ be the area of the largest connected component and $A_T$ the total vessel-mask area:

$$F_{LC} = \frac{A_L}{A_T} \quad \text{when } A_T > 0$$

This feature captures whether the detected vessel structure forms a spatially coherent dominant component rather than being distributed primarily across small isolated responses.

### 7.2 Vessel Gradient Energy
Gradient magnitude is computed using Sobel derivatives. For a vessel mask $M$, the feature is the mean squared gradient magnitude inside the detected vessel region:

$$E_G = \frac{1}{|M|} \sum_{(x,y) \in M} G(x,y)^2$$

where $G(x,y)$ is the Sobel gradient magnitude. Higher values indicate stronger local intensity transitions within the detected vessel region.

### 7.3 Local Vessel-Background Contrast
For each detected vessel region, a local surrounding region is constructed through morphological dilation (radius = 3 pixels).

The final feature measures the absolute difference between the mean vessel intensity and the mean intensity of its local surrounding region:

$$C_{VB} = |\mu_V - \mu_B|$$

where $\mu_V$ is the mean intensity in the vessel region and $\mu_B$ is the mean intensity in the surrounding local background. This feature captures local vessel-to-background distinguishability rather than relying solely on global image contrast.

---

## 8. Image-Quality Features

Two general image-quality features are retained.

### 8.1 Tenengrad Sharpness
Tenengrad sharpness is computed from Sobel gradient magnitude over the full image:

$$T = \frac{1}{N} \sum_{x,y} G(x,y)^2$$

where $G$ is the Sobel gradient magnitude. Higher values indicate stronger overall high-frequency structure.

### 8.2 Local RMS Contrast
Local contrast is estimated using the local standard deviation of image intensity. For a local neighborhood $W$ ($15 \times 15$ window):

$$C_{\text{RMS}}(x,y) = \sqrt{\mathbb{E}[I^2] - \mathbb{E}[I]^2}$$

The final feature is the mean local RMS contrast over the image.

---

## 9. Temporal Features

The candidate frames form a short temporal sequence. The temporal branch contains five features:

### 9.1 Sharpness Ratio to Previous Frame
For frame $t$:

$$R_{S,P}(t) = \frac{S(t)}{S(t-1) + \epsilon}$$

where $S(t)$ is the Tenengrad sharpness.

### 9.2 Contrast Ratio to Previous Frame
$$R_{C,P}(t) = \frac{C(t)}{C(t-1) + \epsilon}$$

where $C(t)$ is local RMS contrast.

### 9.3 Sharpness Peak Ratio
$$R_{S,\text{peak}}(t) = \frac{S(t)}{\max_j S(j) + \epsilon}$$

### 9.4 Vessel Gradient Peak Ratio
$$R_{VG,\text{peak}}(t) = \frac{E_G(t)}{\max_j E_G(j) + \epsilon}$$

### 9.5 Vessel Contrast Peak Ratio
$$R_{VC,\text{peak}}(t) = \frac{C_{VB}(t)}{\max_j C_{VB}(j) + \epsilon}$$

---

## 10. Feature Normalization

Each feature is normalized independently within the candidate window using robust percentile min-max normalization. For feature values $x$:

$$p_5 = P_5(x), \quad p_{95} = P_{95}(x)$$

$$x_n = \operatorname{clip}\left( \frac{x - p_5}{p_{95} - p_5}, 0, 1 \right)$$

Higher values correspond to better quality across all normalized features.

---

## 11. Branch Scores

After normalization, each feature branch is computed as the mean of its constituent normalized features.

### 11.1 Vessel Score
$$S_V = \frac{F_{LC,n} + E_{G,n} + C_{VB,n}}{3}$$

### 11.2 Quality Score
$$S_Q = \frac{T_n + C_{\text{RMS},n}}{2}$$

### 11.3 Temporal Score
$$S_T = \operatorname{mean}\left( R_{S,P}, R_{C,P}, R_{S,\text{peak}}, R_{VG,\text{peak}}, R_{VC,\text{peak}} \right)$$

---

## 12. Final Scoring Function

The final score combines the three branches:

$$S_{\text{final}} = 0.40 S_V + 0.40 S_Q + 0.20 S_T$$

The frame with the maximum final score is selected:

$$f^* = \arg\max_f S_{\text{final}}(f)$$

---

## 13. Why These Three Branches?

| Branch | What it measures | Why it is needed |
|---|---|---|
| **Vessel** | Vessel structure and vessel/background distinction | Directly targets the clinical visual content |
| **Quality** | Sharpness and local contrast | Captures general image visibility |
| **Temporal** | Relative improvement and proximity to window peaks | Uses the temporal structure available in the candidate window |

---

## 14. Experimental Feature Selection

Several candidate features and processing approaches were investigated before locking the final pipeline.

### Vessel features investigated
- Largest component fraction *(Retained)*
- Vessel gradient energy *(Retained)*
- Local vessel-background contrast *(Retained)*
- Skeleton continuity *(Discarded - highly redundant with largest component fraction, $\rho \approx 0.99$)*

### Quality features investigated
- Tenengrad *(Retained)*
- Local RMS contrast *(Retained)*
- High-frequency energy, Gradient energy, Laplacian variance *(Discarded - redundant)*

### Artifact-related features
- Blackhat energy *(Discarded - not sufficiently reliable across all dataset cases)*

---

## 15. Temporal Information

Two types of temporal information were found useful:
1. **Relative improvement:** Comparing a frame with the previous frame captures whether image quality is improving or deteriorating.
2. **Relative position within the candidate window:** Comparing a frame against the window-level peak captures whether it is close to the strongest observed response.

---

## 16. Development-Set Evaluation

The development dataset contains **10 cases** and **198 candidate frames**.

The system accepts a variable number of frames per window:

```python
from src.pipeline import select_best_frame

best_frame, scores = select_best_frame(window)
```

---

## 17. Evaluation Metrics

- **Ground-Truth Rank ($r$):** The rank of the annotated best frame after sorting candidates by predicted score (Rank 1 = highest-scoring frame).
- **Top-1 Accuracy:** Proportion of cases where GT rank = 1.
- **Top-3 Accuracy:** Proportion of cases where GT rank $\leq 3$.
- **Top-5 Accuracy:** Proportion of cases where GT rank $\leq 5$.
- **Normalized Rank ($R_{\text{norm}}$):** For $N$ frames and rank $r$:

$$R_{\text{norm}} = 1 - \frac{r - 1}{N - 1}$$

*(1.0 = ranked first, 0.0 = ranked last)*

---

## 18. Branch-Weight Stability

Leave-one-case-out analysis confirmed optimal stability with:
- **Vessel Weight:** `0.40`
- **Quality Weight:** `0.40`
- **Temporal Weight:** `0.20`

---

## 19. Failure Cases

1. **Vessel structure is visible but the model prefers a neighboring frame:** Minor shifts occur when global quality features peak slightly off the best vessel frame.
2. **Strong global quality dominates vessel-specific information:** Non-vessel background structures can occasionally elevate Tenengrad sharpness.
3. **Vesselness response does not perfectly represent true vessels:** Sato vesselness can respond to non-coronary tubular artifacts.
4. **Annotated frame sensitivity:** Consecutive frames are often visually near-identical, making exact single-frame match challenging despite near-identical quality.

---

## 20. Interpretability

For every frame, outputs can be decomposed into:

$$\text{Vessel score} + \text{Quality score} + \text{Temporal score} \longrightarrow \text{Final score}$$

The pipeline saves:
- Normalized images and Sato vesselness maps
- Binary vessel masks
- Raw & normalized feature tables
- Score curves and candidate frame grid plots

---

## 21. Project Structure

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

---

## 22. Installation

Python 3.10+ is recommended.

```bash
python -m venv .venv
```

**Windows PowerShell:**
```powershell
.venv\Scripts\Activate.ps1
```

**Linux/macOS:**
```bash
source .venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 23. Basic Usage

```python
from pathlib import Path
from src.pipeline import select_best_frame

image_dir = Path("path/to/case")
frames = sorted(
    str(p) for p in image_dir.iterdir()
    if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
)

best_frame, scores = select_best_frame(frames)

print("Selected frame:", best_frame)
for item in scores:
    print(item)
```

---

## 24. Command-Line Usage

```bash
python run.py --input path/to/frame_directory --config configs/default.yaml
```

---

## 25. Development-Set Evaluation

```bash
python scripts/evaluate_dev.py
```

Outputs metrics to `results/`.

---

## 26. Visualization Generation

```bash
python scripts/generate_visualizations.py
```

Generates grid plots and score curves in `outputs/dev_visualizations/`.

---

## 27. Configuration

Main parameters in `configs/default.yaml`:

```yaml
vesselness:
  scales: [1.0, 2.0, 3.0]
  black_ridges: true

vessel_mask:
  threshold_percentile: 90.0
  min_component_size: 50
  morphology:
    radius: 1

quality_features:
  local_rms_contrast:
    window_size: 15

normalization:
  lower_percentile: 5.0
  upper_percentile: 95.0

scoring:
  branch_weights:
    vessel: 0.40
    quality: 0.40
    temporal: 0.20
```

---

## 28. Computational Complexity

For $N$ frames, image size $H \times W$, and $K$ Sato scales, dominant computational complexity is:

$$\mathcal{O}(N \cdot K \cdot H \cdot W)$$

The pipeline scales linearly with the number of candidate frames. No GPU or deep-learning inference is required.

---

## 29. Reproducibility

- Deterministic feature definitions
- Fixed percentile bounds
- Parameter-free normalization relative to window
- No learned deep neural networks

---

## 30. Summary

The final best-frame selection evaluates:

$$S_{\text{final}} = 0.40 S_V + 0.40 S_Q + 0.20 S_T$$

It guarantees that chosen frames provide **strong vessel information, adequate global clarity, and favorable temporal context** without relying on deep learning architectures.