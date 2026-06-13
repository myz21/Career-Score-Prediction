# Career Score Prediction

## BTK Datathon 2026 — Happy Llama Team

> Predicting student career success scores using a hybrid CatBoost + TabPFN ensemble with role-aware feature engineering.

---

## Problem Statement

Given a dataset of student profiles — including technical skill scores, portfolio quality, interview performance, and target role — predict the `career_success_score` (0–100) for each student.

The challenge: **the same skill level means different things in different roles.** A high `backend_score` is much more valuable for a Backend Developer than for a Data Scientist. This insight drives our core feature engineering strategy.

---

## Solution Architecture (may be change)

```
Train Data (CSV)
      │
      ▼
┌─────────────────────────────────────────────┐
│           Preprocessing Pipeline            │
│  • Missing value imputation                 │
│  • TF-IDF on mentor_feedback_text (500 dim) │
│  • SVD reduction (50 components)            │
│  • Role-aware feature engineering           │
└─────────────────────────────────────────────┘
      │
      ├──────────────────┬──────────────────┐
      ▼                  ▼                  ▼
┌──────────┐      ┌──────────┐       ┌──────────┐
│ CatBoost │      │  TabPFN  │       │  OOF     │
│ (w/ TF-  │      │  v3 (SVD │       │  Blend   │
│  IDF)    │      │  only)   │       │  Weights │
└──────────┘      └──────────┘       └──────────┘
      │                  │
      └──────────┬───────┘
                 ▼
        Weighted Blend (0.45 / 0.55)
                 │
                 ▼
        Final Predictions (clipped 0–100)
```

---

## Key Feature Engineering: Role-Aware Skills (will be added)

The most important innovation in this solution is **role-relevant skill features**. Instead of treating all skill scores equally, we compute skill metrics specific to each student's `target_role`.

```python
role_skill_map = {
    "backend":   ["backend_score", "sql_score", "cloud_score", "devops_score", "problem_solving_score"],
    "frontend":  ["frontend_score", "portfolio_score", "communication_score", "coding_score"],
    "data":      ["machine_learning_score", "sql_score", "data_structures_score", "problem_solving_score"],
    "ml":        ["machine_learning_score", "data_structures_score", "problem_solving_score", "coding_score"],
    "devops":    ["devops_score", "cloud_score", "backend_score", "problem_solving_score"],
    "fullstack": ["frontend_score", "backend_score", "sql_score", "cloud_score", "portfolio_score"],
    "mobile":    ["frontend_score", "backend_score", "coding_score", "portfolio_score"],
}
```

**Derived features:**
| Feature | Description |
|---|---|
| `role_relevant_skill_mean` | Mean of skills relevant to student's target role |
| `role_relevant_skill_max` | Max skill score within target role |
| `role_relevant_skill_min` | Min skill score within target role |
| `role_skill_lift` | `role_relevant_skill_mean` − `overall_skill_mean` |
| `role_skill_x_project_quality` | Interaction: role fit × project quality |
| `role_skill_x_technical_interview` | Interaction: role fit × interview score |
| `role_skill_x_portfolio` | Interaction: role fit × portfolio score |


---

## Getting Started

### Installation

**1 — Clone the repo**
```bash
git clone https://github.com/myz21/Career-Score-Prediction.git
cd Career-Score-Prediction
```

**2 — Install core dependencies**
```bash
pip install -e .
```

**3 — Install PyTorch** *(pick one)*

| Environment | Command |
|---|---|
| 🖥️ GPU with CUDA 12.8 (`teknofest_yarisma` env) | `pip install -r requirements-gpu.txt` |
| 💻 CPU-only (Kaggle kernel / no GPU) | `pip install -r requirements-cpu.txt` |

> **Why separate files?** PyTorch CUDA builds need `--index-url https://download.pytorch.org/whl/cu128`, which cannot be expressed inside `pyproject.toml`.

**Optional — dev tools (JupyterLab, widgets)**
```bash
pip install -e ".[dev]"
```

### Running the Notebook

```bash
jupyter notebook son_kaggle.ipynb
```

The notebook expects `train.csv` and `test_x.csv` in the working directory *(not included due to size — download from the BTK Datathon 2026 competition page)*.

---

## Model Details

### CatBoost
- **Iterations:** 3000–4000  
- **Learning rate:** 0.02–0.025  
- **Depth:** 4–5  
- **L2 regularization:** 8–10  
- **Features:** Structured + TF-IDF text features (500 dims)  
- **Categorical features:** Auto-detected (target role, university tier, department, etc.)

### TabPFN v3
- **Input:** Structured features + SVD-reduced text (50 dims); TF-IDF columns dropped  
- **Mode:** Low memory, batch prediction  
- **Inference:** Per-fold, averaged across 5 folds

### Ensemble
- **Cross-validation:** `StratifiedKFold(n_splits=5)` with target-bin stratification
- **Blend:** `0.45 × CatBoost + 0.55 × TabPFN` (OOF-optimized)
- **Post-processing:** `np.clip(predictions, 0, 100)`

---

## Validation Strategy

**Problem:** Standard `KFold` was too optimistic — public leaderboard showed a gap.

**Fix:** Switched to **stratified KFold on target quantile bins**:

```python
from sklearn.model_selection import StratifiedKFold
import pandas as pd

y_bins = pd.qcut(y, q=10, labels=False, duplicates="drop")
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_bins)):
    ...
```

This ensures each fold has a representative distribution of career success scores.

---

## 🔍 Key Insights (from SHAP Analysis)

1. **`project_quality_score`** — strongest single predictor
2. **`technical_interview_score`** — second most important
3. **`role_relevant_skill_mean`** — role-aware skill aggregate (engineered)
4. **`portfolio_score`** — especially important for frontend/fullstack roles
5. **Text features** (TF-IDF on mentor feedback) add noise at high dimensions; SVD reduction helps TabPFN

---

## Strategy Notes

See [`TRAINING_STRATEGY_PROMPT.md`](TRAINING_STRATEGY_PROMPT.md) for detailed training strategy and CV analysis.

See [`UZMANLIK_ALANI_PROMPT.md`](UZMANLIK_ALANI_PROMPT.md) for the complete role-aware feature engineering guide.

---

## 👥 Team

**Happy Llama** — Datathon 2026  
GitHub: [@myz21](https://github.com/myz21)
Github: [@ebrarkuz](https://github.com/ebrarkuz)
---

## License

This project is licensed under the MIT License.
