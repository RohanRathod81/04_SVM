# Support Vector Machines — SVC & SVR

Practice project exploring both sides of Support Vector Machines: classification with **SVC** and regression with **SVR**, using two classic scikit-learn built-in datasets.

## 📌 Overview

This project implements and compares:
- **SVC (Support Vector Classifier)** on the **Iris** dataset — a 3-class flower classification problem.
- **SVR (Support Vector Regressor)** on the **Diabetes** dataset — predicting a continuous disease progression score.

The goal was to build intuition for how the margin-based logic behind SVMs adapts across classification and regression, and how kernel choice and hyperparameters (`C`, `gamma`, `epsilon`) affect model behavior.

## 📊 Datasets

Both datasets are loaded directly via `sklearn.datasets` — no external files or manual cleaning required, since these are standard pre-processed benchmark datasets.

| Dataset | Task | Samples | Features | Target |
|---|---|---|---|---|
| Iris (`load_iris`) | Classification | 150 | 4 | 3 flower species |
| Diabetes (`load_diabetes`) | Regression | 442 | 10 | Disease progression (continuous) |

## 🛠️ Model Details

**SVC (svc_implementation.ipynb)**
- Kernels explored: RBF (default), linear, polynomial, sigmoid
- Manual `C` sweep tested: 0.5, 1, 2, 3, 4, 5
- Train/test split with `random_state=42` for reproducibility

**SVR (svr_implementation.ipynb)**
- Kernels explored: RBF (default), linear, polynomial
- Hyperparameter tuning: `GridSearchCV` over `C` [1, 2, 5, 10, 50, 100], `kernel` [rbf, linear], `epsilon` [0.01, 0.1, 0.2, 0.3, 0.5], scored on R², cv=5
- Preprocessing: `StandardScaler` (feature scaling matters for SVM's distance-based margins)
- Evaluation metric: `r2_score`

## 📈 Results

### SVC — Kernel comparison (default C)

| Kernel | Accuracy |
|---|---|
| RBF (default) | 0.933 |
| Linear | 0.911 |
| Poly | 0.867 |
| Sigmoid | 0.911 |

### SVC — Best model (RBF kernel) classification report

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| 0 | 1.00 | 1.00 | 1.00 | 15 |
| 1 | 0.93 | 0.88 | 0.90 | 16 |
| 2 | 0.87 | 0.93 | 0.90 | 14 |
| **Accuracy** | | | **0.93** | 45 |

### SVC — C-value sweep (RBF kernel)

| C | Accuracy |
|---|---|
| 0.5 | 0.911 |
| 1 | 0.933 |
| 2 | 0.911 |
| 3 | 0.911 |
| 4 | 0.933 |
| 5 | 0.933 |

### SVR — Kernel comparison

| Kernel | Test R² |
|---|---|
| RBF (default `SVR()`) | 0.488 |
| Linear | 0.443 |
| Poly | 0.242 |

### SVR — Default model vs. GridSearchCV-tuned model

| Model | Train R² | Test R² |
|---|---|---|
| Default `SVR()` (RBF) | 0.660 | 0.488 |
| GridSearchCV best (`kernel=linear, C=10, epsilon=0.1`) | 0.515 | 0.474 |

## 💡 Key Learnings

- On Iris, the default RBF kernel (0.933 accuracy) outperformed linear, poly, and sigmoid — non-linear separability in the data favored RBF's flexibility.
- Tuning `C` alone on SVC gave marginal gains (0.911 → 0.933 at C=1, 4, 5) — kernel choice mattered more than this particular hyperparameter for this dataset.
- On SVR, the untuned default `SVR()` (test R² = 0.488) actually outperformed the GridSearchCV-selected "best" model (test R² = 0.474) — a good reminder that grid search optimizes cross-validation score on the training folds, which doesn't always guarantee a better held-out test score, especially with a modest dataset like Diabetes (442 samples).
- SVC and SVR share the same optimization philosophy — the difference is a margin around a *boundary* (classification) vs. a margin around a *function* (regression, via the epsilon-insensitive tube).
- Feature scaling (`StandardScaler`) was essential for both models, since SVM margins are distance-based.

## 🔭 Future Improvements

- Investigate why GridSearchCV's selected model underperformed the default on the test set — possibly widen the `epsilon`/`C` grid or use nested cross-validation.
- Try SVC/SVR on a real-world (non-built-in) dataset with actual data cleaning involved.
- Visualize decision boundaries for SVC and the epsilon-tube for SVR.

## ⚙️ Tech Stack

- Python, scikit-learn, pandas, Jupyter Notebook

## ▶️ How to Run

```bash
pip install scikit-learn pandas
jupyter notebook svc_implementation.ipynb   # or svr_implementation.ipynb
```

---

*Part of an ongoing ML practice series. Previous: Decision Tree Classification. Next up: Random Forest, Bagging, and Boosting.*
