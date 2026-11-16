"""
Breast Cancer Diagnosis Prediction using Machine Learning
Author: Shubhanshi Shekhar (2427010175), AIML, Section C

Runs the complete pipeline on the Wisconsin Diagnostic Breast Cancer dataset
(bundled with scikit-learn):
    EDA -> split -> scaling -> baseline models -> grid search -> evaluation -> PCA

Usage (from the project root):
    python src/main.py
Tables are saved in results/ and plots in figures/.
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay,
                             accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
RES, FIG = ROOT / "results", ROOT / "figures"
RES.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)


def metrics(y_true, y_pred):
    return {
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred), 4),
        "Recall": round(recall_score(y_true, y_pred), 4),
        "F1": round(f1_score(y_true, y_pred), 4),
    }


# ------------------------------------------------------------------ 1. Data
data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target          # target: 0 = malignant, 1 = benign
print("Shape:", X.shape)
print("Class counts:", y.value_counts().to_dict(), "| missing values:", int(X.isna().sum().sum()))

# ------------------------------------------------------------------ 2. EDA
fig, ax = plt.subplots(figsize=(4.5, 4))
counts = y.map({0: "Malignant", 1: "Benign"}).value_counts()
ax.bar(counts.index, counts.values, color=["#c44e52", "#4c72b0"])
for i, v in enumerate(counts.values):
    ax.text(i, v + 5, str(v), ha="center")
ax.set_title("Class distribution")
ax.set_ylabel("Samples")
fig.tight_layout()
fig.savefig(FIG / "01_class_distribution.png", dpi=150)
plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, col in zip(axes, ["mean radius", "mean concavity", "worst perimeter"]):
    ax.hist(X.loc[y == 0, col], bins=25, alpha=0.65, label="Malignant")
    ax.hist(X.loc[y == 1, col], bins=25, alpha=0.65, label="Benign")
    ax.set_title(col)
    ax.legend()
fig.tight_layout()
fig.savefig(FIG / "02_feature_distributions.png", dpi=150)
plt.close(fig)

corr = X.corr()
fig, ax = plt.subplots(figsize=(9, 8))
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(30))
ax.set_yticks(range(30))
ax.set_xticklabels(X.columns, rotation=90, fontsize=6)
ax.set_yticklabels(X.columns, fontsize=6)
fig.colorbar(im, fraction=0.04)
ax.set_title("Feature correlation matrix")
fig.tight_layout()
fig.savefig(FIG / "03_correlation_heatmap.png", dpi=150)
plt.close(fig)

upper = corr.abs().where(np.triu(np.ones(corr.shape), 1).astype(bool)).stack()
print("\nMost correlated feature pairs:\n", upper.nlargest(3).round(3))
print("\nGroup means:\n", X.groupby(y.map({0: "malignant", 1: "benign"}))[["mean radius", "mean concavity"]].mean().round(2))

# ------------------------------------------------------------ 3. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)
print(f"\nTrain: {len(X_train)}  Test: {len(X_test)}")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# ------------------------------------------------- 4. Baseline models
baseline = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=SEED),
    "Random Forest": RandomForestClassifier(random_state=SEED),
    "SVM (RBF)": SVC(random_state=SEED),
    "KNN (k=5)": KNeighborsClassifier(),
}
rows = []
for name, model in baseline.items():
    pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
    pipe.fit(X_train, y_train)
    row = {"Model": name, **metrics(y_test, pipe.predict(X_test))}
    row["CV Accuracy"] = round(cross_val_score(pipe, X_train, y_train, cv=cv).mean(), 4)
    rows.append(row)
baseline_df = pd.DataFrame(rows)
baseline_df.to_csv(RES / "baseline_results.csv", index=False)
print("\n=== Baseline models ===\n", baseline_df.to_string(index=False))

# ------------------------------------------------ 5. Hyperparameter tuning
grids = {
    "SVM (RBF)": (SVC(probability=True, random_state=SEED),
                  {"model__C": [0.1, 1, 10, 100], "model__gamma": ["scale", 0.01, 0.001]}),
    "Random Forest": (RandomForestClassifier(random_state=SEED),
                      {"model__n_estimators": [100, 200, 300], "model__max_depth": [None, 5, 10]}),
    "Logistic Regression": (LogisticRegression(max_iter=2000),
                            {"model__C": [0.01, 0.1, 1, 10, 100]}),
}
tuned_rows, tuned_models = [], {}
for name, (model, grid) in grids.items():
    pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
    search = GridSearchCV(pipe, grid, cv=cv, scoring="f1")
    search.fit(X_train, y_train)
    pred = search.predict(X_test)
    proba = search.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    tuned_models[name] = search
    tuned_rows.append({
        "Model": name, "Best params": search.best_params_,
        "CV F1": round(search.best_score_, 4), **metrics(y_test, pred),
        "ROC-AUC": round(roc_auc_score(y_test, proba), 4),
        "Malignant found": tn, "Malignant missed": fp,
        "Benign flagged malignant": fn, "Benign found": tp,
    })
tuned_df = pd.DataFrame(tuned_rows)
tuned_df.to_csv(RES / "tuned_results.csv", index=False)
print("\n=== Tuned models ===\n", tuned_df.to_string(index=False))

# ----------------------------------------- 6. Plots for tuned models
best = tuned_models["SVM (RBF)"]
fig, ax = plt.subplots(figsize=(4.8, 4.2))
ConfusionMatrixDisplay.from_estimator(
    best, X_test, y_test, display_labels=["Malignant", "Benign"], cmap="Blues", ax=ax)
ax.set_title("Tuned SVM - confusion matrix")
fig.tight_layout()
fig.savefig(FIG / "04_confusion_matrix_svm.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(5.5, 4.5))
for name, search in tuned_models.items():
    RocCurveDisplay.from_estimator(search, X_test, y_test, name=name, ax=ax)
ax.plot([0, 1], [0, 1], "k--", lw=0.8)
ax.set_title("ROC curves (tuned models)")
fig.tight_layout()
fig.savefig(FIG / "05_roc_curves.png", dpi=150)
plt.close(fig)

# --------------------------------------------- 7. Feature importance
rf = RandomForestClassifier(n_estimators=200, random_state=SEED).fit(X_train, y_train)
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
imp.round(4).to_csv(RES / "feature_importance.csv", header=["importance"])
print("\nTop 5 features:\n", imp.head(5).round(3))

fig, ax = plt.subplots(figsize=(6.5, 4.5))
imp.head(10)[::-1].plot.barh(ax=ax, color="#4c72b0")
ax.set_title("Top 10 features (Random Forest)")
fig.tight_layout()
fig.savefig(FIG / "06_feature_importance.png", dpi=150)
plt.close(fig)

# ------------------------------------------------------------ 8. PCA
scaler = StandardScaler().fit(X_train)
pca_full = PCA().fit(scaler.transform(X_train))
cum = np.cumsum(pca_full.explained_variance_ratio_)
n95 = int(np.argmax(cum >= 0.95)) + 1
print(f"\nPC1 = {pca_full.explained_variance_ratio_[0]:.3f}, PC2 = {pca_full.explained_variance_ratio_[1]:.3f}")
print(f"Components for 95% variance: {n95}")

pca_svm = Pipeline([("scaler", StandardScaler()), ("pca", PCA(n_components=n95)),
                    ("model", SVC(random_state=SEED))]).fit(X_train, y_train)
pca_acc = accuracy_score(y_test, pca_svm.predict(X_test))
print(f"SVM on {n95} PCA components - test accuracy: {pca_acc:.4f}")
pd.DataFrame({"components_for_95pct": [n95], "svm_test_accuracy": [round(pca_acc, 4)]}).to_csv(
    RES / "pca_results.csv", index=False)

fig, ax = plt.subplots(figsize=(5.5, 4))
ax.plot(range(1, 31), cum, marker="o", ms=3)
ax.axhline(0.95, color="r", ls="--", lw=0.8)
ax.set_xlabel("Number of components")
ax.set_ylabel("Cumulative explained variance")
ax.set_title("PCA explained variance")
fig.tight_layout()
fig.savefig(FIG / "07_pca_variance.png", dpi=150)
plt.close(fig)

print("\nDone. Tables saved in results/, plots in figures/.")
