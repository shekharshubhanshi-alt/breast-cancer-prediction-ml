# Breast Cancer Diagnosis Prediction using Machine Learning

Classifies breast tumours as **malignant** or **benign** from 30 cell-nucleus measurements, using the Wisconsin Diagnostic Breast Cancer (WDBC) dataset. Five classifiers are compared, the stronger ones are tuned with cross-validated grid search, and PCA is used to see how far the features can be compressed.

**Author:** Shubhanshi Shekhar
**Registration No.:** 2427010175
**Program:** AIML, Section C

> This is an academic project. It is not a medical device and should not be used for real diagnosis.

## Dataset

- WDBC dataset (UCI Machine Learning Repository), loaded directly from scikit-learn, so nothing needs to be downloaded
- 569 samples, 30 numeric features, no missing values
- 357 benign and 212 malignant samples

## Method

1. Exploratory analysis (class balance, distributions, correlations)
2. Stratified 80/20 train/test split (`random_state = 42`)
3. Feature scaling with `StandardScaler`, kept inside a `Pipeline` so there is no data leakage
4. Baseline models: Logistic Regression, Decision Tree, Random Forest, SVM (RBF), KNN
5. Hyperparameter tuning with 5-fold stratified `GridSearchCV` (SVM, Random Forest, Logistic Regression)
6. Evaluation with accuracy, precision, recall, F1, confusion matrix and ROC-AUC
7. PCA to reduce the 30 features

## Results (test set, 114 samples)

| Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| **SVM (RBF), tuned (C=10, gamma=0.01)** | **0.9825** | **0.9861** | **0.9977** |
| Logistic Regression, tuned (C=0.1) | 0.9737 | 0.9793 | 0.9957 |
| Random Forest, tuned | 0.9474 | 0.9583 | 0.9937 |

PCA: 10 components keep 95% of the variance, and an SVM on them reaches 0.9737 accuracy.

Full tables are in `results/` and plots are in `figures/`.

## Project structure

```
breast-cancer-prediction/
├── src/
│   └── main.py              # complete pipeline
├── results/                 # CSV tables produced by main.py
├── figures/                 # plots produced by main.py
├── requirements.txt
├── .gitignore
└── README.md
```

## How to run

```bash
pip install -r requirements.txt
python src/main.py
```

Running the script prints the metrics and regenerates everything in `results/` and `figures/`.

## Tech used

Python 3, NumPy, pandas, matplotlib, scikit-learn

## References

- W. H. Wolberg, W. N. Street, O. L. Mangasarian, *Breast Cancer Wisconsin (Diagnostic) Data Set*, UCI Machine Learning Repository
- F. Pedregosa et al., *Scikit-learn: Machine Learning in Python*, JMLR 12, 2011
