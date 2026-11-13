# Mohirdev — Data Science & AI Praktikum

My notebooks from the **"Data Science va Sun'iy Intellekt Praktikum"** course at [Mohirdev](https://mohirdev.uz): end-of-module projects, Kaggle competition entries, and the Machine Learning module lessons.

**Samoyiddin Jo'rayev** · Computer Engineering student, Inha University in Tashkent · Data Science

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-189FDD)
![LightGBM](https://img.shields.io/badge/LightGBM-2C7BB6)
![CatBoost](https://img.shields.io/badge/CatBoost-FFCC00)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/🤗%20Transformers-FFD21E)

> Notebooks are written in **Uzbek** (course language). Summaries below are in English.

---

## 🏆 Projects

| # | Project | Task | Models | Result |
|---|---------|------|--------|--------|
| 1 | [Mushroom classification](projects/01_mushroom_classification.ipynb) | Edible vs. poisonous (binary classification) | KNN, Decision Tree, Random Forest, Logistic Regression | Random Forest — **100% accuracy** on test and in 5-fold CV |
| 2 | [Bank customer churn](projects/02_bank_customer_churn.ipynb) | Predict which bank customers will leave | Logistic Regression, Decision Tree, SVM, Random Forest, XGBoost | Random Forest — **F1 0.627**, ROC-AUC 0.864 (class-balanced) |
| 3 | [Wholesale customer segmentation](projects/03_wholesale_customer_segmentation.ipynb) | Segment clients by spending categories | K-Means (elbow + silhouette) | **4 segments** with marketing recommendations for each |
| 4 | [IMDB sentiment: RNN vs DistilBERT](projects/04_imdb_sentiment_rnn_vs_distilbert.ipynb) | Positive/negative movie reviews (NLP) | PyTorch RNN, fine-tuned DistilBERT | DistilBERT **87.6%** vs RNN 59.6% accuracy |

**Highlights**
- **Mushrooms** — full EDA, compared 4 classifiers, feature importance, and an extra experiment checking whether fewer features give the same result. Focus on false negatives (a poisonous mushroom labelled safe).
- **Bank churn** — anomaly checks, one-hot encoding, scaling; shows the "accuracy trap" on imbalanced data and how `class_weight='balanced'` raises recall.
- **Wholesale segmentation** — chose k=4 over the silhouette-optimal k=2 because k=2 only repeats the existing HoReCa / retail split; each segment is profiled with concrete actions for marketers.
- **Sentiment analysis** — built a vocabulary and trained an RNN from scratch, then fine-tuned DistilBERT with Hugging Face `Trainer`; tested both on new reviews.

## 📊 Kaggle competitions (Mohirdev)

| Competition | Metric | Approach | CV score |
|-------------|--------|----------|----------|
| [Bank customer churn prediction](kaggle-competitions/bank_customer_churn_prediction.ipynb) | ROC-AUC | Logistic Regression baseline → LightGBM, XGBoost, CatBoost; weighted blend (CatBoost 0.85 + LightGBM 0.15) | **AUC ≈ 0.935** |
| [Cirrhosis patient survival](kaggle-competitions/cirrhosis_survival_prediction.ipynb) | Multi-class log loss (3 outcomes) | Logistic Regression baseline → LightGBM + XGBoost blend, extra original data | **log loss 0.365** (baseline 0.724) |

Both use 5-fold stratified cross-validation, out-of-fold blending, feature-importance analysis and produce a `submission.csv`. Competition data is not included — download it from the competition page and place it in `./data`.

## 📚 Machine Learning module (Module 5) — lessons

| # | Notebook | Topic | Dataset |
|---|----------|-------|---------|
| 01 | [End-to-end ML project](ml-lessons/01_end_to_end_ml_project.ipynb) | CRISP-DM steps, EDA, train/test split | California housing |
| 02 | [Data preparation pipeline](ml-lessons/02_data_preparation_pipeline.ipynb) | `Pipeline`, `ColumnTransformer`, custom transformers | California housing |
| 03 | [Model training & evaluation](ml-lessons/03_model_training_and_evaluation.ipynb) | Linear Regression, Decision Tree, Random Forest, cross-validation, saving models | California housing |
| 04 | [Simple linear regression](ml-lessons/04_simple_linear_regression.ipynb) | Price vs. area | Tashkent (Chilonzor) housing |
| 05 | [Linear regression with scikit-learn](ml-lessons/05_linear_regression_sklearn.ipynb) | `LinearRegression`, evaluation | Tashkent (Chilonzor) housing |
| 06 | [Multiple linear regression](ml-lessons/06_multiple_linear_regression.ipynb) | Several features, train/test split | Tashkent housing |
| 07 | [Polynomial / non-linear regression](ml-lessons/07_polynomial_regression.ipynb) | `PolynomialFeatures`, `curve_fit` | China GDP |
| 08 | [k-NN classification](ml-lessons/08_knn_breast_cancer.ipynb) | k-NN, `GridSearchCV` | Breast cancer diagnosis |
| 09 | [Decision tree](ml-lessons/09_decision_tree_drug_recommendation.ipynb) | Decision Tree, Random Forest | Drug recommendation (drug200) |
| 10 | [K-Means](ml-lessons/10_kmeans_clustering.ipynb) | K-Means on synthetic blobs | `make_blobs` |
| 11 | [Customer segmentation](ml-lessons/11_customer_segmentation_kmeans.ipynb) | Normalization + K-Means | Customer data |
| 12 | [Hierarchical clustering](ml-lessons/12_hierarchical_clustering_cars.ipynb) | Agglomerative clustering, dendrogram | Cars |

`ml-lessons/in-progress/` holds two practice tasks that are not finished yet (Tashkent house prices, ad spend vs. sales).

## 🛠 Skills

Data cleaning & EDA · feature engineering · scikit-learn pipelines · regression · classification · clustering · imbalanced data · cross-validation · gradient boosting (XGBoost, LightGBM, CatBoost) · model blending · NLP with RNNs and Transformers · visualization (matplotlib, seaborn)

## ▶️ How to run

Most notebooks load data straight from public URLs and run on [Google Colab](https://colab.research.google.com/) without changes (open a notebook on GitHub → replace `github.com` with `githubtocolab.com` in the URL).

Locally:

```bash
git clone https://github.com/samoyiddin1394/mohirdev-data-science.git
cd mohirdev-data-science
pip install -r requirements.txt
jupyter notebook
```

The IMDB notebook is best run on a GPU (Colab: *Runtime → Change runtime type → GPU*).

## 🔗 Related

- [tashkent_real_estate_project](https://github.com/samoyiddin1394/tashkent_real_estate_project) — Tashkent apartment price analysis
- [ds_praktikum](https://github.com/samoyiddin1394/ds_praktikum) — earlier pandas exercises from the same course (Module 3)

---

## 🇺🇿 Qisqacha

Bu repozitoriyada Mohirdev'dagi **"Data Science va Sun'iy Intellekt Praktikum"** kursida bajargan ishlarim jamlangan:

- **`projects/`** — modullar bo'yicha yakuniy amaliy vazifalar: qo'ziqorinlarni klassifikatsiya qilish, bank mijozlarining ketishini bashorat qilish, ulgurji savdo mijozlarini segmentlash, IMDB sharhlarini RNN va DistilBERT bilan tasniflash.
- **`kaggle-competitions/`** — Mohirdev Kaggle musobaqalari: bank mijozlarining ketishi (ROC-AUC ≈ 0.935) va sirroz bemorlarining holatini bashorat qilish (log loss 0.365).
- **`ml-lessons/`** — 5-modul (Machine Learning) darslari: regressiya, klassifikatsiya, klasterlash, pipeline.

Barcha daftarlar o'zbek tilida yozilgan.
