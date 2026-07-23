# Diabetes Dataset (Pima Indians)

A well-documented dataset and codebase for building and evaluating machine learning models for diabetes prediction, using clinical diagnostic data from female patients of Pima Indian heritage.

---
## Project Goal

The primary objective of this project is to build a machine learning model capable of predicting the onset of diabetes in female patients of Pima Indian heritage. This repository serves as a complete, end-to-end demonstration of a data science workflow, tackling a real-world healthcare challenge. The focus is on meticulous data cleaning, exploratory data analysis, feature engineering, and the implementation and evaluation of a predictive model to identify high-risk individuals.

## Project Overview

This repository contains dataset documentation and Python code for building and evaluating machine learning models that predict diabetes using clinical data from the Pima Indians Diabetes Dataset. The workflow runs inside a single leak-free scikit-learn pipeline: biological missing-value handling, median imputation, feature engineering, one-hot encoding, and scaling are all fit on the training partition only, so no test information leaks into preprocessing. Models are compared under a stratified train-test split, and results are exported as publication-ready figures and a consolidated benchmark table.

---

## Project Structure

```
diabetes_prediction_project/
  data/raw/diabetes.csv     # place the downloaded dataset here (not tracked)
  outputs/                  # figures and benchmark table generated at runtime
  src/diabetes_prediction.py
  requirements.txt
  .gitignore
  README.md
  LICENSE
```

---

## Getting Started

To run the code and reproduce the results:

1. Download `diabetes.csv` from [the official Kaggle page](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database).
2. Place `diabetes.csv` in `data/raw/`.
3. Install the dependencies:
    ```
    pip install -r requirements.txt
    ```
4. Run the pipeline:
    ```
    python src/diabetes_prediction.py
    ```

The script prints the benchmark table and the classification report for the best model, and writes all figures plus `benchmark_results.csv` to `outputs/`.

---

## Pipeline and Methodology

- **Biological missing values:** In this dataset a value of 0 is a missing-data placeholder for `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI`. These zeros are replaced with `NaN`. A 0 in `Pregnancies` is valid biological data and is left untouched.
- **Leak-free preprocessing:** A stratified 80/20 split is taken first. Median imputation, one-hot encoding, and standard scaling are then fit strictly on the training partition and applied to the test partition inside a single scikit-learn `Pipeline`, eliminating data leakage.
- **Feature engineering (inside the pipeline):** `Glucose_Age_Ratio`, a `Pregnancies` by `Age` interaction, and binned `BMI_Category` and `Age_Group` features. These derivations are row-wise or use fixed clinical bins, so they carry no fitted state and cannot leak.
- **Stratified evaluation:** The split preserves the original class distribution across train and test.
- **Models benchmarked:** Logistic Regression, Random Forest, HistGradientBoosting, and SVM, each trained on the identical preprocessing pipeline.

---

## Benchmark Results

Stratified 80/20 split, `random_state=42`, identical leak-free pipeline per model. Metrics are computed on the held-out test set.

| Model | Accuracy | Macro F1 | Weighted F1 |
|----------------------|----------|----------|-------------|
| HistGradientBoosting | 0.7792   | 0.7508   | 0.7759      |
| Random Forest        | 0.7403   | 0.7068   | 0.7364      |
| Logistic Regression  | 0.7338   | 0.6980   | 0.7290      |
| SVM                  | 0.7403   | 0.6938   | 0.7294      |

Best model: **HistGradientBoosting**. Generated artifacts in `outputs/`:

- `correlation_heatmap.png`
- `class_balance.png`
- `confusion_matrix.png`
- `benchmark_results.csv`

---

## Dataset Context

The dataset was developed by the **National Institute of Diabetes and Digestive and Kidney Diseases**. It comprises diagnostic measurements from females (age >= 21) of Pima Indian heritage, with the primary objective being prediction of diabetes onset as defined by the World Health Organization criteria.

---

## Data Source and Collection

- **Original owners:** National Institute of Diabetes and Digestive and Kidney Diseases
- **Data donor:** Vincent Sigillito, Applied Physics Laboratory, The Johns Hopkins University
- **Location:** Phoenix, Arizona, USA
- **Patients:** Female, Pima Indian heritage, age >= 21 years
- **Date received:** 9 May 1990
- **Kaggle link:** [https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

---

## Features

| Column                   | Description                                                                              |
|--------------------------|------------------------------------------------------------------------------------------|
| Pregnancies              | Number of times pregnant                                                                 |
| Glucose                  | Plasma glucose concentration 2 hours into an oral glucose tolerance test (mg/dL)         |
| BloodPressure            | Diastolic blood pressure (mm Hg)                                                         |
| SkinThickness            | Triceps skin fold thickness (mm)                                                         |
| Insulin                  | 2-Hour serum insulin (mu U/ml)                                                           |
| BMI                      | Body mass index (weight in kg/(height in m)^2)                                           |
| DiabetesPedigreeFunction | Genetic risk score ("diabetes pedigree function")                                        |
| Age                      | Age in years                                                                             |
| Outcome                  | 0: non-diabetic, 1: diabetic (class variable)                                            |

---

## Data Details

- **Number of instances:** 768
- **Number of features:** 8 + 1 class label
- **All columns are numeric**
- **Missing values:** Some zeros denote missing data in medical measurements and should be cleaned before analysis.

### Class Distribution

| Outcome | Count |
|---------|-------|
| 0       | 500   |
| 1       | 268   |

---

## Example Data Row

| Pregnancies | Glucose | BloodPressure | SkinThickness | Insulin | BMI  | DiabetesPedigreeFunction | Age | Outcome |
|:-----------:|:-------:|:-------------:|:-------------:|:-------:|:----:|:-----------------------:|:---:|:-------:|
|      6      |   148   |      72       |      35       |    0    | 33.6 |          0.627          |  50  |    1    |

---

## Key Feature Distributions (Binned Samples)

#### Pregnancies

| Range     | Count |
|-----------|-------|
| 0.00-1.70 | 246   |
| 1.70-3.40 | 178   |
| 3.40-5.10 | 125   |
| 5.10-6.80 | 50    |
| 6.80-8.50 | 83    |
| 8.50-10.20| 52    |
| 10.20-11.90 | 11  |
| 11.90-13.60 | 19  |
| 13.60-15.30 | 3   |
| 15.30-17.00 | 1   |

#### Glucose

| Range        | Count |
|--------------|-------|
| 0.00-19.90   | 5     |
| 39.80-59.70  | 4     |
| 59.70-79.60  | 32    |
| 79.60-99.50  | 156   |
| 99.50-119.40 | 211   |
| 119.40-139.30 | 163  |
| 139.30-159.20 | 95   |
| 159.20-179.10 | 56   |
| 179.10-199.00 | 46   |

#### BloodPressure

| Range         | Count |
|---------------|-------|
| 0.00-12.20    | 35    |
| 12.20-24.40   | 1     |
| 24.40-36.60   | 2     |
| 36.60-48.80   | 13    |
| 48.80-61.00   | 107   |
| 61.00-73.20   | 261   |
| 73.20-85.40   | 243   |
| 85.40-97.60   | 87    |
| 97.60-109.80  | 14    |
| 109.80-122.00 | 5     |

*Distribution summaries for other features are available in the dataset's documentation.*

---

## Research and Usage Notes

- Widely used in research and education:
    - **Smith, J.W., Everhart, J.E., Dickson, W.C., Knowler, W.C., & Johannes, R.S. (1988):**
      Forecasted diabetes onset using the ADAP learning algorithm, achieving 76% sensitivity and specificity.
- Best practices include treating zeros as missing in relevant medical features, scaling inputs, and evaluating classification models using appropriate metrics.

---

## License

- **CC0: Public Domain.** Free for research, education, and application.

---

## Tags

Diabetes, Pima Indians, Medical, Machine Learning, Classification, Public Domain

---

## Sources

- National Institute of Diabetes and Digestive and Kidney Diseases
- Kaggle Dataset: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

---

**Maintained by:** Rudranshu Paul
Feel free to connect via [LinkedIn](www.linkedin.com/in/rudranshu-paul-376024293) or email(rudranshupaul.24@kgpian.iitkgp.ac.in) for collaboration.
