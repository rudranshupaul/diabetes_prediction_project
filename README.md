# Diabetes Dataset (Pima Indians)

A well-documented dataset and codebase for building and evaluating machine learning models for diabetes prediction, using clinical diagnostic data from female patients of Pima Indian heritage.

---

## Project Overview

This repository contains dataset documentation and Python code for building and evaluating machine learning models that predict diabetes using clinical data from the Pima Indians Diabetes Dataset. The project encompasses preprocessing, feature engineering, model training, and evaluation steps, following best practices for transparency and reproducibility.

---

## Getting Started

To run the code and reproduce the results:

1. Download `diabetes.csv` from [the official Kaggle page](https://www.kaggle.com/datasets/mathchi/diabetes-data-set).
2. Place `diabetes.csv` in your working directory, next to your `.py` script.
3. Install Python libraries:
    ```
    pip install pandas numpy matplotlib seaborn scikit-learn
    ```
4. Run the code:
    - For a Python script:
      ```
      python diabetespredict.py
      ```
    - For a Jupyter notebook, open and run `diabetes_notebook.ipynb` if provided.

---

## Dataset Context

The dataset was developed by the **National Institute of Diabetes and Digestive and Kidney Diseases**. It comprises diagnostic measurements from females (age ≥ 21) of Pima Indian heritage, with the primary objective being prediction of diabetes onset as defined by the World Health Organization criteria.

---

## Data Source and Collection

- **Original owners:** National Institute of Diabetes and Digestive and Kidney Diseases  
- **Data donor:** Vincent Sigillito, Applied Physics Laboratory, The Johns Hopkins University  
- **Location:** Phoenix, Arizona, USA  
- **Patients:** Female, Pima Indian heritage, age ≥ 21 years  
- **Date received:** 9 May 1990  
- **Kaggle link:** [https://www.kaggle.com/datasets/mathchi/diabetes-data-set](https://www.kaggle.com/datasets/mathchi/diabetes-data-set)

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
| 0.00–1.70 | 246   |
| 1.70–3.40 | 178   |
| 3.40–5.10 | 125   |
| 5.10–6.80 | 50    |
| 6.80–8.50 | 83    |
| 8.50–10.20| 52    |
| 10.20–11.90 | 11  |
| 11.90–13.60 | 19  |
| 13.60–15.30 | 3   |
| 15.30–17.00 | 1   |

#### Glucose

| Range        | Count |
|--------------|-------|
| 0.00–19.90   | 5     |
| 39.80–59.70  | 4     |
| 59.70–79.60  | 32    |
| 79.60–99.50  | 156   |
| 99.50–119.40 | 211   |
| 119.40–139.30 | 163  |
| 139.30–159.20 | 95   |
| 159.20–179.10 | 56   |
| 179.10–199.00 | 46   |

#### BloodPressure

| Range         | Count |
|---------------|-------|
| 0.00–12.20    | 35    |
| 12.20–24.40   | 1     |
| 24.40–36.60   | 2     |
| 36.60–48.80   | 13    |
| 48.80–61.00   | 107   |
| 61.00–73.20   | 261   |
| 73.20–85.40   | 243   |
| 85.40–97.60   | 87    |
| 97.60–109.80  | 14    |
| 109.80–122.00 | 5     |

*Distribution summaries for other features are available in the dataset's documentation.*

---

## Research and Usage Notes

- Widely used in research and education:
    - **Smith, J.W., Everhart, J.E., Dickson, W.C., Knowler, W.C., & Johannes, R.S. (1988):**
      Forecasted diabetes onset using the ADAP learning algorithm, achieving 76% sensitivity and specificity.
- Best practices include treating zeros as missing in relevant medical features, scaling inputs, and evaluating classification models using appropriate metrics.

---

## License

- **CC0: Public Domain** — free for research, education, and application.

---

## Tags

Diabetes, Pima Indians, Medical, Machine Learning, Classification, Public Domain

---

## Sources

- National Institute of Diabetes and Digestive and Kidney Diseases
- Kaggle Dataset: https://www.kaggle.com/datasets/mathchi/diabetes-data-set

---

**Maintained by:** Rudranshu Paul  
Feel free to connect via [LinkedIn](www.linkedin.com/in/rudranshu-paul-376024293) or email(rudranshupaul.24@kgpian.iitkgp.ac.in) for collaboration.
