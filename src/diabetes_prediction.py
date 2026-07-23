"""Diabetes onset prediction on the Pima Indians dataset.

Production pipeline: biological zero-as-missing handling, leak-free median
imputation and scaling (fit on the training partition only), feature
engineering performed inside the sklearn pipeline, stratified evaluation of
four classifiers, and publication-ready EDA and benchmark outputs.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe rendering

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.base import ClassifierMixin  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import (  # noqa: E402
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import (  # noqa: E402
    FunctionTransformer,
    OneHotEncoder,
    StandardScaler,
)
from sklearn.svm import SVC  # noqa: E402

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "diabetes.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET = "Outcome"

# 0 is a missing-value placeholder for these columns; 0 pregnancies is valid.
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
BASE_FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
NUMERIC_FEATURES = BASE_FEATURES + ["Glucose_Age_Ratio", "Pregnancies_Age_Interaction"]
CATEGORICAL_FEATURES = ["BMI_Category", "Age_Group"]

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("diabetes_prediction")


# --------------------------------------------------------------------------- #
# Data loading and biological cleaning
# --------------------------------------------------------------------------- #
def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV and replace biological zero placeholders with NaN."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download it from "
            "https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database "
            "and place diabetes.csv in data/raw/."
        )
    df = pd.read_csv(path)
    expected = set(BASE_FEATURES + [TARGET])
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {sorted(missing)}")
    df[ZERO_AS_MISSING] = df[ZERO_AS_MISSING].replace(0, np.nan)
    logger.info(
        "Loaded %d rows; NaN counts after zero->NaN:\n%s",
        len(df),
        df[ZERO_AS_MISSING].isna().sum().to_string(),
    )
    return df


# --------------------------------------------------------------------------- #
# Feature engineering (row-wise, leak-free by construction)
# --------------------------------------------------------------------------- #
def engineer_features(X: pd.DataFrame) -> pd.DataFrame:
    """Add ratio, interaction, and binned categorical features.

    Every operation is row-wise or uses fixed clinical bins, so this transform
    carries no fitted state and cannot leak information from the test split.
    """
    X = X.copy()
    X["Glucose_Age_Ratio"] = X["Glucose"] / (X["Age"] + 1e-5)
    X["Pregnancies_Age_Interaction"] = X["Pregnancies"] * X["Age"]
    X["BMI_Category"] = pd.cut(
        X["BMI"],
        bins=[0, 18.5, 24.9, 29.9, np.inf],
        labels=["Underweight", "Normal", "Overweight", "Obese"],
    ).astype("object")
    X["Age_Group"] = pd.cut(
        X["Age"],
        bins=[20, 30, 40, 50, 60, np.inf],
        labels=["20-29", "30-39", "40-49", "50-59", "60+"],
        right=False,
    ).astype("object")
    return X


def build_preprocessor() -> Pipeline:
    """Impute (median) -> engineer -> one-hot encode + scale.

    Imputation, encoding, and scaling are fit within the pipeline, so they
    observe the training partition only.
    """
    impute = SimpleImputer(strategy="median").set_output(transform="pandas")
    engineer = FunctionTransformer(engineer_features).set_output(transform="pandas")
    encode_scale = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(
        steps=[
            ("impute", impute),
            ("engineer", engineer),
            ("encode_scale", encode_scale),
        ]
    )


def build_models() -> dict[str, ClassifierMixin]:
    """Return the classifiers to benchmark under an identical pipeline."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            random_state=RANDOM_STATE
        ),
        "SVM": SVC(random_state=RANDOM_STATE),
    }


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def evaluate_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[pd.DataFrame, Pipeline, str]:
    """Fit each model on an identical leak-free pipeline.

    Returns the benchmark table, the best fitted pipeline (by macro F1), and
    the best model's name.
    """
    records: list[dict[str, object]] = []
    fitted: dict[str, Pipeline] = {}
    for name, model in build_models().items():
        pipe = Pipeline([("preprocess", build_preprocessor()), ("model", model)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        fitted[name] = pipe
        records.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Macro F1": f1_score(y_test, y_pred, average="macro"),
                "Weighted F1": f1_score(y_test, y_pred, average="weighted"),
            }
        )
        logger.info(
            "%s -> acc=%.4f macroF1=%.4f",
            name,
            records[-1]["Accuracy"],
            records[-1]["Macro F1"],
        )
    bench = (
        pd.DataFrame(records)
        .sort_values("Macro F1", ascending=False)
        .reset_index(drop=True)
    )
    best_name = str(bench.iloc[0]["Model"])
    return bench, fitted[best_name], best_name


# --------------------------------------------------------------------------- #
# Publication-ready outputs
# --------------------------------------------------------------------------- #
def save_correlation_heatmap(df: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        df[BASE_FEATURES + [TARGET]].corr(),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
    )
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_class_balance(y: pd.Series, path: Path) -> None:
    counts = y.value_counts().sort_index()
    labels = ["Non-diabetic (0)", "Diabetic (1)"]
    plt.figure(figsize=(6, 5))
    sns.barplot(x=labels, y=counts.to_numpy(), hue=labels, palette="viridis", legend=False)
    plt.xlabel("Outcome")
    plt.ylabel("Count")
    plt.title("Class Balance")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_confusion_matrix(
    y_true: pd.Series, y_pred: np.ndarray, model_name: str, path: Path
) -> None:
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        confusion_matrix(y_true, y_pred),
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(f"Confusion Matrix ({model_name})")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset()

    X = df[BASE_FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    logger.info("Stratified split -> train=%d test=%d", len(X_train), len(X_test))

    bench, best_pipe, best_name = evaluate_models(X_train, X_test, y_train, y_test)
    best_pred = best_pipe.predict(X_test)

    save_correlation_heatmap(df, OUTPUT_DIR / "correlation_heatmap.png")
    save_class_balance(y, OUTPUT_DIR / "class_balance.png")
    save_confusion_matrix(y_test, best_pred, best_name, OUTPUT_DIR / "confusion_matrix.png")

    bench.to_csv(OUTPUT_DIR / "benchmark_results.csv", index=False)

    metric_cols = ["Accuracy", "Macro F1", "Weighted F1"]
    print("\n=== Benchmark (stratified 80/20, leak-free pipeline) ===")
    print(
        bench.to_string(
            index=False,
            formatters={c: "{:.4f}".format for c in metric_cols},
        )
    )
    print(f"\nBest model: {best_name}")
    print("\nClassification report (best model):")
    print(classification_report(y_test, best_pred))
    logger.info("Outputs written to %s", OUTPUT_DIR)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 - top-level guard for a CLI script
        logger.exception("Pipeline failed: %s", exc)
        raise
