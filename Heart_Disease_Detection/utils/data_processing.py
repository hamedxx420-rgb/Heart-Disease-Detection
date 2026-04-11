from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw_data.csv"
CLEANED_DATA_PATH = DATA_DIR / "cleaned_data.csv"

EXPECTED_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target",
]

CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
NUMERICAL_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
TARGET_COLUMN = "target"


def load_dataset(path: Path | str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the dataset and validate the expected schema."""
    dataset_path = Path(path)
    df = pd.read_csv(dataset_path)
    missing_columns = [column for column in EXPECTED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")
    return df[EXPECTED_COLUMNS].copy()


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize types, fill missing values, and normalize the target column."""
    cleaned = df.copy()
    cleaned = cleaned.replace("?", pd.NA)

    for column in EXPECTED_COLUMNS:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    for column in NUMERICAL_FEATURES:
        cleaned[column] = cleaned[column].fillna(cleaned[column].median())

    for column in CATEGORICAL_FEATURES:
        cleaned[column] = cleaned[column].fillna(cleaned[column].mode().iloc[0])

    cleaned[TARGET_COLUMN] = cleaned[TARGET_COLUMN].fillna(0).astype(int)
    cleaned[TARGET_COLUMN] = (cleaned[TARGET_COLUMN] > 0).astype(int)
    return cleaned


def create_clean_dataset(
    raw_path: Path | str = RAW_DATA_PATH,
    cleaned_path: Path | str = CLEANED_DATA_PATH,
) -> pd.DataFrame:
    """Load, clean, and save the cleaned dataset."""
    cleaned = clean_dataset(load_dataset(raw_path))
    output_path = Path(cleaned_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_path, index=False)
    return cleaned


def get_feature_lists() -> Dict[str, List[str]]:
    return {
        "categorical": CATEGORICAL_FEATURES.copy(),
        "numerical": NUMERICAL_FEATURES.copy(),
        "target": [TARGET_COLUMN],
    }


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    return df.drop(columns=[TARGET_COLUMN]), df[TARGET_COLUMN]


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numerical", numeric_pipeline, NUMERICAL_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def dataset_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return a compact statistical summary for reporting and notebooks."""
    summary = df.describe(include="all").transpose()
    summary["missing_values"] = df.isna().sum()
    return summary
