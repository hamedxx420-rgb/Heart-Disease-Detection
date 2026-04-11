from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from rule_based_system.rules import evaluate_dataframe
from utils.data_processing import build_preprocessor, create_clean_dataset, split_features_target

MODEL_PATH = PROJECT_ROOT / "ml_model" / "heart_disease_decision_tree.joblib"
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics.json"
REPORT_PATH = PROJECT_ROOT / "reports" / "accuracy_comparison.md"
IMPORTANCE_PATH = PROJECT_ROOT / "reports" / "figures" / "feature_importance.png"
VALIDATION_PATH = PROJECT_ROOT / "reports" / "validation_predictions.csv"


def get_train_test_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    cleaned_df = create_clean_dataset()
    features, target = split_features_target(cleaned_df)
    return train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )


def build_model_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", DecisionTreeClassifier(random_state=42)),
        ]
    )


def evaluate_predictions(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
    }


def feature_importance_frame(model: Pipeline) -> pd.DataFrame:
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    return pd.DataFrame(
        {
            "feature": preprocessor.get_feature_names_out(),
            "importance": classifier.feature_importances_,
        }
    ).sort_values("importance", ascending=False)


def save_feature_importance_plot(importance_df: pd.DataFrame) -> None:
    top_features = importance_df.head(12).sort_values("importance")
    plt.figure(figsize=(10, 6))
    sns.barplot(data=top_features, x="importance", y="feature", palette="crest")
    plt.title("Decision Tree Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    IMPORTANCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(IMPORTANCE_PATH, dpi=200)
    plt.close()


def write_comparison_report(
    model_metrics: Dict[str, float],
    rule_metrics: Dict[str, float],
    best_params: Dict[str, object],
) -> None:
    report = f"""# Accuracy Comparison Report

## Decision Tree Metrics

| Metric | Value |
| --- | --- |
| Accuracy | {model_metrics['accuracy']} |
| Precision | {model_metrics['precision']} |
| Recall | {model_metrics['recall']} |
| F1-score | {model_metrics['f1_score']} |

Best hyperparameters: `{best_params}`

## Expert System Metrics

| Metric | Value |
| --- | --- |
| Accuracy | {rule_metrics['accuracy']} |
| Precision | {rule_metrics['precision']} |
| Recall | {rule_metrics['recall']} |
| F1-score | {rule_metrics['f1_score']} |

## Observations

- The decision tree is tuned with grid search and typically achieves stronger predictive performance on the validation set.
- The expert system is more transparent because every prediction is tied to explicit rules.
- Combining both approaches in the Streamlit app gives users both a data-driven score and a human-readable explanation.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def train_and_evaluate() -> Dict[str, object]:
    x_train, x_test, y_train, y_test = get_train_test_data()

    param_grid = {
        "classifier__criterion": ["gini", "entropy"],
        "classifier__max_depth": [3, 5, 7, 9, None],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4],
    }

    search = GridSearchCV(
        estimator=build_model_pipeline(),
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        scoring="f1",
    )
    search.fit(x_train, y_train)

    best_model = search.best_estimator_
    y_pred = best_model.predict(x_test)
    model_metrics = evaluate_predictions(y_test, y_pred)
    importance_df = feature_importance_frame(best_model)
    save_feature_importance_plot(importance_df)

    rule_predictions = evaluate_dataframe(x_test)["prediction"]
    rule_metrics = evaluate_predictions(y_test, rule_predictions)

    validation_df = x_test.copy()
    validation_df["actual"] = y_test.values
    validation_df["decision_tree_prediction"] = y_pred
    validation_df["expert_system_prediction"] = rule_predictions.values
    VALIDATION_PATH.write_text(validation_df.to_csv(index=False), encoding="utf-8")

    metrics_payload = {
        "best_params": search.best_params_,
        "decision_tree": model_metrics,
        "expert_system": rule_metrics,
        "classification_report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
    }
    METRICS_PATH.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    write_comparison_report(model_metrics, rule_metrics, search.best_params_)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)

    return {
        "model": best_model,
        "model_metrics": model_metrics,
        "rule_metrics": rule_metrics,
        "best_params": search.best_params_,
    }


if __name__ == "__main__":
    result = train_and_evaluate()
    print("Training complete.")
    print(json.dumps(result["model_metrics"], indent=2))
    print("Expert system comparison:")
    print(json.dumps(result["rule_metrics"], indent=2))
