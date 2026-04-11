from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from rule_based_system.rules import assess_patient

INPUT_FIELDS = [
    ("age", int, "Age in years"),
    ("sex", int, "Sex (0=female, 1=male)"),
    ("cp", int, "Chest pain type (0-3)"),
    ("trestbps", int, "Resting blood pressure"),
    ("chol", int, "Serum cholesterol"),
    ("fbs", int, "Fasting blood sugar > 120 mg/dl (0/1)"),
    ("restecg", int, "Resting ECG result (0-2)"),
    ("thalach", int, "Maximum heart rate achieved"),
    ("exang", int, "Exercise induced angina (0/1)"),
    ("oldpeak", float, "ST depression induced by exercise"),
    ("slope", int, "Slope of the peak exercise ST segment (0-2)"),
    ("ca", int, "Number of major vessels colored by fluoroscopy"),
    ("thal", int, "Thalassemia class (0-3)"),
]


def collect_user_input() -> Dict[str, float | int]:
    payload: Dict[str, float | int] = {}
    for field_name, field_type, prompt in INPUT_FIELDS:
        payload[field_name] = field_type(input(f"{prompt}: ").strip())
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the rule-based heart disease expert system.")
    parser.add_argument(
        "--input-json",
        help="Optional JSON object containing patient features. If omitted, interactive prompts are used.",
    )
    args = parser.parse_args()

    payload = json.loads(args.input_json) if args.input_json else collect_user_input()
    result = assess_patient(payload)

    print(f"Risk label: {result['risk_label']}")
    print(f"Binary prediction: {result['prediction']}")
    print(f"Score: {result['score']}")
    print("Reasons:")
    for reason in result["reasons"]:
        print(f"- {reason}")


if __name__ == "__main__":
    main()
