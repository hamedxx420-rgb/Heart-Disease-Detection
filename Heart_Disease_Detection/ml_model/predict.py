from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import joblib
import pandas as pd

from ml_model.train_model import MODEL_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict heart disease risk using the trained decision tree.")
    parser.add_argument(
        "--input-json",
        required=True,
        help="JSON object containing patient features that match the training schema.",
    )
    args = parser.parse_args()

    model = joblib.load(MODEL_PATH)
    payload = json.loads(args.input_json)
    frame = pd.DataFrame([payload])
    prediction = int(model.predict(frame)[0])
    probability = float(model.predict_proba(frame)[0][1])

    print(json.dumps({"prediction": prediction, "probability": round(probability, 4)}, indent=2))


if __name__ == "__main__":
    main()
