from __future__ import annotations

import collections
import collections.abc
from typing import Dict, Iterable, List

import pandas as pd

# Compat shim for older frozendict/experta dependencies on modern Python.
if not hasattr(collections, "Mapping"):
    collections.Mapping = collections.abc.Mapping
if not hasattr(collections, "MutableMapping"):
    collections.MutableMapping = collections.abc.MutableMapping
if not hasattr(collections, "Sequence"):
    collections.Sequence = collections.abc.Sequence

from experta import Fact, KnowledgeEngine, MATCH, Rule, TEST


class Patient(Fact):
    """Facts describing one patient record."""


class HeartDiseaseRiskEngine(KnowledgeEngine):
    def __init__(self) -> None:
        super().__init__()
        self.score = 0
        self.reasons: List[str] = []

    def reset_state(self) -> None:
        self.score = 0
        self.reasons = []

    def add_reason(self, points: int, reason: str) -> None:
        self.score += points
        self.reasons.append(f"{reason} ({points:+d})")

    @Rule(Patient(age=MATCH.age, cp=MATCH.cp), TEST(lambda age, cp: age >= 60 and cp == 0))
    def rule_age_and_asymptomatic_pain(self) -> None:
        self.add_reason(2, "Older patient with asymptomatic chest pain")

    @Rule(
        Patient(trestbps=MATCH.trestbps, exang=MATCH.exang),
        TEST(lambda trestbps, exang: trestbps >= 140 and exang == 1),
    )
    def rule_pressure_and_exercise_angina(self) -> None:
        self.add_reason(2, "Elevated resting blood pressure with exercise-induced angina")

    @Rule(Patient(chol=MATCH.chol, age=MATCH.age), TEST(lambda chol, age: chol >= 240 and age >= 50))
    def rule_cholesterol_and_age(self) -> None:
        self.add_reason(1, "High cholesterol in an older adult")

    @Rule(Patient(oldpeak=MATCH.oldpeak, slope=MATCH.slope), TEST(lambda oldpeak, slope: oldpeak >= 2 and slope == 0))
    def rule_oldpeak_and_slope(self) -> None:
        self.add_reason(2, "Large ST depression with downsloping ST segment")

    @Rule(Patient(ca=MATCH.ca, thal=MATCH.thal), TEST(lambda ca, thal: ca >= 2 and thal == 3))
    def rule_vessels_and_thal(self) -> None:
        self.add_reason(2, "Multiple affected vessels with reversible defect")

    @Rule(
        Patient(thalach=MATCH.thalach, exang=MATCH.exang),
        TEST(lambda thalach, exang: thalach < 120 and exang == 1),
    )
    def rule_low_heart_rate_capacity(self) -> None:
        self.add_reason(2, "Low peak heart rate and exercise-induced angina")

    @Rule(Patient(fbs=MATCH.fbs, chol=MATCH.chol), TEST(lambda fbs, chol: fbs == 1 and chol >= 230))
    def rule_blood_sugar_and_cholesterol(self) -> None:
        self.add_reason(1, "Elevated fasting blood sugar with high cholesterol")

    @Rule(
        Patient(restecg=MATCH.restecg, oldpeak=MATCH.oldpeak),
        TEST(lambda restecg, oldpeak: restecg in (1, 2) and oldpeak >= 1),
    )
    def rule_restecg_and_oldpeak(self) -> None:
        self.add_reason(1, "Abnormal resting ECG plus ST depression")

    @Rule(
        Patient(cp=MATCH.cp, thalach=MATCH.thalach, exang=MATCH.exang),
        TEST(lambda cp, thalach, exang: cp in (1, 2, 3) and thalach >= 150 and exang == 0),
    )
    def rule_protective_good_capacity(self) -> None:
        self.add_reason(-1, "Higher exercise capacity without exercise-induced angina")

    @Rule(
        Patient(trestbps=MATCH.trestbps, chol=MATCH.chol, oldpeak=MATCH.oldpeak),
        TEST(lambda trestbps, chol, oldpeak: trestbps < 130 and chol < 200 and oldpeak < 1),
    )
    def rule_protective_vitals(self) -> None:
        self.add_reason(-1, "Blood pressure, cholesterol, and ST depression are all in safer ranges")

    @Rule(Patient(age=MATCH.age, cp=MATCH.cp, ca=MATCH.ca), TEST(lambda age, cp, ca: age < 45 and cp in (1, 2, 3) and ca == 0))
    def rule_younger_patient(self) -> None:
        self.add_reason(-1, "Younger patient with no visibly affected vessels")

    @Rule(Patient(sex=MATCH.sex, thalach=MATCH.thalach, slope=MATCH.slope), TEST(lambda sex, thalach, slope: sex == 0 and thalach >= 140 and slope == 2))
    def rule_female_protective_pattern(self) -> None:
        self.add_reason(-1, "Higher peak heart rate with upsloping ST segment")

    @Rule(
        Patient(age=MATCH.age, ca=MATCH.ca, oldpeak=MATCH.oldpeak),
        TEST(lambda age, ca, oldpeak: age >= 55 and ca >= 1 and oldpeak >= 1.5),
    )
    def rule_age_vessels_oldpeak(self) -> None:
        self.add_reason(1, "Age, affected vessels, and ST depression point to elevated risk")


def normalize_payload(payload: Dict[str, float | int]) -> Dict[str, float | int]:
    normalized = dict(payload)
    normalized["target"] = int(normalized.get("target", 0))
    return normalized


def score_to_label(score: int) -> str:
    if score >= 4:
        return "high"
    if score >= 2:
        return "moderate"
    return "low"


def assess_patient(payload: Dict[str, float | int]) -> Dict[str, object]:
    engine = HeartDiseaseRiskEngine()
    engine.reset()
    engine.reset_state()
    engine.declare(Patient(**normalize_payload(payload)))
    engine.run()

    return {
        "score": engine.score,
        "risk_label": score_to_label(engine.score),
        "prediction": int(engine.score >= 2),
        "reasons": engine.reasons,
    }


def evaluate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for _, row in df.iterrows():
        result = assess_patient(row.to_dict())
        rows.append(
            {
                "prediction": result["prediction"],
                "risk_label": result["risk_label"],
                "score": result["score"],
                "reasons": " | ".join(result["reasons"]),
            }
        )
    return pd.DataFrame(rows)


def reasons_preview(reasons: Iterable[str]) -> str:
    items = list(reasons)
    return "; ".join(items[:3]) if items else "No rules fired"
