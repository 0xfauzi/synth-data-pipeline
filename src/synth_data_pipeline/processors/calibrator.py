from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


@dataclass
class CalibrationModel:
    method: str
    model: Any

    def apply(self, value: float) -> float:
        if self.model is None:
            return value
        if self.method == "isotonic":
            return float(self.model.predict(np.asarray([value]))[0])
        if self.method == "platt":
            logits = np.asarray([[value]])
            prob = self.model.predict_proba(logits)[0][1]
            return float(prob)
        return value

    def export(self) -> Dict[str, Any]:
        if self.method == "isotonic":
            return {
                "method": "isotonic",
                "x": self.model.X_thresholds_.tolist(),
                "y": self.model.y_thresholds_.tolist(),
            }
        if self.method == "platt":
            return {
                "method": "platt",
                "coef": self.model.coef_.ravel().tolist(),
                "intercept": self.model.intercept_.ravel().tolist(),
            }
        return {"method": "identity"}

    @staticmethod
    def from_export(params: Dict[str, Any]) -> "CalibrationModel":
        method = params.get("method", "identity")
        if method == "isotonic":
            model = IsotonicRegression(out_of_bounds="clip")
            model.X_thresholds_ = np.asarray(params["x"])
            model.y_thresholds_ = np.asarray(params["y"])
            return CalibrationModel(method="isotonic", model=model)
        if method == "platt":
            model = LogisticRegression()
            model.coef_ = np.asarray([params["coef"]])
            model.intercept_ = np.asarray(params["intercept"])
            model.classes_ = np.asarray([0, 1])
            return CalibrationModel(method="platt", model=model)
        return CalibrationModel(method="identity", model=None)


@dataclass
class ProbabilityCalibrator:
    labels: Sequence[str]
    method: str = "isotonic"
    calibrators: Dict[str, CalibrationModel] = field(default_factory=dict)

    def fit(
        self,
        predictions: Iterable[Dict[str, Any]],
        truths: Iterable[Dict[str, Any]],
    ) -> None:
        pred_list = list(predictions)
        truth_list = list(truths)
        if len(pred_list) != len(truth_list):
            raise ValueError("Predictions and truths must have the same length.")

        overall_preds, overall_truth = self._collect_overall(pred_list, truth_list)
        self.calibrators["cringe_prob"] = self._fit_single(overall_preds, overall_truth)

        label_truths = self._collect_labels(pred_list, truth_list)
        for label in self.labels:
            preds, trues = label_truths.get(label, ([], []))
            self.calibrators[label] = self._fit_single(preds, trues)

    def transform(self, predictions: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        calibrated = []
        for pred in predictions:
            updated = deepcopy(pred)
            judgment = updated.get("judgment", updated)
            if "cringe_prob" in judgment:
                judgment["cringe_prob"] = self._apply_value(
                    "cringe_prob", judgment["cringe_prob"]
                )
            labels = judgment.get("labels")
            if isinstance(labels, dict):
                for label, value in labels.items():
                    labels[label] = self._apply_value(label, value)
            calibrated.append(updated)
        return calibrated

    def save(self, path: str | Path) -> None:
        data = {key: model.export() for key, model in self.calibrators.items()}
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, labels: Sequence[str], path: str | Path) -> "ProbabilityCalibrator":
        content = json.loads(Path(path).read_text())
        calibrator = cls(labels=labels)
        for key, params in content.items():
            calibrator.calibrators[key] = CalibrationModel.from_export(params)
        return calibrator

    def _apply_value(self, key: str, value: Any) -> float:
        if not isinstance(value, (int, float)):
            return value
        model = self.calibrators.get(key)
        if not model:
            return float(value)
        return self._clip(model.apply(float(value)))

    def _fit_single(self, preds: Sequence[float], truths: Sequence[int]) -> CalibrationModel:
        preds = np.asarray(preds, dtype=float)
        truths = np.asarray(truths, dtype=int)

        if len(preds) == 0 or len(np.unique(truths)) < 2:
            return CalibrationModel(method="identity", model=None)

        if self.method == "platt":
            model = LogisticRegression(max_iter=200)
            model.fit(preds.reshape(-1, 1), truths)
            return CalibrationModel(method="platt", model=model)

        # default to isotonic regression
        model = IsotonicRegression(out_of_bounds="clip")
        model.fit(preds, truths)
        return CalibrationModel(method="isotonic", model=model)

    @staticmethod
    def _clip(value: float) -> float:
        return max(0.0, min(1.0, value))

    def _collect_overall(
        self,
        predictions: List[Dict[str, Any]],
        truths: List[Dict[str, Any]],
    ) -> tuple[List[float], List[int]]:
        preds: List[float] = []
        targets: List[int] = []
        for pred, truth in zip(predictions, truths):
            judgment = pred.get("judgment", pred)
            prob = judgment.get("cringe_prob")
            truth_value = self._extract_overall_truth(truth)
            if isinstance(prob, (int, float)) and truth_value is not None:
                preds.append(float(prob))
                targets.append(int(truth_value))
        return preds, targets

    def _collect_labels(
        self,
        predictions: List[Dict[str, Any]],
        truths: List[Dict[str, Any]],
    ) -> Dict[str, tuple[List[float], List[int]]]:
        result: Dict[str, tuple[List[float], List[int]]] = {
            label: ([], []) for label in self.labels
        }
        for pred, truth in zip(predictions, truths):
            judgment = pred.get("judgment", pred)
            pred_labels = judgment.get("labels", {})
            truth_labels = self._extract_truth_labels(truth)
            for label in self.labels:
                pred_value = pred_labels.get(label)
                truth_value = truth_labels.get(label)
                if isinstance(pred_value, (int, float)) and truth_value is not None:
                    result[label][0].append(float(pred_value))
                    result[label][1].append(int(truth_value))
        return result

    @staticmethod
    def _extract_truth_labels(truth: Dict[str, Any]) -> Dict[str, int]:
        labels = truth.get("labels") or truth.get("truth") or {}
        converted: Dict[str, int] = {}
        for key, value in labels.items():
            if isinstance(value, bool):
                converted[key] = int(value)
            elif isinstance(value, (int, float)):
                converted[key] = int(round(value))
        return converted

    @staticmethod
    def _extract_overall_truth(truth: Dict[str, Any]) -> Optional[int]:
        for key in ("cringe", "cringe_prob", "label", "overall"):
            value = truth.get(key)
            if isinstance(value, bool):
                return int(value)
            if isinstance(value, (int, float)):
                return int(round(value))
        return None

