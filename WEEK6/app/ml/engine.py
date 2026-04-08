"""
app/ml/engine.py
────────────────────────────────────────────────────────────────────────────
ML Inference Engine — orchestrates all four detection levels.

Level 1  Anomaly Detection     (unsupervised — Isolation Forest, OC-SVM, Autoencoder)
Level 2  Hybrid Detection      (anomaly gate + classifier confirmation)
Level 3  ML Classification     (XGBoost / RandomForest multi-class)
Level 4  Packet Monitor        (no ML — raw logging only)

Public interface
  engine = MLEngine()
  engine.load()                           # load models from disk
  result = await engine.predict(flow)     # returns PredictionResult
"""

from __future__ import annotations
import asyncio
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, List

import numpy as np
import joblib

from app.config import settings
from app.capture.sniffer import FlowRecord

logger = logging.getLogger("netguard.ml")

SEVERITY_MAP = {
    "BENIGN":        None,          # not a threat
    "PortScan":      "HIGH",
    "DDoS":          "CRITICAL",
    "BruteForce":    "HIGH",
    "MalwareBeacon": "HIGH",
    "PacketAnomaly": "MEDIUM",
    "ZeroDay":       "CRITICAL",
    "ANOMALY":       "MEDIUM",     # generic unsupervised anomaly
}

ATTACK_DISPLAY = {
    "PortScan":      "Port Scan",
    "DDoS":          "DDoS",
    "BruteForce":    "Brute Force",
    "MalwareBeacon": "Malware Beacon",
    "PacketAnomaly": "Packet Anomaly",
    "ZeroDay":       "Zero-Day",
    "ANOMALY":       "Anomaly",
}


@dataclass
class PredictionResult:
    is_threat: bool
    attack_type: Optional[str]       # human-readable label
    severity: Optional[str]          # LOW / MEDIUM / HIGH / CRITICAL
    confidence: float                # 0.0 – 1.0
    ml_level: int                    # 1 – 4
    ml_model: str
    anomaly_score: float             # Isolation Forest score (negative = anomaly)
    raw: dict = field(default_factory=dict)
    latency_ms: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class _AutoencoderWrapper:
    """Thin wrapper around a PyTorch autoencoder for inference."""

    def __init__(self, model, threshold: float, device):
        self.model     = model
        self.threshold = threshold
        self.device    = device

    def score(self, X: np.ndarray) -> np.ndarray:
        """Return reconstruction errors (higher = more anomalous)."""
        import torch
        self.model.eval()
        with torch.no_grad():
            t   = torch.tensor(X, dtype=torch.float32).to(self.device)
            out = self.model(t)
            err = ((t - out) ** 2).mean(dim=1).cpu().numpy()
        return err

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return 1 = normal, -1 = anomaly (same convention as sklearn)."""
        err = self.score(X)
        return np.where(err > self.threshold, -1, 1)


class MLEngine:
    """
    Loads all ML models and provides a single `predict()` coroutine.
    Thread-safe for reads (model inference is GIL-held but models are
    shared read-only after loading).
    """

    def __init__(self):
        self.scaler        = None
        self.iforest       = None
        self.ocsvm         = None
        self.autoencoder   = None
        self.classifier    = None
        self.label_encoder = None
        self._loaded       = False
        self._lock         = asyncio.Lock()

    # ── Loading ────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load all models from disk. Call once at startup."""
        md = settings.model_dir

        self.scaler = self._load_pkl(settings.scaler, "Scaler")
        self.iforest = self._load_pkl(settings.anomaly_model, "IsolationForest")
        self.ocsvm   = self._load_pkl(settings.svm_model, "OneClassSVM")
        self.classifier   = self._load_pkl(settings.classifier_model, "XGBClassifier")
        self.label_encoder = self._load_pkl(settings.label_encoder, "LabelEncoder")
        self.autoencoder  = self._load_autoencoder()

        self._loaded = True
        logger.info("ML engine ready.")

    def _load_pkl(self, path: Path, name: str):
        try:
            m = joblib.load(path)
            logger.info(f"  Loaded {name} from {path}")
            return m
        except Exception as exc:
            logger.warning(f"  Could not load {name} from {path}: {exc}")
            return None

    def _load_autoencoder(self):
        path = settings.autoencoder_model
        try:
            import torch
            # rebuild architecture
            from scripts.train_models import build_autoencoder
            model = build_autoencoder(n_features=21)
            model.load_state_dict(torch.load(path, map_location="cpu"))
            model.eval()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            wrapper = _AutoencoderWrapper(
                model, settings.autoencoder_threshold, device
            )
            logger.info(f"  Loaded Autoencoder from {path}")
            return wrapper
        except Exception as exc:
            logger.warning(f"  Could not load Autoencoder: {exc}")
            return None

    # ── Prediction ─────────────────────────────────────────────────────────

    async def predict(
        self,
        flow: FlowRecord,
        ml_level: int = 1,
    ) -> PredictionResult:
        """
        Run inference for the given FlowRecord.
        ml_level: 1=Anomaly, 2=Hybrid, 3=Classification, 4=Monitor
        """
        t0 = time.perf_counter()

        # Level 4 — no inference
        if ml_level == 4 or not self._loaded:
            return PredictionResult(
                is_threat=False, attack_type=None, severity=None,
                confidence=0.0, ml_level=4, ml_model="None",
                anomaly_score=0.0,
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        X_raw = np.array([flow.to_feature_vector()], dtype=np.float32)
        X     = self.scaler.transform(X_raw) if self.scaler else X_raw

        if ml_level == 1:
            result = self._predict_anomaly(X, flow)
        elif ml_level == 2:
            result = self._predict_hybrid(X, flow)
        elif ml_level == 3:
            result = self._predict_classify(X, flow)
        else:
            result = self._predict_anomaly(X, flow)

        result.latency_ms = (time.perf_counter() - t0) * 1000
        return result

    # ── Level 1: Anomaly ───────────────────────────────────────────────────

    def _predict_anomaly(self, X: np.ndarray, flow: FlowRecord) -> PredictionResult:
        votes      = []
        scores     = {}
        iforest_sc = 0.0

        if self.iforest:
            sc = float(self.iforest.score_samples(X)[0])
            iforest_sc = sc
            scores["iforest"] = sc
            votes.append(-1 if sc < settings.anomaly_threshold else 1)

        if self.ocsvm:
            pred = int(self.ocsvm.predict(X)[0])
            scores["ocsvm"] = pred
            votes.append(pred)

        if self.autoencoder:
            pred_ae = int(self.autoencoder.predict(X)[0])
            err     = float(self.autoencoder.score(X)[0])
            scores["autoencoder"] = err
            votes.append(pred_ae)

        if not votes:
            return self._no_model_result()

        anomaly_votes = votes.count(-1)
        is_anomaly    = anomaly_votes > len(votes) / 2
        confidence    = anomaly_votes / len(votes)

        raw = {"votes": votes, "scores": scores, "n_models": len(votes)}

        if is_anomaly:
            return PredictionResult(
                is_threat=True,
                attack_type="Anomaly",
                severity="MEDIUM" if confidence < 0.8 else "HIGH",
                confidence=confidence,
                ml_level=1,
                ml_model="IForest+OCSVM+AE",
                anomaly_score=iforest_sc,
                raw=raw,
            )
        return PredictionResult(
            is_threat=False, attack_type=None, severity=None,
            confidence=1 - confidence, ml_level=1,
            ml_model="IForest+OCSVM+AE",
            anomaly_score=iforest_sc, raw=raw,
        )

    # ── Level 2: Hybrid ────────────────────────────────────────────────────

    def _predict_hybrid(self, X: np.ndarray, flow: FlowRecord) -> PredictionResult:
        # Gate: run anomaly first
        anomaly_res = self._predict_anomaly(X, flow)
        if not anomaly_res.is_threat:
            anomaly_res.ml_level = 2
            return anomaly_res
        # Confirm with classifier
        clf_res = self._predict_classify(X, flow)
        clf_res.ml_level = 2
        clf_res.ml_model = "Hybrid(AnomalyGate+XGB)"
        clf_res.anomaly_score = anomaly_res.anomaly_score
        return clf_res

    # ── Level 3: Classification ────────────────────────────────────────────

    def _predict_classify(self, X: np.ndarray, flow: FlowRecord) -> PredictionResult:
        iforest_sc = 0.0
        if self.iforest:
            iforest_sc = float(self.iforest.score_samples(X)[0])

        if not self.classifier or not self.label_encoder:
            return self._no_model_result()

        proba = self.classifier.predict_proba(X)[0]
        cls_idx  = int(np.argmax(proba))
        conf     = float(proba[cls_idx])
        cls_name = self.label_encoder.classes_[cls_idx]

        raw = {
            "classes": list(self.label_encoder.classes_),
            "probabilities": proba.tolist(),
        }

        is_benign = cls_name.upper() in ("BENIGN", "NORMAL")
        if is_benign or conf < settings.classifier_confidence:
            return PredictionResult(
                is_threat=False, attack_type=None, severity=None,
                confidence=conf, ml_level=3, ml_model="XGBoost",
                anomaly_score=iforest_sc, raw=raw,
            )

        severity = SEVERITY_MAP.get(cls_name, "MEDIUM")
        display  = ATTACK_DISPLAY.get(cls_name, cls_name)
        return PredictionResult(
            is_threat=True,
            attack_type=display,
            severity=severity,
            confidence=conf,
            ml_level=3,
            ml_model="XGBoost",
            anomaly_score=iforest_sc,
            raw=raw,
        )

    # ── Helpers ────────────────────────────────────────────────────────────

    def _no_model_result(self) -> PredictionResult:
        return PredictionResult(
            is_threat=False, attack_type=None, severity=None,
            confidence=0.0, ml_level=4, ml_model="None",
            anomaly_score=0.0, raw={"error": "no models loaded"},
        )


# Singleton instance
ml_engine = MLEngine()