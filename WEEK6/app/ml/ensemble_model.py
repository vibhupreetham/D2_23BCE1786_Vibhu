import numpy as np
import joblib
import torch
import json
from pathlib import Path

FEATURE_COUNT = 23

class EnsembleIDS:
    def __init__(self, model_dir="models"):
        model_dir = Path(model_dir)

        print("🔄 Loading models...")

        self.xgb = joblib.load(model_dir / "classifier_xgb.pkl")
        self.scaler = joblib.load(model_dir / "scaler.pkl")
        self.le = joblib.load(model_dir / "label_encoder.pkl")

        try:
            from app.ml.lstm_model import build_model
        except ImportError:
            from lstm_model import build_model

        with open(model_dir / "lstm_config.json") as f:
            cfg = json.load(f)

        self.classes = cfg["classes"]
        self.seq_len = cfg["seq_len"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.lstm = build_model(FEATURE_COUNT, len(self.classes))
        self.lstm.load_state_dict(torch.load(model_dir / "lstm_classifier.pt", map_location="cpu"))
        self.lstm.eval().to(self.device)

        self.buffers = {}

        print("✅ Models loaded successfully!")

    def preprocess(self, features):
        x = np.array(features, dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = np.clip(x, 0, 1e9)
        x[4] = np.log1p(x[4])
        x[5] = np.log1p(x[5])
        x[2] = np.log1p(x[2])
        return self.scaler.transform([x])[0]

    def update_sequence(self, src_ip, features):
        if src_ip not in self.buffers:
            self.buffers[src_ip] = []
        self.buffers[src_ip].append(features)
        if len(self.buffers[src_ip]) > self.seq_len:
            self.buffers[src_ip].pop(0)

    def get_sequence(self, src_ip):
        seq = self.buffers.get(src_ip, [])
        if len(seq) == 0:
            return None
        if len(seq) < self.seq_len:
            pad = [np.zeros_like(seq[0]) for _ in range(self.seq_len - len(seq))]
            seq = pad + seq
        return np.array(seq, dtype=np.float32)

    def predict(self, src_ip, raw_features):
        x = self.preprocess(raw_features)

        xgb_probs = self.xgb.predict_proba([x])[0]

        self.update_sequence(src_ip, x)
        seq = self.get_sequence(src_ip)

        if seq is None:
            idx = np.argmax(xgb_probs)
            return self.le.inverse_transform([idx])[0], float(xgb_probs[idx])

        with torch.no_grad():
            tensor = torch.tensor(seq).unsqueeze(0).to(self.device)
            lstm_logits = self.lstm(tensor)
            lstm_probs = torch.softmax(lstm_logits, dim=-1).cpu().numpy()[0]

        final_probs = 0.6 * xgb_probs + 0.4 * lstm_probs

        idx = np.argmax(final_probs)
        label = self.le.inverse_transform([idx])[0]
        confidence = float(final_probs[idx])

        return label, confidence


if __name__ == "__main__":
    print("🚀 Starting Ensemble IDS Test...")

    ids = EnsembleIDS("models")

    features = [
        1000, 10, 5000, 500,
        20, 10000, 50, 10, 5, 100,
        1, 1, 0, 0, 0, 0,
        12345, 80,
        1, 0, 0,
        0.002, 400
    ]

    label, conf = ids.predict("192.168.1.10", features)

    print("\n✅ RESULT")
    print("Prediction:", label)
    print("Confidence:", conf)