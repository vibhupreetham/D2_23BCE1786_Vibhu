import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("train")

# ─────────────────────────────────────────────
# FEATURES (must match preprocessing)
# ─────────────────────────────────────────────

FEATURE_NAMES = [
    "duration_ms", "packet_count", "byte_count", "avg_pkt_size",
    "pkt_rate", "byte_rate", "iat_mean", "iat_std", "iat_min", "iat_max",
    "syn_count", "ack_count", "fin_count", "rst_count", "psh_count", "urg_count",
    "src_port", "dst_port", "protocol_tcp", "protocol_udp", "protocol_icmp",
    "pkt_ratio", "byte_per_pkt"
]

# ─────────────────────────────────────────────
# LOAD DATA (FULL FIXED)
# ─────────────────────────────────────────────

def load_dataset(path):
    log.info(f"Loading dataset from {path} ...")
    df = pd.read_csv(path)

    # Ensure all features exist
    for col in FEATURE_NAMES:
        if col not in df.columns:
            log.warning(f"Missing column {col} → filling with 0")
            df[col] = 0.0

    df = df[FEATURE_NAMES + ["label"]]

    # Clean bad values
    df[FEATURE_NAMES] = (
        df[FEATURE_NAMES]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .clip(lower=0, upper=1e9)
    )

    # Log scaling (important)
    for col in ["pkt_rate", "byte_rate", "byte_count"]:
        if col in df.columns:
            df[col] = np.log1p(df[col])

    X = df[FEATURE_NAMES].values.astype(np.float32)
    y = df["label"].astype(str).values

    log.info(f"Loaded {len(X)} samples, {len(np.unique(y))} classes: {np.unique(y)}")
    return X, y

# ─────────────────────────────────────────────
# AUTOENCODER
# ─────────────────────────────────────────────

def build_autoencoder(n_features):
    import torch
    import torch.nn as nn

    class AE(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(n_features, 16),
                nn.ReLU(),
                nn.Linear(16, 8),
                nn.ReLU(),
                nn.Linear(8, 4),
            )
            self.decoder = nn.Sequential(
                nn.Linear(4, 8),
                nn.ReLU(),
                nn.Linear(8, 16),
                nn.ReLU(),
                nn.Linear(16, n_features),
            )

        def forward(self, x):
            return self.decoder(self.encoder(x))

    return AE()

def train_autoencoder(X):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    import torch.nn as nn

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Training Autoencoder on {device}...")

    model = build_autoencoder(X.shape[1]).to(device)

    ds = TensorDataset(torch.tensor(X, dtype=torch.float32))
    dl = DataLoader(ds, batch_size=256, shuffle=True)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(20):
        total = 0
        for (xb,) in dl:
            xb = xb.to(device)
            out = model(xb)
            loss = loss_fn(out, xb)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item()

        log.info(f"AE Epoch {epoch+1}: {total/len(dl):.6f}")

    return model

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default="models")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(exist_ok=True)

    # ─────────────────────────────────────────
    # LOAD
    # ─────────────────────────────────────────

    X, y = load_dataset(args.dataset)

    # ─────────────────────────────────────────
    # SCALER
    # ─────────────────────────────────────────

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, out / "scaler.pkl")

    # ─────────────────────────────────────────
    # LABEL ENCODER
    # ─────────────────────────────────────────

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    joblib.dump(le, out / "label_encoder.pkl")

    # ─────────────────────────────────────────
    # ANOMALY MODELS
    # ─────────────────────────────────────────

    benign_mask = (y == "BENIGN")
    X_benign = X_scaled[benign_mask]

    log.info("Training Isolation Forest...")
    iforest = IsolationForest(n_estimators=200, contamination=0.05, n_jobs=-1)
    iforest.fit(X_benign)
    joblib.dump(iforest, out / "anomaly_iforest.pkl")

    log.info("Training One-Class SVM...")
    ocsvm = OneClassSVM(kernel="rbf", nu=0.05)
    ocsvm.fit(X_benign[:5000])
    joblib.dump(ocsvm, out / "anomaly_ocsvm.pkl")

    # ─────────────────────────────────────────
    # CLASSIFIER (FINAL)
    # ─────────────────────────────────────────

    import xgboost as xgb

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_scaled, y_enc, test_size=0.2, stratify=y_enc, random_state=42
    )

    # Class weights
    classes = np.unique(y_enc)
    weights = compute_class_weight("balanced", classes=classes, y=y_enc)
    class_weights = {i: w for i, w in enumerate(weights)}

    sample_weights = np.array([class_weights[y] for y in y_tr])

    clf = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=10,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=3,
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )

    log.info("Training XGBoost...")
    clf.fit(X_tr, y_tr, sample_weight=sample_weights)

    y_pred = clf.predict(X_te)

    log.info("\n" + classification_report(
        y_te, y_pred,
        target_names=le.classes_,
        zero_division=0
    ))

    joblib.dump(clf, out / "classifier_xgb.pkl")

    # ─────────────────────────────────────────
    # AUTOENCODER
    # ─────────────────────────────────────────

    log.info("Training Autoencoder...")
    model = train_autoencoder(X_benign[:5000])

    import torch
    torch.save(model.state_dict(), out / "autoencoder.pt")

    log.info("\n✅ All models trained successfully!")

# ─────────────────────────────────────────────

if __name__ == "__main__":
    main()