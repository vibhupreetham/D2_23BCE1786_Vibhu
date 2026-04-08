import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger("lstm")

SEQ_LEN = 10

FEATURE_NAMES = [
    "duration_ms","packet_count","byte_count","avg_pkt_size",
    "pkt_rate","byte_rate","iat_mean","iat_std","iat_min","iat_max",
    "syn_count","ack_count","fin_count","rst_count","psh_count","urg_count",
    "src_port","dst_port","protocol_tcp","protocol_udp","protocol_icmp",
    "pkt_ratio","byte_per_pkt"
]

# ─────────────────────────────────────────────
# LOAD DATA (🔥 FIXED)
# ─────────────────────────────────────────────

def load_dataset(path):
    df = pd.read_csv(path)

    # ensure columns exist
    for col in FEATURE_NAMES:
        if col not in df.columns:
            df[col] = 0

    # 🔥 HARD CLEANING (VERY IMPORTANT)
    df[FEATURE_NAMES] = (
        df[FEATURE_NAMES]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    # 🔥 FORCE NON-NEGATIVE (FIX FOR LOG ERROR)
    df[FEATURE_NAMES] = df[FEATURE_NAMES].clip(lower=0)

    # 🔥 SAFE LOG TRANSFORM
    for col in ["pkt_rate","byte_rate","byte_count"]:
        df[col] = np.log1p(df[col])

    # final safety check
    df[FEATURE_NAMES] = df[FEATURE_NAMES].replace([np.inf, -np.inf], 0).fillna(0)

    X = df[FEATURE_NAMES].values.astype(np.float32)
    y = df["label"].astype(str).values

    return X, y

# ─────────────────────────────────────────────
# BALANCE
# ─────────────────────────────────────────────

def balance_data(X, y, max_per_class=40000):
    X_out, y_out = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        if len(idx) > max_per_class:
            idx = np.random.choice(idx, max_per_class, replace=False)
        X_out.append(X[idx])
        y_out.append(y[idx])
    return np.vstack(X_out), np.concatenate(y_out)

# ─────────────────────────────────────────────
# SEQUENCES
# ─────────────────────────────────────────────

def make_sequences(X, y, seq_len=SEQ_LEN):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        seq = X[i:i+seq_len]
        labels = y[i:i+seq_len]

        vals, counts = np.unique(labels, return_counts=True)
        label = vals[np.argmax(counts)]

        Xs.append(seq)
        ys.append(label)

    return np.array(Xs), np.array(ys)

# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────

def build_model(input_size, num_classes):
    import torch
    import torch.nn as nn

    class LSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(input_size, 64, 2, batch_first=True, dropout=0.3)
            self.fc = nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, num_classes)
            )

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])

    return LSTM()

# ─────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────

def train(args):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report

    out = Path(args.out)
    out.mkdir(exist_ok=True)

    X, y = load_dataset(args.dataset)

    X, y = balance_data(X, y)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_seq, y_seq = make_sequences(X, y_enc)
    logger.info(f"Sequences: {X_seq.shape}")

    X_tr, X_te, y_tr, y_te = train_test_split(X_seq, y_seq, test_size=0.2)

    tr_ds = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr))
    te_ds = TensorDataset(torch.tensor(X_te), torch.tensor(y_te))

    tr_dl = DataLoader(tr_ds, batch_size=64, shuffle=True)
    te_dl = DataLoader(te_ds, batch_size=128)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(X_seq.shape[2], len(le.classes_)).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for ep in range(args.epochs):
        model.train()
        total = 0

        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)

            logits = model(xb)
            loss = loss_fn(logits, yb)

            opt.zero_grad()
            loss.backward()

            # 🔥 gradient clipping (prevents NaN)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

            opt.step()
            total += loss.item()

        logger.info(f"Epoch {ep+1}: {total/len(tr_dl):.4f}")

    # evaluation
    model.eval()
    preds, true = [], []

    with torch.no_grad():
        for xb, yb in te_dl:
            p = model(xb.to(device)).argmax(1).cpu().numpy()
            preds.extend(p)
            true.extend(yb.numpy())

    logger.info("\n" + classification_report(true, preds, target_names=le.classes_))

    torch.save(model.state_dict(), out / "lstm_classifier.pt")

    cfg = {"classes": list(le.classes_), "seq_len": SEQ_LEN}
    (out / "lstm_config.json").write_text(json.dumps(cfg, indent=2))

    logger.info("✅ LSTM model saved")

# ─────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default="models")
    ap.add_argument("--epochs", type=int, default=20)

    train(ap.parse_args())