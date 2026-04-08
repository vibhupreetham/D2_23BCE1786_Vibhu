import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("preprocess")

OUTPUT_FEATURES = [
    "duration_ms","packet_count","byte_count","avg_pkt_size",
    "pkt_rate","byte_rate","iat_mean","iat_std","iat_min","iat_max",
    "syn_count","ack_count","fin_count","rst_count","psh_count","urg_count",
    "src_port","dst_port","protocol_tcp","protocol_udp","protocol_icmp",
    "pkt_ratio","byte_per_pkt","label"
]

LABEL_MAP = {
    "BENIGN":"BENIGN","Normal":"BENIGN",
    "PortScan":"PortScan","Reconnaissance":"PortScan",
    "DDoS":"DDoS","DoS":"DDoS","DoS Hulk":"DDoS",
    "DoS GoldenEye":"DDoS","DoS slowloris":"DDoS","DoS Slowhttptest":"DDoS",
    "FTP-Patator":"BruteForce","SSH-Patator":"BruteForce",
    "Bot":"MalwareBeacon","Backdoor":"MalwareBeacon","Worms":"MalwareBeacon",
    "Fuzzers":"PacketAnomaly","Generic":"PacketAnomaly","Analysis":"PacketAnomaly",
    "Web Attack - XSS":"PacketAnomaly",
    "Web Attack - Sql Injection":"PacketAnomaly",
    "Web Attack - Brute Force":"BruteForce",
    "Exploits":"ZeroDay","Shellcode":"ZeroDay","Infiltration":"ZeroDay",
}

def safe_read_csv(path):
    for enc in ["utf-8","latin-1","cp1252"]:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except:
            continue
    raise ValueError(f"Cannot read {path}")

def load_csvs(path):
    csvs = sorted(Path(path).glob("*.csv"))
    csvs = [f for f in csvs if "features" not in f.name.lower() and "list" not in f.name.lower()]

    frames = []
    for f in csvs:
        try:
            frames.append(safe_read_csv(f))
        except Exception as e:
            log.warning(f"Skipping {f.name}: {e}")

    df = pd.concat(frames, ignore_index=True)
    df.columns = df.columns.str.strip()
    return df

def encode_protocol(series):
    if np.issubdtype(series.dtype, np.number):
        p = series.fillna(0).astype(int)
        return pd.DataFrame({
            "protocol_tcp":(p==6).astype(int),
            "protocol_udp":(p==17).astype(int),
            "protocol_icmp":(p==1).astype(int)
        })
    p = series.astype(str).str.lower()
    return pd.DataFrame({
        "protocol_tcp":(p=="tcp").astype(int),
        "protocol_udp":(p=="udp").astype(int),
        "protocol_icmp":(p=="icmp").astype(int)
    })

def clean(df):
    df = df.replace([np.inf,-np.inf],0).fillna(0)
    return df

def process_cic(path):
    df = load_csvs(path)
    out = pd.DataFrame()

    out["duration_ms"]=pd.to_numeric(df.get("Flow Duration",0))/1000
    out["packet_count"]=df.get("Total Fwd Packets",0)
    out["byte_count"]=df.get("Total Length of Fwd Packets",0)

    out["avg_pkt_size"]=df.get("Average Packet Size",0)
    out["pkt_rate"]=df.get("Flow Packets/s",0)
    out["byte_rate"]=df.get("Flow Bytes/s",0)

    out["iat_mean"]=df.get("Flow IAT Mean",0)
    out["iat_std"]=df.get("Flow IAT Std",0)
    out["iat_min"]=df.get("Flow IAT Min",0)
    out["iat_max"]=df.get("Flow IAT Max",0)

    out["syn_count"]=df.get("SYN Flag Count",0)
    out["ack_count"]=df.get("ACK Flag Count",0)
    out["fin_count"]=0
    out["rst_count"]=0
    out["psh_count"]=0
    out["urg_count"]=0

    out["src_port"]=df.get("Source Port",0)
    out["dst_port"]=df.get("Destination Port",0)

    proto=df["Protocol"] if "Protocol" in df.columns else pd.Series([6]*len(df))
    out=pd.concat([out,encode_protocol(proto)],axis=1)

    labels=df.get("Label","BENIGN").astype(str).str.replace("�","-")
    out["label"]=labels.map(LABEL_MAP).fillna("PacketAnomaly")

    # 🔥 New features
    out["pkt_ratio"]=out["packet_count"]/(out["byte_count"]+1)
    out["byte_per_pkt"]=out["byte_count"]/(out["packet_count"]+1)

    return clean(out)

def process_unsw(path):
    df=load_csvs(path)
    out=pd.DataFrame()

    out["duration_ms"]=pd.to_numeric(df.get("dur",0))*1000
    out["packet_count"]=df.get("spkts",0)+df.get("dpkts",0)
    out["byte_count"]=df.get("sbytes",0)+df.get("dbytes",0)

    out["avg_pkt_size"]=out["byte_count"]/(out["packet_count"]+1)
    out["pkt_rate"]=df.get("rate",0)
    out["byte_rate"]=out["byte_count"]

    out["iat_mean"]=df.get("sinpkt",0)
    out["iat_std"]=df.get("sjit",0)
    out["iat_min"]=0
    out["iat_max"]=out["iat_mean"]+2*out["iat_std"]

    out["syn_count"]=0
    out["ack_count"]=0
    out["fin_count"]=0
    out["rst_count"]=0
    out["psh_count"]=0
    out["urg_count"]=0

    out["src_port"]=df.get("sport",0)
    out["dst_port"]=df.get("dsport",0)

    proto=df["proto"] if "proto" in df.columns else pd.Series(["tcp"]*len(df))
    out=pd.concat([out,encode_protocol(proto)],axis=1)

    labels=df.get("attack_cat","BENIGN").astype(str)
    out["label"]=labels.map(LABEL_MAP).fillna("PacketAnomaly")

    out["pkt_ratio"]=out["packet_count"]/(out["byte_count"]+1)
    out["byte_per_pkt"]=out["byte_count"]/(out["packet_count"]+1)

    return clean(out)

def balance(df,samples_per_class=80000):
    parts=[]
    for lbl in df["label"].unique():
        d=df[df["label"]==lbl]
        if len(d)>samples_per_class:
            d=d.sample(samples_per_class,random_state=42)
        parts.append(d)
    return shuffle(pd.concat(parts),random_state=42)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--cic")
    ap.add_argument("--unsw")
    ap.add_argument("--output",default="data/combined.csv")
    args=ap.parse_args()

    cic=process_cic(args.cic) if args.cic else None
    unsw=process_unsw(args.unsw) if args.unsw else None

    df=pd.concat([cic,unsw],ignore_index=True)
    df=balance(df)

    df.to_csv(args.output,index=False)
    log.info(f"Saved → {args.output}")

if __name__=="__main__":
    main()