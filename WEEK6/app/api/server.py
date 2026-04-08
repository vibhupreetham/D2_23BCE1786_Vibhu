from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from app.capture.sniffer import PacketSniffer
from app.ml.ensemble_model import EnsembleIDS

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

queue = asyncio.Queue()
ids = None
latest_alerts = []
MAX_ALERTS = 100

@app.on_event("startup")
async def startup_event():
    global ids
    print("🚀 Starting IDS backend...")

    ids = EnsembleIDS("models")

    loop = asyncio.get_running_loop()
    sniffer = PacketSniffer(queue, interface="en0")
    sniffer.start(loop)

    async def process():
        while True:
            flow = await queue.get()
            try:
                features = flow.to_feature_vector()
                label, conf = ids.predict(flow.src_ip, features)
                alert = {
                    "src_ip": flow.src_ip,
                    "dst_ip": flow.dst_ip,
                    "label": label,
                    "confidence": float(conf),
                }
                latest_alerts.insert(0, alert)
                if len(latest_alerts) > MAX_ALERTS:
                    latest_alerts.pop()
            except Exception as e:
                print("❌ Processing error:", e)

    asyncio.create_task(process())

@app.get("/alerts")
def get_alerts():
    return latest_alerts

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": ids is not None}