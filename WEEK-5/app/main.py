"""
app/main.py
────────────────────────────────────────────────────────────────────────────
FastAPI application entry-point.

REST endpoints
  POST /capture/start          start packet capture
  POST /capture/stop           stop packet capture
  POST /capture/replay         replay a PCAP file
  GET  /predict                run ML on a manually supplied feature vector
  GET  /alerts                 paginated threat alert history
  GET  /alerts/{id}            single alert detail
  GET  /logs                   paginated traffic log history
  GET  /stats                  live statistics
  POST /settings/ml-level      change detection level (1-4)
  GET  /blocked                list blocked IPs
  POST /blocked/{ip}/unblock   remove a firewall rule
  GET  /health                 liveness probe

WebSocket
  WS  /ws                      real-time event stream (threats + stats)
"""

from __future__ import annotations
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import (
    FastAPI, WebSocket, WebSocketDisconnect,
    Depends, HTTPException, BackgroundTasks, Query
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import select, desc, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.database import (
    init_db, get_session,
    TrafficLog, ThreatAlert, ModelPrediction, BlockedIP
)
from app.ml.engine import ml_engine
from app.ml.pipeline import pipeline
from app.capture.sniffer import PacketSniffer, PcapReplay, FlowRecord

logger = logging.getLogger("netguard.api")

# ── Capture state ──────────────────────────────────────────────────────────
_sniffer: Optional[PacketSniffer] = None
_pipeline_task: Optional[asyncio.Task] = None


# ── Lifespan ───────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline_task
    # Init DB
    await init_db()
    logger.info("Database initialised.")

    # Load ML models
    ml_engine.load()

    # Connect pipeline WS clients set
    # (pipeline uses pipeline._ws_clients directly)

    # Start inference pipeline
    _pipeline_task = asyncio.create_task(pipeline.start())
    logger.info("Application startup complete.")

    yield

    # Shutdown
    if _sniffer:
        _sniffer.stop()
    pipeline.stop()
    if _pipeline_task:
        _pipeline_task.cancel()
    logger.info("Application shutdown.")


# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="NetGuard IDS API",
    description="Real-Time Network Threat Detection System — ML-powered IDS",
    version="2.4.1",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic schemas ───────────────────────────────────────────────────────

class CaptureStartRequest(BaseModel):
    interface: str = settings.capture_interface
    bpf_filter: str = settings.capture_filter

class CaptureReplayRequest(BaseModel):
    pcap_path: str
    speed: float = 1.0

class MLLevelRequest(BaseModel):
    level: int   # 1-4

class PredictRequest(BaseModel):
    features: List[float]  # 21-element feature vector

class AlertResponse(BaseModel):
    id: int
    timestamp: str
    src_ip: str
    dst_ip: Optional[str]
    src_port: Optional[int]
    dst_port: Optional[int]
    protocol: Optional[str]
    attack_type: Optional[str]
    severity: Optional[str]
    confidence: float
    ml_level: int
    ml_model: Optional[str]
    anomaly_score: Optional[float]
    is_blocked: bool
    country_code: Optional[str]
    country_name: Optional[str]

    class Config:
        from_attributes = True


# ── Capture endpoints ──────────────────────────────────────────────────────

@app.post("/capture/start", tags=["Capture"])
async def start_capture(req: CaptureStartRequest):
    global _sniffer
    if _sniffer:
        return {"status": "already_running"}
    loop = asyncio.get_event_loop()
    _sniffer = PacketSniffer(
        queue=pipeline.flow_queue,
        interface=req.interface,
        bpf_filter=req.bpf_filter,
        window_secs=settings.flow_window_seconds,
    )
    _sniffer.start(loop)
    return {"status": "started", "interface": req.interface, "filter": req.bpf_filter}


@app.post("/capture/stop", tags=["Capture"])
async def stop_capture():
    global _sniffer
    if not _sniffer:
        return {"status": "not_running"}
    _sniffer.stop()
    _sniffer = None
    return {"status": "stopped"}


@app.post("/capture/replay", tags=["Capture"])
async def replay_pcap(req: CaptureReplayRequest, background_tasks: BackgroundTasks):
    replayer = PcapReplay(req.pcap_path, pipeline.flow_queue)
    background_tasks.add_task(replayer.replay, req.speed)
    return {"status": "replaying", "path": req.pcap_path, "speed": req.speed}


@app.get("/capture/status", tags=["Capture"])
async def capture_status():
    return {
        "running": _sniffer is not None,
        "interface": _sniffer.interface if _sniffer else None,
        "queue_depth": pipeline.flow_queue.qsize(),
    }


# ── Predict endpoint ───────────────────────────────────────────────────────

@app.post("/predict", tags=["ML"])
async def predict(req: PredictRequest):
    """Run ML inference on a manually supplied 21-element feature vector."""
    if len(req.features) != 21:
        raise HTTPException(400, "Feature vector must have exactly 21 elements.")
    # Build a synthetic FlowRecord from the feature vector
    f = req.features
    flow = FlowRecord(
        src_ip="0.0.0.0", dst_ip="0.0.0.0",
        src_port=int(f[16]), dst_port=int(f[17]),
        protocol="TCP" if f[18] else "UDP" if f[19] else "ICMP",
        duration_ms=f[0], packet_count=int(f[1]), byte_count=int(f[2]),
        avg_pkt_size=f[3], pkt_rate=f[4], byte_rate=f[5],
        iat_mean=f[6], iat_std=f[7], iat_min=f[8], iat_max=f[9],
        syn_count=int(f[10]), ack_count=int(f[11]), fin_count=int(f[12]),
        rst_count=int(f[13]), psh_count=int(f[14]), urg_count=int(f[15]),
    )
    result = await ml_engine.predict(flow, ml_level=pipeline._ml_level)
    return result.to_dict()


# ── Alerts endpoints ───────────────────────────────────────────────────────

@app.get("/alerts", tags=["Alerts"], response_model=List[AlertResponse])
async def get_alerts(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, le=500),
    severity: Optional[str] = None,
    attack_type: Optional[str] = None,
    db: AsyncSession = Depends(get_session),
):
    q = select(ThreatAlert).order_by(desc(ThreatAlert.timestamp))
    if severity:
        q = q.where(ThreatAlert.severity == severity.upper())
    if attack_type:
        q = q.where(ThreatAlert.attack_type.ilike(f"%{attack_type}%"))
    q = q.offset(skip).limit(limit)
    result = await db.execute(q)
    rows = result.scalars().all()
    return [_alert_to_schema(r) for r in rows]


@app.get("/alerts/{alert_id}", tags=["Alerts"])
async def get_alert(alert_id: int, db: AsyncSession = Depends(get_session)):
    result = await db.execute(
        select(ThreatAlert).where(ThreatAlert.id == alert_id)
    )
    row = result.scalar_one_or_none()
    if not row:
        raise HTTPException(404, "Alert not found")
    return _alert_to_schema(row)


# ── Logs endpoints ─────────────────────────────────────────────────────────

@app.get("/logs", tags=["Logs"])
async def get_logs(
    skip: int = 0, limit: int = 100,
    db: AsyncSession = Depends(get_session),
):
    result = await db.execute(
        select(TrafficLog).order_by(desc(TrafficLog.timestamp)).offset(skip).limit(limit)
    )
    rows = result.scalars().all()
    return [
        {
            "id": r.id,
            "timestamp": r.timestamp.isoformat(),
            "src_ip": r.src_ip, "dst_ip": r.dst_ip,
            "src_port": r.src_port, "dst_port": r.dst_port,
            "protocol": r.protocol,
            "packet_count": r.packet_count, "byte_count": r.byte_count,
            "duration_ms": r.duration_ms,
            "pkt_rate": r.pkt_rate, "byte_rate": r.byte_rate,
        }
        for r in rows
    ]


# ── Stats endpoint ─────────────────────────────────────────────────────────

@app.get("/stats", tags=["Stats"])
async def get_stats(db: AsyncSession = Depends(get_session)):
    sev_counts = {}
    for sev in ("LOW", "MEDIUM", "HIGH", "CRITICAL"):
        r = await db.execute(
            select(func.count()).where(ThreatAlert.severity == sev)
        )
        sev_counts[sev] = r.scalar()

    attack_counts_raw = await db.execute(
        select(ThreatAlert.attack_type, func.count())
        .group_by(ThreatAlert.attack_type)
        .order_by(desc(func.count()))
        .limit(10)
    )
    attack_counts = dict(attack_counts_raw.all())

    total_logs = (await db.execute(select(func.count(TrafficLog.id)))).scalar()
    total_alerts = (await db.execute(select(func.count(ThreatAlert.id)))).scalar()
    total_blocked = (await db.execute(
        select(func.count()).where(BlockedIP.is_active == True)
    )).scalar()

    return {
        "pipeline": pipeline.get_stats(),
        "ml_level": pipeline._ml_level,
        "total_logs": total_logs,
        "total_alerts": total_alerts,
        "total_blocked": total_blocked,
        "severity_breakdown": sev_counts,
        "attack_type_breakdown": attack_counts,
    }


# ── Settings endpoints ─────────────────────────────────────────────────────

@app.post("/settings/ml-level", tags=["Settings"])
async def set_ml_level(req: MLLevelRequest):
    if not 1 <= req.level <= 4:
        raise HTTPException(400, "ml_level must be 1-4")
    pipeline.set_ml_level(req.level)
    return {"ml_level": req.level}


# ── Blocked IPs ────────────────────────────────────────────────────────────

@app.get("/blocked", tags=["Firewall"])
async def list_blocked(db: AsyncSession = Depends(get_session)):
    result = await db.execute(
        select(BlockedIP).where(BlockedIP.is_active == True)
        .order_by(desc(BlockedIP.blocked_at))
    )
    rows = result.scalars().all()
    return [
        {
            "id": r.id, "ip_address": r.ip_address, "reason": r.reason,
            "blocked_at": r.blocked_at.isoformat(), "alert_id": r.alert_id,
        }
        for r in rows
    ]


@app.post("/blocked/{ip}/unblock", tags=["Firewall"])
async def unblock_ip(ip: str, db: AsyncSession = Depends(get_session)):
    import datetime
    result = await db.execute(
        select(BlockedIP).where(BlockedIP.ip_address == ip, BlockedIP.is_active == True)
    )
    row = result.scalar_one_or_none()
    if not row:
        raise HTTPException(404, f"IP {ip} not in blocked list")
    row.is_active = False
    row.unblocked_at = datetime.datetime.utcnow()
    await db.commit()
    # TODO: actually remove the iptables rule
    return {"status": "unblocked", "ip": ip}


# ── Health ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health():
    return {
        "status": "ok",
        "ml_engine_ready": ml_engine._loaded,
        "capture_running": _sniffer is not None,
        "queue_depth": pipeline.flow_queue.qsize(),
    }


# ── WebSocket ──────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    pipeline._ws_clients.add(ws)
    logger.info(f"WS client connected ({len(pipeline._ws_clients)} total)")
    try:
        while True:
            # keep-alive: receive pings from client
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        pipeline._ws_clients.discard(ws)
        logger.info(f"WS client disconnected ({len(pipeline._ws_clients)} remaining)")


# ── Helper ─────────────────────────────────────────────────────────────────

def _alert_to_schema(r: ThreatAlert) -> dict:
    return {
        "id":           r.id,
        "timestamp":    r.timestamp.isoformat(),
        "src_ip":       r.src_ip,
        "dst_ip":       r.dst_ip,
        "src_port":     r.src_port,
        "dst_port":     r.dst_port,
        "protocol":     r.protocol,
        "attack_type":  r.attack_type,
        "severity":     r.severity,
        "confidence":   round(r.confidence * 100, 1),
        "ml_level":     r.ml_level,
        "ml_model":     r.ml_model,
        "anomaly_score":r.anomaly_score,
        "is_blocked":   r.is_blocked,
        "country_code": r.country_code,
        "country_name": r.country_name,
    }