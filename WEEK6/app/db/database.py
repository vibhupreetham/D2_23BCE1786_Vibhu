"""
app/db/database.py — async SQLAlchemy engine, session, and ORM models
"""
from __future__ import annotations
import datetime
from sqlalchemy import (
    Column, Integer, String, Float, Boolean,
    DateTime, Text, Index, event
)
from sqlalchemy.ext.asyncio import (
    create_async_engine, AsyncSession, async_sessionmaker
)
from sqlalchemy.orm import DeclarativeBase
from app.config import settings


# ── Engine ─────────────────────────────────────────────────────────────────
engine = create_async_engine(
    settings.database_url,
    echo=settings.app_debug,
    connect_args={"check_same_thread": False}   # SQLite only
    if "sqlite" in settings.database_url else {},
)

AsyncSessionLocal = async_sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)


# ── Base ───────────────────────────────────────────────────────────────────
class Base(DeclarativeBase):
    pass


# ── ORM Models ─────────────────────────────────────────────────────────────

class TrafficLog(Base):
    """Raw packet/flow record captured from the network."""
    __tablename__ = "traffic_logs"

    id           = Column(Integer, primary_key=True, index=True)
    timestamp    = Column(DateTime, default=datetime.datetime.utcnow, index=True)
    src_ip       = Column(String(45), nullable=False, index=True)
    dst_ip       = Column(String(45), nullable=False)
    src_port     = Column(Integer)
    dst_port     = Column(Integer, index=True)
    protocol     = Column(String(10))
    packet_count = Column(Integer, default=1)
    byte_count   = Column(Integer, default=0)
    duration_ms  = Column(Float, default=0.0)
    # TCP flags
    syn_count    = Column(Integer, default=0)
    ack_count    = Column(Integer, default=0)
    fin_count    = Column(Integer, default=0)
    rst_count    = Column(Integer, default=0)
    # Flow-level features
    avg_pkt_size = Column(Float, default=0.0)
    pkt_rate     = Column(Float, default=0.0)
    byte_rate    = Column(Float, default=0.0)
    iat_mean     = Column(Float, default=0.0)   # inter-arrival time mean
    iat_std      = Column(Float, default=0.0)
    # Raw feature vector (JSON string)
    feature_vec  = Column(Text)

    __table_args__ = (
        Index("ix_traffic_src_dst", "src_ip", "dst_ip"),
    )


class ThreatAlert(Base):
    """Detected threat record with ML prediction details."""
    __tablename__ = "threat_alerts"

    id              = Column(Integer, primary_key=True, index=True)
    timestamp       = Column(DateTime, default=datetime.datetime.utcnow, index=True)
    traffic_log_id  = Column(Integer, index=True)   # FK to TrafficLog
    src_ip          = Column(String(45), nullable=False, index=True)
    dst_ip          = Column(String(45))
    src_port        = Column(Integer)
    dst_port        = Column(Integer)
    protocol        = Column(String(10))
    attack_type     = Column(String(64), index=True)
    severity        = Column(String(16), index=True)   # LOW/MEDIUM/HIGH/CRITICAL
    confidence      = Column(Float)                    # 0.0 – 1.0
    ml_level        = Column(Integer)                  # 1-4
    ml_model        = Column(String(64))
    anomaly_score   = Column(Float)
    # Actions taken
    is_blocked      = Column(Boolean, default=False)
    email_sent      = Column(Boolean, default=False)
    # GeoIP
    country_code    = Column(String(4))
    country_name    = Column(String(64))
    city            = Column(String(64))
    latitude        = Column(Float)
    longitude       = Column(Float)
    # Extra
    raw_prediction  = Column(Text)    # JSON of full model output

    __table_args__ = (
        Index("ix_alert_sev_ts", "severity", "timestamp"),
    )


class ModelPrediction(Base):
    """Stores every model inference for audit / retraining."""
    __tablename__ = "model_predictions"

    id             = Column(Integer, primary_key=True, index=True)
    timestamp      = Column(DateTime, default=datetime.datetime.utcnow)
    traffic_log_id = Column(Integer)
    model_name     = Column(String(64))
    input_features = Column(Text)    # JSON
    raw_output     = Column(Text)    # JSON
    predicted_class= Column(String(64))
    confidence     = Column(Float)
    latency_ms     = Column(Float)


class BlockedIP(Base):
    """IPs that have been auto-blocked by the system."""
    __tablename__ = "blocked_ips"

    id           = Column(Integer, primary_key=True, index=True)
    ip_address   = Column(String(45), unique=True, nullable=False, index=True)
    reason       = Column(String(128))
    blocked_at   = Column(DateTime, default=datetime.datetime.utcnow)
    alert_id     = Column(Integer)
    is_active    = Column(Boolean, default=True)
    unblocked_at = Column(DateTime, nullable=True)


# ── Helpers ────────────────────────────────────────────────────────────────

async def init_db():
    """Create all tables on startup."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_session() -> AsyncSession:
    """FastAPI dependency — yields an async DB session."""
    async with AsyncSessionLocal() as session:
        yield session