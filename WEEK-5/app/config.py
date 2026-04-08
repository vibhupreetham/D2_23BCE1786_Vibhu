"""
app/config.py — centralised settings loaded from .env
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path


class Settings(BaseSettings):
    # Server
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_debug: bool = False
    secret_key: str = "changeme"

    # Database
    database_url: str = "sqlite+aiosqlite:///./netguard.db"

    # Capture
    capture_interface: str = "eth0"
    capture_filter: str = ""
    capture_timeout: int = 1
    flow_window_seconds: int = 5

    # Model paths
    model_dir: Path = Path("models")
    anomaly_model: Path = Path("models/anomaly_iforest.pkl")
    svm_model: Path = Path("models/anomaly_ocsvm.pkl")
    autoencoder_model: Path = Path("models/autoencoder.pt")
    classifier_model: Path = Path("models/classifier_xgb.pkl")
    label_encoder: Path = Path("models/label_encoder.pkl")
    scaler: Path = Path("models/scaler.pkl")

    # Thresholds
    anomaly_threshold: float = -0.1
    autoencoder_threshold: float = 0.05
    classifier_confidence: float = 0.70

    # Email alerts
    alert_email_enabled: bool = False
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_pass: str = ""
    alert_recipient: str = ""
    alert_severity_threshold: str = "HIGH"

    # Auto-block
    autoblock_enabled: bool = False
    autoblock_severity: str = "CRITICAL"
    autoblock_backend: str = "iptables"

    # GeoIP
    geoip_db_path: Path = Path("data/GeoLite2-City.mmdb")

    # WebSocket
    ws_broadcast_interval: float = 1.0

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()