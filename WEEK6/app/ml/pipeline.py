"""
app/ml/pipeline.py
────────────────────────────────────────────────────────────────────────────
The inference pipeline glues every subsystem together:

  PacketSniffer → FlowQueue → MLEngine → DB → Alerts → WebSocket broadcast

It runs as a long-lived asyncio task started at application startup.
"""

from __future__ import annotations
import asyncio
import json
import logging
import time
from typing import Optional

from app.capture.sniffer import FlowRecord
from app.ml.engine import ml_engine, PredictionResult
from app.config import settings
from app.db.database import AsyncSessionLocal, TrafficLog, ThreatAlert, ModelPrediction

logger = logging.getLogger("netguard.pipeline")


class InferencePipeline:
    """
    Consumes FlowRecords from an asyncio.Queue, runs ML inference,
    persists results, fires alerts, and publishes to WebSocket clients.
    """

    def __init__(self):
        self.flow_queue: asyncio.Queue[FlowRecord] = asyncio.Queue(maxsize=2000)
        self._running   = False
        self._ml_level  = 1          # can be changed at runtime via API
        self._ws_clients: set        = set()   # ConnectionManager injects this
        self._stats     = {
            "packets_processed": 0,
            "threats_detected":  0,
            "ips_blocked":       0,
            "flows_per_sec":     0.0,
        }
        self._flow_count_window = []

    # ── Lifecycle ──────────────────────────────────────────────────────────

    async def start(self):
        self._running = True
        logger.info("Inference pipeline started.")
        await asyncio.gather(
            self._consume_flows(),
            self._stats_ticker(),
        )

    def stop(self):
        self._running = False

    # ── Consumer ───────────────────────────────────────────────────────────

    async def _consume_flows(self):
        while self._running:
            try:
                flow: FlowRecord = await asyncio.wait_for(
                    self.flow_queue.get(), timeout=1.0
                )
            except asyncio.TimeoutError:
                continue

            try:
                await self._process_flow(flow)
            except Exception as exc:
                logger.error(f"Pipeline error: {exc}", exc_info=True)
            finally:
                self.flow_queue.task_done()
                self._flow_count_window.append(time.time())

    async def _process_flow(self, flow: FlowRecord):
        self._stats["packets_processed"] += flow.packet_count

        # ── 1. Persist raw traffic log ────────────────────────────────────
        log_id = await self._save_traffic_log(flow)

        # ── 2. ML Inference ───────────────────────────────────────────────
        t0     = time.perf_counter()
        result = await ml_engine.predict(flow, ml_level=self._ml_level)
        latency_ms = (time.perf_counter() - t0) * 1000

        # ── 3. Persist prediction ─────────────────────────────────────────
        await self._save_prediction(log_id, flow, result, latency_ms)

        # ── 4. Handle threats ─────────────────────────────────────────────
        if result.is_threat:
            self._stats["threats_detected"] += 1
            alert_id = await self._save_alert(log_id, flow, result)
            event    = self._build_ws_event(flow, result, alert_id)

            # Auto-block
            if (
                settings.autoblock_enabled
                and result.severity in ("HIGH", "CRITICAL")
                and result.severity >= settings.autoblock_severity
            ):
                blocked = await self._block_ip(flow.src_ip, alert_id)
                if blocked:
                    self._stats["ips_blocked"] += 1
                    event["blocked"] = True

            # Email alert
            if (
                settings.alert_email_enabled
                and result.severity in self._email_severities()
            ):
                asyncio.create_task(self._send_email_alert(flow, result))

            # Broadcast to WebSocket clients
            await self._broadcast(event)

    # ── DB helpers ─────────────────────────────────────────────────────────

    async def _save_traffic_log(self, flow: FlowRecord) -> int:
        async with AsyncSessionLocal() as session:
            record = TrafficLog(
                src_ip=flow.src_ip,
                dst_ip=flow.dst_ip,
                src_port=flow.src_port,
                dst_port=flow.dst_port,
                protocol=flow.protocol,
                packet_count=flow.packet_count,
                byte_count=flow.byte_count,
                duration_ms=flow.duration_ms,
                syn_count=flow.syn_count,
                ack_count=flow.ack_count,
                fin_count=flow.fin_count,
                rst_count=flow.rst_count,
                avg_pkt_size=flow.avg_pkt_size,
                pkt_rate=flow.pkt_rate,
                byte_rate=flow.byte_rate,
                iat_mean=flow.iat_mean,
                iat_std=flow.iat_std,
                feature_vec=json.dumps(flow.to_feature_vector()),
            )
            session.add(record)
            await session.commit()
            await session.refresh(record)
            return record.id

    async def _save_prediction(
        self, log_id: int, flow: FlowRecord,
        result: PredictionResult, latency_ms: float
    ):
        async with AsyncSessionLocal() as session:
            pred = ModelPrediction(
                traffic_log_id=log_id,
                model_name=result.ml_model,
                input_features=json.dumps(flow.to_feature_vector()),
                raw_output=json.dumps(result.raw),
                predicted_class=result.attack_type or "BENIGN",
                confidence=result.confidence,
                latency_ms=latency_ms,
            )
            session.add(pred)
            await session.commit()

    async def _save_alert(
        self, log_id: int, flow: FlowRecord, result: PredictionResult
    ) -> int:
        geo = await self._geoip(flow.src_ip)
        async with AsyncSessionLocal() as session:
            alert = ThreatAlert(
                traffic_log_id=log_id,
                src_ip=flow.src_ip,
                dst_ip=flow.dst_ip,
                src_port=flow.src_port,
                dst_port=flow.dst_port,
                protocol=flow.protocol,
                attack_type=result.attack_type,
                severity=result.severity,
                confidence=result.confidence,
                ml_level=result.ml_level,
                ml_model=result.ml_model,
                anomaly_score=result.anomaly_score,
                raw_prediction=json.dumps(result.raw),
                country_code=geo.get("country_code"),
                country_name=geo.get("country_name"),
                city=geo.get("city"),
                latitude=geo.get("lat"),
                longitude=geo.get("lon"),
            )
            session.add(alert)
            await session.commit()
            await session.refresh(alert)
            return alert.id

    # ── GeoIP ──────────────────────────────────────────────────────────────

    async def _geoip(self, ip: str) -> dict:
        try:
            import geoip2.database
            with geoip2.database.Reader(str(settings.geoip_db_path)) as reader:
                resp = reader.city(ip)
                return {
                    "country_code": resp.country.iso_code,
                    "country_name": resp.country.name,
                    "city":         resp.city.name,
                    "lat":          resp.location.latitude,
                    "lon":          resp.location.longitude,
                }
        except Exception:
            return {}

    # ── Firewall ───────────────────────────────────────────────────────────

    async def _block_ip(self, ip: str, alert_id: int) -> bool:
        import subprocess, sys
        try:
            if settings.autoblock_backend == "iptables":
                cmd = ["iptables", "-A", "INPUT", "-s", ip, "-j", "DROP"]
            else:  # windows_firewall
                cmd = [
                    "netsh", "advfirewall", "firewall", "add", "rule",
                    f"name=NetGuard_block_{ip}", "protocol=any",
                    "dir=in", "action=block", f"remoteip={ip}",
                ]
            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            _, err = await result.communicate()
            if result.returncode == 0:
                logger.info(f"Blocked IP {ip}")
                # Persist to DB
                from app.db.database import BlockedIP
                async with AsyncSessionLocal() as session:
                    b = BlockedIP(ip_address=ip, reason="auto-block", alert_id=alert_id)
                    session.add(b)
                    await session.commit()
                return True
            logger.warning(f"Block failed for {ip}: {err.decode()}")
        except Exception as exc:
            logger.warning(f"Could not block {ip}: {exc}")
        return False

    # ── Email alert ────────────────────────────────────────────────────────

    async def _send_email_alert(self, flow: FlowRecord, result: PredictionResult):
        try:
            import aiosmtplib
            from email.message import EmailMessage
            msg = EmailMessage()
            msg["Subject"] = f"[NetGuard] {result.severity} — {result.attack_type} from {flow.src_ip}"
            msg["From"]    = settings.smtp_user
            msg["To"]      = settings.alert_recipient
            msg.set_content(
                f"Threat detected:\n\n"
                f"  Attack:     {result.attack_type}\n"
                f"  Severity:   {result.severity}\n"
                f"  Confidence: {result.confidence*100:.1f}%\n"
                f"  Source:     {flow.src_ip}:{flow.src_port}\n"
                f"  Target:     {flow.dst_ip}:{flow.dst_port}\n"
                f"  Protocol:   {flow.protocol}\n"
                f"  ML Model:   {result.ml_model}\n"
            )
            await aiosmtplib.send(
                msg,
                hostname=settings.smtp_host,
                port=settings.smtp_port,
                username=settings.smtp_user,
                password=settings.smtp_pass,
                start_tls=True,
            )
            logger.info(f"Email alert sent for {flow.src_ip}")
        except Exception as exc:
            logger.warning(f"Email alert failed: {exc}")

    # ── WebSocket broadcast ────────────────────────────────────────────────

    async def _broadcast(self, event: dict):
        if not self._ws_clients:
            return
        msg = json.dumps(event)
        dead = set()
        for ws in list(self._ws_clients):
            try:
                await ws.send_text(msg)
            except Exception:
                dead.add(ws)
        self._ws_clients -= dead

    def _build_ws_event(
        self, flow: FlowRecord, result: PredictionResult, alert_id: int
    ) -> dict:
        import datetime
        return {
            "type":       "threat_alert",
            "id":         alert_id,
            "timestamp":  datetime.datetime.utcnow().isoformat(),
            "src_ip":     flow.src_ip,
            "dst_ip":     flow.dst_ip,
            "src_port":   flow.src_port,
            "dst_port":   flow.dst_port,
            "protocol":   flow.protocol,
            "attack_type": result.attack_type,
            "severity":   result.severity,
            "confidence": round(result.confidence * 100, 1),
            "ml_level":   result.ml_level,
            "ml_model":   result.ml_model,
            "anomaly_score": round(result.anomaly_score, 4),
            "blocked":    False,
        }

    # ── Stats ticker ───────────────────────────────────────────────────────

    async def _stats_ticker(self):
        """Recalculate flows/sec and broadcast stats every second."""
        while self._running:
            await asyncio.sleep(1.0)
            now = time.time()
            self._flow_count_window = [t for t in self._flow_count_window if now - t < 5]
            self._stats["flows_per_sec"] = round(len(self._flow_count_window) / 5, 1)
            await self._broadcast({
                "type": "stats",
                **self._stats,
            })

    # ── Helpers ────────────────────────────────────────────────────────────

    def _email_severities(self):
        order = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        threshold = settings.alert_severity_threshold
        try:
            idx = order.index(threshold)
        except ValueError:
            idx = 2
        return order[idx:]

    def set_ml_level(self, level: int):
        assert 1 <= level <= 4
        self._ml_level = level
        logger.info(f"ML level changed to {level}")

    def get_stats(self) -> dict:
        return dict(self._stats)


# Singleton
pipeline = InferencePipeline()