import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List

from scapy.all import sniff, IP, TCP, UDP

FLOW_TIMEOUT = 10  # seconds


@dataclass
class FlowRecord:
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str

    start_time: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)

    packet_count: int = 0
    byte_count: int = 0

    syn_count: int = 0
    ack_count: int = 0
    fin_count: int = 0
    rst_count: int = 0
    psh_count: int = 0
    urg_count: int = 0

    iats: List[float] = field(default_factory=list)
    last_pkt_time: float = None

    def update(self, pkt):
        now = time.time()

        if self.last_pkt_time:
            self.iats.append(now - self.last_pkt_time)

        self.last_pkt_time = now
        self.last_seen = now

        self.packet_count += 1
        self.byte_count += len(pkt)

        if TCP in pkt:
            flags = pkt[TCP].flags
            self.syn_count += int(flags & 0x02 != 0)
            self.ack_count += int(flags & 0x10 != 0)
            self.fin_count += int(flags & 0x01 != 0)
            self.rst_count += int(flags & 0x04 != 0)
            self.psh_count += int(flags & 0x08 != 0)
            self.urg_count += int(flags & 0x20 != 0)

    def to_feature_vector(self):
        duration = max((self.last_seen - self.start_time) * 1000, 1)

        avg_pkt_size = self.byte_count / max(self.packet_count, 1)
        pkt_rate = self.packet_count / (duration / 1000)
        byte_rate = self.byte_count / (duration / 1000)

        iat_mean = sum(self.iats) / len(self.iats) if self.iats else 0
        iat_std = (sum((x - iat_mean) ** 2 for x in self.iats) / len(self.iats)) ** 0.5 if self.iats else 0
        iat_min = min(self.iats) if self.iats else 0
        iat_max = max(self.iats) if self.iats else 0

        proto_tcp = 1 if self.protocol == "TCP" else 0
        proto_udp = 1 if self.protocol == "UDP" else 0
        proto_icmp = 0

        pkt_ratio = self.packet_count / duration
        byte_per_pkt = self.byte_count / max(self.packet_count, 1)

        return [
            duration,
            self.packet_count,
            self.byte_count,
            avg_pkt_size,
            pkt_rate,
            byte_rate,
            iat_mean,
            iat_std,
            iat_min,
            iat_max,
            self.syn_count,
            self.ack_count,
            self.fin_count,
            self.rst_count,
            self.psh_count,
            self.urg_count,
            self.src_port,
            self.dst_port,
            proto_tcp,
            proto_udp,
            proto_icmp,
            pkt_ratio,
            byte_per_pkt
        ]


class PacketSniffer:
    def __init__(self, queue, interface="en0"):
        self.queue = queue
        self.interface = interface
        self.flows = {}
        self.running = False

    def _flow_key(self, pkt):
        if IP not in pkt:
            return None

        proto = "TCP" if TCP in pkt else "UDP" if UDP in pkt else "OTHER"

        return (
            pkt[IP].src,
            pkt[IP].dst,
            pkt.sport if hasattr(pkt, "sport") else 0,
            pkt.dport if hasattr(pkt, "dport") else 0,
            proto
        )

    def _process_packet(self, pkt):
        key = self._flow_key(pkt)
        if not key:
            return

        if key not in self.flows:
            self.flows[key] = FlowRecord(*key)

        self.flows[key].update(pkt)

    async def _flush_flows(self):
        while self.running:
            now = time.time()

            for key in list(self.flows.keys()):
                flow = self.flows[key]
                if now - flow.last_seen > FLOW_TIMEOUT:
                    await self.queue.put(flow)
                    del self.flows[key]

            await asyncio.sleep(1)

    def start(self, loop):
        self.running = True

        loop.create_task(self._flush_flows())

        def run():
            sniff(
                iface=self.interface,
                prn=self._process_packet,
                store=False
            )

        import threading
        threading.Thread(target=run, daemon=True).start()

    def stop(self):
        self.running = False