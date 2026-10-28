# -*- coding: utf-8 -*-
"""UDP 수신 → 스무딩 → 상태 판정 — 렌더러에 넘길 프레임 인자를 만든다.

receive_latest 는 쌓인 패킷을 전부 꺼내 가장 최근 것 하나만 돌려준다. 한 번에 하나씩 꺼내면 렌더가 송신보다
느린 순간마다 패킷이 쌓이고 화면은 옛 차선을 그린다. LaneFeed 는 receive 와 trim 이 같이 쓴다 — 두 곳이 따로 판정하면
보정할 때 본 화면과 주행 때 화면이 달라진다.
"""
from __future__ import annotations

import select
import socket
import time
from typing import Any

from .packet import WARNING_STATES, PacketError, _decode_packet
from .render import STATE_LOST, STATE_NORMAL, normalize_state
from .smooth import LaneSmoother
from .state import StateTracker, ensure_state_config


# 기본값 212992B 면 젯슨 패킷(~260B, 커널 계산으로 ~1KB)이 221개, 28fps 로
# 7.9초치가 쌓인다. 매 프레임 전부 비우므로 평소에는 1~2개만 들어 있고,
# 버퍼는 렌더가 멈췄을 때만 의미가 있다. 가득 차면 커널은 새 패킷을 버리고
# 옛 패킷을 남기므로 크게 잡을수록 멈춘 뒤 첫 화면이 더 옛날 것이 된다.
# 16384 요청 -> 실제 32768, 젯슨 패킷 34개(1.2초)다.
DEFAULT_RECV_BUFFER_BYTES = 16_384


def open_receiver(
    bind_host: str, port: int, recv_buffer_bytes: int = DEFAULT_RECV_BUFFER_BYTES
) -> socket.socket:
    """수신용 논블로킹 UDP 소켓."""
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    receiver.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, int(recv_buffer_bytes))
    receiver.bind((bind_host, port))
    receiver.setblocking(False)
    return receiver


def receive_latest(
    receiver: socket.socket, timeout: float, last_sequence: int
) -> tuple[dict[str, Any] | None, int, int, Any]:
    """쌓인 패킷을 전부 꺼내 가장 최근 것 하나만 돌려준다.

    한 번에 하나씩 꺼내면 렌더가 송신보다 느린 순간마다 커널 버퍼에 패킷이
    쌓이고, 화면은 그만큼 옛날 차선을 계속 그린다.
    seq 가 last_sequence 보다 작은 패킷(역전)은 쓰지 않는다.

    반환: (패킷 또는 None, 안 쓰고 버린 정상 패킷 수, 깨진 패킷 수,
    마지막 깨진 패킷의 송신 주소)
    """
    readable, _, _ = select.select([receiver], [], [], timeout)
    if not readable:
        return None, 0, 0, None
    best: dict[str, Any] | None = None
    valid = 0
    invalid = 0
    invalid_address = None
    while True:
        try:
            payload, address = receiver.recvfrom(65_535)
        except BlockingIOError:
            break
        try:
            packet = _decode_packet(payload)
        except (PacketError, ValueError, TypeError):
            invalid += 1
            invalid_address = address
            continue
        valid += 1
        if packet["seq"] < last_sequence:
            continue
        if best is None or packet["seq"] >= best["seq"]:
            best = packet
    skipped = valid - (1 if best is not None else 0)
    return best, skipped, invalid, invalid_address


def _blank_debug(**overrides: Any) -> dict[str, Any]:
    info = {
        "status": "ok",
        "lanes": 0,
        "seq": 0,
        "fps": 0.0,
        "inference_ms": 0.0,
        "age_ms": 0.0,
        "render_ms": 0.0,
        "dropped": 0,
        "skipped": 0,
        "lag_ms": None,
        "queue_ms": None,
        "warning": "none",
    }
    info.update(overrides)
    return info


def _latency_line(info: dict[str, Any]) -> str:
    """젯슨 ts 기준 지연. LAG 는 시계 차이를 포함한 원값, Q 는 지금까지 본
    최솟값을 뺀 값이라 시계 차이가 빠지고 밀린 만큼만 남는다."""
    lag = info.get("lag_ms")
    queue = info.get("queue_ms")
    lag_text = "--" if lag is None else f"{lag:.0f}ms"
    queue_text = "--" if queue is None else f"{queue:.0f}ms"
    return f"LAG {lag_text}  Q {queue_text}  SKIP {info.get('skipped', 0)}"


# ---------------------------------------------------------------------------
# receive: UDP 수신 + 이 파일의 화면 구성
# ---------------------------------------------------------------------------


class LaneFeed:
    """UDP 수신 -> 스무딩 -> 상태 계층까지. 렌더러에 넘길 인자를 만든다.

    receive 와 calibrate trim 이 같은 경로로 실데이터를 받도록 한 곳에
    둔다. 두 곳이 따로 판정하면 보정할 때 본 화면과 주행 때 화면이 달라진다.
    """

    def __init__(self, config: dict[str, Any], bind_host: str, port: int) -> None:
        ensure_state_config(config)
        display = config["display"]
        self.smoother = LaneSmoother(
            alpha=float(display.get("smoothing_alpha", 0.55)),
            association_distance=float(display.get("association_distance", 0.18)),
            jump_gate_distance=float(display.get("jump_gate_distance", 0.05)),
            jump_gate_frames=int(display.get("jump_gate_frames", 3)),
        )
        # 무수신 판정과 state 디바운싱은 전부 여기에 있다. 이 루프는 판정을
        # 하지 않고 결과만 받아 그린다.
        self.tracker = StateTracker(config)

        network = config.get("network", {})
        self.socket = open_receiver(
            bind_host,
            port,
            int(network.get("recv_buffer_bytes", DEFAULT_RECV_BUFFER_BYTES)),
        )

        self.latest: dict[str, Any] | None = None
        self.latest_received: float | None = None
        self.last_sequence = -1
        # 깨진 패킷 수
        self.dropped = 0
        # 더 새 패킷이 같이 쌓여 있어서 안 쓰고 버린 정상 패킷 수
        self.skipped = 0
        # 지연 측정. 젯슨 ts 는 epoch 초라 파이 시계와 차이가 섞여 있으므로
        # 최솟값을 기준선으로 잡고 그 위로 얼마나 밀렸는지를 따로 본다.
        self.lag_ms: float | None = None
        self.lag_floor_ms: float | None = None

    def close(self) -> None:
        self.socket.close()

    def poll(self, timeout: float) -> None:
        """쌓인 패킷을 전부 꺼내 가장 최근 것만 쓴다. 없으면 timeout 만큼 기다린다."""
        packet, skipped, invalid, address = receive_latest(
            self.socket, timeout, self.last_sequence
        )
        self.skipped += skipped
        for _ in range(invalid):
            self.dropped += 1
            if self.dropped % 30 == 1:
                print(f"ignored invalid packet from {address}")
        if packet is None:
            return
        packet["lanes"] = self.smoother.update(packet["lanes"], packet.get("lane_ids"))
        self.latest = packet
        self.latest_received = time.monotonic()
        self.last_sequence = packet["seq"]

    def frame(self, now: float) -> dict[str, Any]:
        """이번 프레임에 그릴 것. elapsed 를 뺀 render() 키워드 인자다."""
        latest = self.latest
        age = now - self.latest_received if self.latest_received is not None else float("inf")

        raw_state = STATE_NORMAL
        confidence = 1.0
        if latest is not None:
            raw_state = normalize_state(latest.get("state", STATE_NORMAL))
            if latest["status"] == "lost":
                # 차선을 못 봤다는 뜻이라 state 2 와 같은 이야기다
                raw_state = STATE_LOST
            confidence = float(latest.get("confidence", 1.0))
        status = self.tracker.update(
            now,
            raw_state=raw_state,
            confidence=confidence,
            last_received=self.latest_received,
        )

        if status.link_lost:
            # 송신기가 재시작해 seq 가 0 으로 돌아와도 다시 받아들인다.
            self.last_sequence = -1
            self.smoother.reset()
            # 송신기가 바뀌었을 수 있으니 시계 기준선도 다시 잡는다
            self.lag_floor_ms = None

        # 렌더 시각 - 송신 시각
        self.lag_ms = None
        if latest is not None and "sent_at" in latest:
            self.lag_ms = (time.time() - float(latest["sent_at"])) * 1000.0
            if self.lag_floor_ms is None or self.lag_ms < self.lag_floor_ms:
                self.lag_floor_ms = self.lag_ms
        queue_ms = None
        if self.lag_ms is not None and self.lag_floor_ms is not None:
            queue_ms = self.lag_ms - self.lag_floor_ms

        warning = "none"
        telemetry: dict[str, Any] = {}
        info = _blank_debug(
            status="timeout",
            dropped=self.dropped,
            skipped=self.skipped,
            age_ms=min(age, 9.999) * 1000.0,
        )
        if latest is not None and not status.link_lost:
            warning = str(latest.get("warning", "none"))
            if warning not in WARNING_STATES:
                warning = "none"
            # confidence 는 상태 계층이 이미 state 에 반영했다. 렌더러에
            # 다시 넘기면 디바운싱을 건너뛴 판정이 한 번 더 붙는다.
            telemetry = {
                "departure_distance": float(latest.get("departure_distance", 0.0)),
                "lkas": bool(latest.get("lkas", True)),
                "acc": bool(latest.get("acc", True)),
                "fps": latest["fps"],
                "inference_ms": latest["inference_ms"],
            }
            info = _blank_debug(
                status=latest["status"],
                lanes=len(latest["lanes"]),
                seq=latest["seq"],
                fps=latest["fps"],
                inference_ms=latest["inference_ms"],
                age_ms=age * 1000.0,
                dropped=self.dropped,
                skipped=self.skipped,
                lag_ms=self.lag_ms,
                queue_ms=queue_ms,
                warning=warning,
            )
        # 페이드가 남아 있는 동안은 마지막으로 받은 좌표를 그대로 어둡게
        # 깔아 둔다. 뚝 끊기는 것보다 사라지는 편이 덜 놀랍다.
        lanes: list[Any] = []
        if latest is not None and status.lanes_visible:
            lanes = latest["lanes"]

        return dict(
            lanes=lanes,
            warning=warning,
            state=status.state,
            lane_state=status.lane_state,
            lane_opacity=status.lane_opacity,
            telemetry=telemetry,
            debug=info,
        )


class _LatencyStats:
    """receive --stats. 2초마다 지연 요약을 한 줄씩 찍는다.

    고치기 전후, 설정을 바꾸기 전후를 숫자로 비교하려는 용도다. LAG 는 젯슨
    시계와의 차이가 섞인 원값이라 절대값보다 추세를 본다. Q 는 최솟값을 뺀
    값이라 계속 커지면 패킷이 밀리고 있다는 뜻이다.
    """

    def __init__(self, feed: "LaneFeed", interval: float = 2.0) -> None:
        self.feed = feed
        self.interval = interval
        self.started = time.monotonic()
        self.window = self.started
        self.frames = 0
        self.lags: list[float] = []
        self.queues: list[float] = []
        self.skipped = feed.skipped
        print("  t[s]   fps  lag_avg  lag_max  q_avg  q_max  skip/s  seq")

    def add(self, now: float, info: dict[str, Any]) -> None:
        self.frames += 1
        if info.get("lag_ms") is not None:
            self.lags.append(float(info["lag_ms"]))
            self.queues.append(float(info["queue_ms"]))
        span = now - self.window
        if span < self.interval:
            return
        skip_rate = (self.feed.skipped - self.skipped) / span
        if self.lags:
            print(
                f"{now - self.started:6.1f} {self.frames / span:5.1f} "
                f"{sum(self.lags) / len(self.lags):8.1f} {max(self.lags):8.1f} "
                f"{sum(self.queues) / len(self.queues):6.1f} {max(self.queues):6.1f} "
                f"{skip_rate:7.1f}  {info.get('seq', 0)}",
                flush=True,
            )
        else:
            print(f"{now - self.started:6.1f} {self.frames / span:5.1f}  no packets",
                  flush=True)
        self.window = now
        self.frames = 0
        self.lags = []
        self.queues = []
        self.skipped = self.feed.skipped
