# -*- coding: utf-8 -*-
"""HUD 메시지 송신 어댑터와 모의 송신기.

HudSender 는 추론 코드가 차선 폴리라인을 UDP 로 보낼 때 쓴다(젯슨 쪽은 ego_lane/hud_message 가 같은 규격으로 보낸다).
HudUiSender 는 여기에 차선 이탈 경고(warning)를 더한다. mock 은 젯슨 없이 HUD 를 시험할 때 가짜 차선을 보낸다.

    python -m hud.sender mock --destination 127.0.0.1:5005
"""
from __future__ import annotations

import argparse
import json
import math
import socket
import time
from typing import Any, Iterable

from .packet import MAX_DATAGRAM_BYTES, PROTOCOL_VERSION, WARNING_STATES, PacketError, _select_hud_lanes


class HudSender:
    """추론 코드에서 사용하는 HUD UDP 송신 어댑터."""

    def __init__(self, host: str, port: int = 5005) -> None:
        self.destination = (host, int(port))
        self.sequence = 0
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def close(self) -> None:
        self.socket.close()

    def send_normalized(
        self,
        lanes: Iterable[Iterable[Iterable[float]]],
        *,
        lane_change: bool = False,
        status: str | None = None,
        fps: float = 0.0,
        inference_ms: float = 0.0,
    ) -> None:
        selected = _select_hud_lanes(lanes, lane_change=lane_change)
        if status is None:
            if not selected:
                status = "lost"
            elif len(selected) < 2:
                status = "degraded"
            else:
                status = "ok"
        packet = {
            "v": PROTOCOL_VERSION,
            "seq": self.sequence,
            "sent_at": time.time(),
            "status": status,
            "fps": round(float(fps), 2),
            "inference_ms": round(float(inference_ms), 2),
            "lane_change": bool(lane_change),
            "lanes": selected,
        }
        payload = json.dumps(packet, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        if len(payload) > MAX_DATAGRAM_BYTES:
            raise PacketError(f"packet too large: {len(payload)} bytes")
        self.socket.sendto(payload, self.destination)
        self.sequence += 1

    def send_pixels(
        self,
        lanes: Iterable[Iterable[Iterable[float]]],
        *,
        frame_width: int,
        frame_height: int,
        lane_change: bool = False,
        status: str | None = None,
        fps: float = 0.0,
        inference_ms: float = 0.0,
    ) -> None:
        width = max(1, int(frame_width) - 1)
        height = max(1, int(frame_height) - 1)
        normalized = [
            [[float(x) / width, float(y) / height] for x, y in lane]
            for lane in lanes
        ]
        self.send_normalized(
            normalized,
            lane_change=lane_change,
            status=status,
            fps=fps,
            inference_ms=inference_ms,
        )

    def __enter__(self) -> "HudSender":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class HudUiSender:
    """추론 코드용 송신기. 기존 HudSender 에 warning 항목을 더한다."""

    def __init__(self, host: str, port: int = 5005) -> None:
        self.destination = (host, int(port))
        self.sequence = 0
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def close(self) -> None:
        self.socket.close()

    def send(
        self,
        lanes: Any,
        *,
        warning: str = "none",
        lane_change: bool = False,
        fps: float = 0.0,
        inference_ms: float = 0.0,
        frame_width: int | None = None,
        frame_height: int | None = None,
        confidence: float = 1.0,
        departure_distance: float = 0.0,
        lkas: bool = True,
        acc: bool = True,
    ) -> None:
        if warning not in WARNING_STATES:
            raise ValueError(f"unknown warning: {warning}")
        if frame_width and frame_height:
            w = max(1, int(frame_width) - 1)
            h = max(1, int(frame_height) - 1)
            lanes = [[[float(x) / w, float(y) / h] for x, y in lane] for lane in lanes]
        selected = _select_hud_lanes(lanes, lane_change=lane_change)
        if not selected:
            status = "lost"
        elif len(selected) < 2:
            status = "degraded"
        else:
            status = "ok"
        packet = {
            "v": 1,
            "seq": self.sequence,
            "sent_at": time.time(),
            "status": status,
            "fps": round(float(fps), 2),
            "inference_ms": round(float(inference_ms), 2),
            "lane_change": bool(lane_change),
            "warning": warning,
            "confidence": round(float(confidence), 3),
            "departure_distance": round(float(departure_distance), 2),
            "lkas": bool(lkas),
            "acc": bool(acc),
            "lanes": selected,
        }
        payload = json.dumps(packet, separators=(",", ":")).encode("utf-8")
        self.socket.sendto(payload, self.destination)
        self.sequence += 1

    def __enter__(self) -> "HudUiSender":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _mock_lane(bottom_x: float, top_x: float, phase: float, count: int = 24) -> list[list[float]]:
    output = []
    for index in range(count):
        y = 0.38 + 0.60 * index / (count - 1)
        progress = (y - 0.38) / 0.60
        x = top_x + (bottom_x - top_x) * progress**1.45
        x += 0.012 * math.sin(phase + progress * 2.2)
        output.append([min(1.0, max(0.0, x)), y])
    return output


def run_mock_sender(args: argparse.Namespace) -> None:
    host, port_text = args.destination.rsplit(":", 1)
    sender = HudSender(host, int(port_text))
    started = time.monotonic()
    print(f"mock sender -> {host}:{port_text}; Ctrl+C to stop")
    try:
        while True:
            elapsed = time.monotonic() - started
            lane_change = bool(args.cycle and int(elapsed / 6) % 2)
            phase = elapsed * 0.8
            if lane_change:
                lanes = [
                    _mock_lane(0.14, 0.42, phase),
                    _mock_lane(0.50, 0.54, phase + 0.2),
                    _mock_lane(0.86, 0.68, phase + 0.4),
                ]
            else:
                lanes = [
                    _mock_lane(0.18, 0.43, phase),
                    _mock_lane(0.82, 0.57, phase + 0.3),
                ]
            sender.send_normalized(
                lanes,
                lane_change=lane_change,
                status="mock",
                fps=args.fps,
            )
            time.sleep(max(0.001, 1.0 / args.fps))
    except KeyboardInterrupt:
        pass
    finally:
        sender.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="모델 없이 2 / 3 개 차선 좌표 송신")
    sub = parser.add_subparsers(dest="command", required=True)
    mock = sub.add_parser("mock", help="모델 없이 2/3개 차선 좌표 송신")
    mock.add_argument("--destination", default="127.0.0.1:5005")
    mock.add_argument("--fps", type=float, default=22.0)
    mock.add_argument("--cycle", action="store_true", help="6초마다 일반/차선 변경 전환")
    args = parser.parse_args()
    run_mock_sender(args)


if __name__ == "__main__":
    main()
