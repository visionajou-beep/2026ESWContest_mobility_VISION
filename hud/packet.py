# -*- coding: utf-8 -*-
"""HUD 패킷 규격 — 검증 · 정제 · 젯슨 포맷(2 차식 계수) 변환.

젯슨이 보내는 JSON 한 개가 프레임 하나다. 받은 패킷은 _decode_packet 이 검사하고,
젯슨 포맷(left / right 2 차식 + y_range)은 _adapt_jetson_packet 이 정규화 폴리라인으로 편다.
"""
from __future__ import annotations

import json
import math
from typing import Any, Iterable


PROTOCOL_VERSION = 1


MAX_DATAGRAM_BYTES = 60_000


# 패킷이 실어 나르는 경고 어휘. 렌더러가 아니라 프로토콜 쪽 값이다.
WARNING_STATES = (
    "none",
    "departure_left",
    "departure_right",
    "change_left",
    "change_right",
)


class PacketError(ValueError):
    """잘못된 HUD 좌표 패킷."""


def _finite(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _segment_span_in_box(
    first: list[float], second: list[float], bounds: tuple[float, float, float, float]
) -> tuple[float, float] | None:
    """선분이 상자 안에 있는 매개변수 구간 [t0, t1] 을 구한다 (Liang-Barsky).

    상자를 전혀 지나지 않으면 None. t 는 first -> second 를 0 -> 1 로 본 값이다.
    """
    x_min, y_min, x_max, y_max = bounds
    dx = second[0] - first[0]
    dy = second[1] - first[1]
    t0, t1 = 0.0, 1.0
    for slope, offset in (
        (-dx, first[0] - x_min),
        (dx, x_max - first[0]),
        (-dy, first[1] - y_min),
        (dy, y_max - first[1]),
    ):
        if slope == 0.0:
            # 이 축으로 움직이지 않는다. 시작부터 밖이면 통째로 버린다.
            if offset < 0.0:
                return None
            continue
        ratio = offset / slope
        if slope < 0.0:
            if ratio > t1:
                return None
            t0 = max(t0, ratio)
        else:
            if ratio < t0:
                return None
            t1 = min(t1, ratio)
    return (t0, t1) if t0 <= t1 else None


def _lerp_point(first: list[float], second: list[float], t: float) -> list[float]:
    return [
        first[0] + (second[0] - first[0]) * t,
        first[1] + (second[1] - first[1]) * t,
    ]


def _clip_polyline(
    points: list[list[float]],
    bounds: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
) -> list[list[float]]:
    """상자 밖으로 나간 구간을 버리고, 경계를 넘는 자리에 교점을 끼운다.

    화면 밖 x 를 0 이나 1 로 뭉개면(0 ~ 1 로 자르는 방식) 여러 점이 같은 경계값에 몰려
    테두리를 따라 달리는 수직선이 생긴다. 다항식이 화면 밖을 크게 도는 구간이
    정확히 그렇다. 그래서 버리는 쪽으로 가되, 그냥 걷어 내면 선이 경계에서
    뚝 끊기므로 경계와의 교점을 계산해 끝점으로 넣는다.

    상자를 들락날락하면 조각이 여러 개 나온다. 차선은 폴리라인 하나로
    다뤄지므로 그중 가장 긴 조각만 남긴다.
    """
    if len(points) < 2:
        return []

    pieces: list[list[list[float]]] = []
    current: list[list[float]] = []

    def flush() -> None:
        nonlocal current
        if len(current) >= 2:
            pieces.append(current)
        current = []

    for index in range(len(points) - 1):
        first = points[index]
        second = points[index + 1]
        span = _segment_span_in_box(first, second, bounds)
        if span is None:
            flush()
            continue
        t0, t1 = span
        if t0 > 0.0:
            # 이 선분에서 상자로 들어왔다. 앞 조각과는 이어지지 않는다.
            flush()
        if not current:
            current = [_lerp_point(first, second, t0)]
        current.append(_lerp_point(first, second, t1))
        if t1 < 1.0:
            # 이 선분에서 상자를 빠져나갔다.
            flush()
    flush()

    if not pieces:
        return []
    return max(pieces, key=_path_length)


def _path_length(points: list[list[float]]) -> float:
    total = 0.0
    for index in range(len(points) - 1):
        dx = points[index + 1][0] - points[index][0]
        dy = points[index + 1][1] - points[index][1]
        total += math.hypot(dx, dy)
    return total


def _sanitize_lane(lane: Iterable[Iterable[float]], max_points: int = 48) -> list[list[float]]:
    points: list[list[float]] = []
    for raw_point in lane:
        point = list(raw_point)
        if len(point) != 2 or not _finite(point[0]) or not _finite(point[1]):
            continue
        points.append([float(point[0]), float(point[1])])
    points.sort(key=lambda item: item[1])
    # 화면 밖은 clamp 하지 않고 잘라 낸다. clamp 하면 경계에 점이 몰려
    # 차선이 테두리를 따라 꺾여 올라간다.
    points = _clip_polyline(points)
    if len(points) > max_points:
        indices = [round(i * (len(points) - 1) / (max_points - 1)) for i in range(max_points)]
        points = [points[index] for index in indices]
    if len(points) < 2:
        return []
    return [[round(x, 5), round(y, 5)] for x, y in points]


def _select_hud_lanes_with_ids(
    lanes: Iterable[Iterable[Iterable[float]]],
    ids: Iterable[Any] | None = None,
    *,
    lane_change: bool,
) -> tuple[list[list[list[float]]], list[Any]]:
    """평상시 2개, 차선 변경 시 최대 3개 차선을 선택한다.

    젯슨이 보내는 차선 트랙 id 를 같이 받아 정렬·선택 뒤에도 차선과 짝이
    유지되게 돌려준다. id 는 LaneSmoother 가 차선 동일성 판정에 쓴다.
    id 를 모르는 차선 자리에는 None 이 들어간다.
    """
    id_list = list(ids) if ids is not None else []
    valid: list[tuple[list[list[float]], Any]] = []
    for index, lane in enumerate(lanes):
        cleaned = _sanitize_lane(lane)
        if not cleaned:
            continue
        lane_id = id_list[index] if index < len(id_list) else None
        valid.append((cleaned, lane_id))
    valid.sort(key=lambda item: item[0][-1][0])
    limit = 3 if lane_change else 2
    if len(valid) > limit:
        bottom_x = [lane[-1][0] for lane, _ in valid]
        if lane_change:
            selected = sorted(
                range(len(valid)),
                key=lambda index: abs(bottom_x[index] - 0.5),
            )[:3]
        else:
            left = [index for index, x in enumerate(bottom_x) if x <= 0.5]
            right = [index for index, x in enumerate(bottom_x) if x > 0.5]
            if left and right:
                selected = [left[-1], right[0]]
            else:
                selected = sorted(
                    range(len(valid)),
                    key=lambda index: abs(bottom_x[index] - 0.5),
                )[:2]
        valid = [valid[index] for index in sorted(selected)]
    return [lane for lane, _ in valid], [lane_id for _, lane_id in valid]


def _select_hud_lanes(
    lanes: Iterable[Iterable[Iterable[float]]],
    *,
    lane_change: bool,
) -> list[list[list[float]]]:
    """평상시 2개, 차선 변경 시 최대 3개 차선을 선택한다."""
    selected, _ = _select_hud_lanes_with_ids(lanes, None, lane_change=lane_change)
    return selected


# 젯슨 실제 송신 포맷 어댑터 -------------------------------------------------
#
# 젯슨은 lanes 배열이 아니라 left/right 를 따로, 그것도 점 배열이 아니라
# 2차 다항식 계수로 보낸다. 실제 패킷 예:
#
#   {"v":1,"seq":292,"ts":1789149939.25,"img_w":960,"img_h":540,
#    "state":0,"conf":1.0,
#    "left":[-0.003151,0.505674,423.193818],
#    "right":[-0.008059,5.896502,-500.737265],
#    "fps":0.0,"id_left":1,"id_right":3,
#    "event":"","intent":"","extra":null,"y_range":[216.0,410.4]}
#
# left/right = [a, b, c] 이고 x = a*y^2 + b*y + c, 좌표 단위는 픽셀,
# y 의 유효 구간이 y_range = [y_top, y_bottom] 이다. 위 패킷을 풀면
# left 는 y=216 에서 x=385, y=410 에서 x=100, right 는 같은 구간에서
# x=397 -> x=562 로 소실점에서 수렴하고 아래에서 벌어진다.

# 계수를 몇 점으로 펴서 넘길지 (12 개 고정).
JETSON_LANE_SAMPLES = 12


# status 필드가 없어서 state 로 유추한다.
JETSON_STATE_STATUS = {0: "ok", 1: "degraded", 2: "lost"}


def _lane_track_id(value: Any) -> Any:
    """차선 트랙 id 를 비교 가능한 값으로 정리한다. 모르면 None."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value:
        return value
    return None


def _is_jetson_packet(packet: dict[str, Any]) -> bool:
    """lanes 가 없고 left/right 가 있으면 젯슨 포맷으로 본다."""
    return "lanes" not in packet and ("left" in packet or "right" in packet)


def _poly_lane_to_normalized(
    coefficients: Any,
    y_range: Any,
    *,
    img_w: float,
    img_h: float,
    samples: int = JETSON_LANE_SAMPLES,
) -> list[list[float]]:
    """2차 계수 + y 구간을 0~1 정규화 폴리라인으로 편다."""
    if not isinstance(coefficients, (list, tuple)) or len(coefficients) != 3:
        raise PacketError("lane must be 3 polynomial coefficients")
    if not all(_finite(value) for value in coefficients):
        raise PacketError("lane coefficients must be finite")
    if not isinstance(y_range, (list, tuple)) or len(y_range) != 2:
        raise PacketError("y_range must be a pair")
    if not all(_finite(value) for value in y_range):
        raise PacketError("y_range must be finite")

    a, b, c = (float(value) for value in coefficients)
    y_top, y_bottom = (float(value) for value in y_range)
    if y_bottom < y_top:
        y_top, y_bottom = y_bottom, y_top
    if y_bottom - y_top < 1.0:
        raise PacketError("y_range span too small")

    # send_pixels 와 같은 규칙으로 나눈다 (마지막 픽셀이 1.0 이 되도록)
    width = max(1.0, float(img_w) - 1.0)
    height = max(1.0, float(img_h) - 1.0)

    points: list[list[float]] = []
    for index in range(samples):
        progress = index / (samples - 1)
        y = y_top + (y_bottom - y_top) * progress
        x = (a * y + b) * y + c
        points.append([x / width, y / height])
    return points


def _adapt_jetson_packet(packet: dict[str, Any]) -> None:
    """젯슨 포맷을 기존 lanes 포맷으로 옮긴다. 원본 키는 지우지 않는다."""
    img_w = packet.get("img_w")
    img_h = packet.get("img_h")
    if not _finite(img_w) or not _finite(img_h):
        raise PacketError("img_w/img_h must be numbers")
    if float(img_w) < 2.0 or float(img_h) < 2.0:
        raise PacketError("img_w/img_h must be at least 2")

    y_range = packet.get("y_range")
    lanes: list[list[list[float]]] = []
    lane_ids: list[Any] = []
    for side in ("left", "right"):
        coefficients = packet.get(side)
        if coefficients is None:
            continue
        # 한쪽이라도 계수가 있으면 y_range 없이는 펼 수 없다
        lanes.append(
            _poly_lane_to_normalized(
                coefficients,
                y_range,
                img_w=float(img_w),
                img_h=float(img_h),
            )
        )
        # 트랙 id 를 차선과 같은 순서로 실어 보낸다. 스무딩이 이걸로 차선
        # 동일성을 판정한다. 없으면 None 자리를 남겨 근접도 폴백으로 간다.
        lane_ids.append(_lane_track_id(packet.get(f"id_{side}")))
    packet["lanes"] = lanes
    packet["lane_ids"] = lane_ids

    # ts -> sent_at
    if "sent_at" not in packet and _finite(packet.get("ts")):
        packet["sent_at"] = float(packet["ts"])

    # status 가 없으면 state 로 유추한다
    if "status" not in packet:
        state = packet.get("state")
        status = None
        if isinstance(state, bool):
            state = None
        if isinstance(state, (int, float)) and _finite(state):
            status = JETSON_STATE_STATUS.get(int(state))
        if status is None:
            # state 도 못 믿으면 HudSender 와 같은 규칙으로 개수를 본다
            if not lanes:
                status = "lost"
            elif len(lanes) < 2:
                status = "degraded"
            else:
                status = "ok"
        packet["status"] = status

    # conf -> confidence (상태 계층이 읽는 이름)
    if "confidence" not in packet and _finite(packet.get("conf")):
        packet["confidence"] = float(packet["conf"])

    # event / intent 는 그대로 살려 두고, 이름이 정확히 일치할 때만
    # warning 으로 올린다. 젯슨 쪽 어휘를 아직 못 받아서 추측 매핑은 안 한다.
    if "warning" not in packet:
        for key in ("event", "intent"):
            value = packet.get(key)
            if isinstance(value, str) and value in WARNING_STATES:
                packet["warning"] = value
                break


def _decode_packet(payload: bytes) -> dict[str, Any]:
    if len(payload) > MAX_DATAGRAM_BYTES:
        raise PacketError("packet exceeds size limit")
    try:
        packet = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PacketError("invalid JSON") from exc
    if not isinstance(packet, dict):
        raise PacketError("packet must be an object")
    if packet.get("v", PROTOCOL_VERSION) != PROTOCOL_VERSION:
        raise PacketError("unsupported protocol version")
    if _is_jetson_packet(packet):
        _adapt_jetson_packet(packet)
    sequence = packet.get("seq")
    if not isinstance(sequence, int) or sequence < 0:
        raise PacketError("seq must be a non-negative integer")
    status = packet.get("status", "ok")
    if status not in {"ok", "degraded", "lost", "mock"}:
        raise PacketError("unknown status")
    lane_change = bool(packet.get("lane_change", False))
    raw_lanes = packet.get("lanes", [])
    if not isinstance(raw_lanes, list):
        raise PacketError("lanes must be a list")
    raw_ids = packet.get("lane_ids")
    if not isinstance(raw_ids, list):
        raw_ids = None
    # 선택·정렬로 차선 순서가 바뀌므로 id 도 같이 따라가야 짝이 안 어긋난다.
    packet["lanes"], packet["lane_ids"] = _select_hud_lanes_with_ids(
        raw_lanes, raw_ids, lane_change=lane_change
    )
    packet["status"] = status
    packet["lane_change"] = lane_change
    packet["fps"] = float(packet.get("fps", 0.0))
    packet["inference_ms"] = float(packet.get("inference_ms", 0.0))
    return packet
