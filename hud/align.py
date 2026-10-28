# -*- coding: utf-8 -*-
"""운전자 시점 정렬 — 카메라 영상 좌표를 HUD 패널 픽셀로 옮기는 3×3 행렬.

차선은 노면 위에 있고 노면은 평면이므로 카메라 영상 → 노면 → 운전자 시점 → HUD 패널이 전부 평면 대 평면 대응이고,
호모그래피끼리 곱해도 호모그래피라 최종 결과는 행렬 하나다. 카메라 위치 · 눈 높이 · 반사 각도 · 좌우 반전이 전부 이 안에 들어간다.
행렬은 노면 위 같은 지점의 대응쌍(카메라 영상, HUD 화면)으로 실측해 구한다(calibrate 모듈). 그 위에 얹는 미세 보정이 TrimParams.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from typing import Any

import cv2
import numpy as np

from .packet import _clip_polyline


MIN_PAIRS = 4


MAX_PAIRS = 10


DEFAULT_ALIGNMENT: dict[str, Any] = {
    "mode": "quad",
    "camera_horizon_y": 0.0,
    "horizon_margin": 0.18,
    "pan_x": 0,
    "pan_y": 0,
    "pairs": [],
    "profiles": [],
    "blend": 0.0,
    "trim": {"dx_px": 0.0, "dy_px": 0.0, "keystone": 0.0, "roll_deg": 0.0},
}


# 트림 한 번 누를 때 움직이는 양. fine 이 기본이고 coarse 는 큰 폭.
DEFAULT_TRIM_STEP: dict[str, dict[str, float]] = {
    "fine": {"move_px": 1.0, "keystone": 0.002, "roll_deg": 0.05},
    "coarse": {"move_px": 8.0, "keystone": 0.01, "roll_deg": 0.5},
}


# 이 범위를 넘으면 보정이 아니라 설치가 틀어진 것이다. 키를 눌러도 더
# 가지 않게 막아 두면 사다리꼴 분모가 0 쪽으로 가는 일도 없다.
TRIM_LIMITS = {"keystone": 0.2, "roll_deg": 15.0}


def ensure_alignment_config(config: dict[str, Any]) -> dict[str, Any]:
    section = config.setdefault("alignment", {})
    for key, value in DEFAULT_ALIGNMENT.items():
        section.setdefault(key, dict(value) if isinstance(value, dict) else value)
    trim = section["trim"]
    for key, value in DEFAULT_ALIGNMENT["trim"].items():
        trim.setdefault(key, value)
    step = section.setdefault("trim_step", {})
    for size, values in DEFAULT_TRIM_STEP.items():
        target = step.setdefault(size, {})
        for key, value in values.items():
            target.setdefault(key, value)
    return section


@dataclass(frozen=True)
class TrimParams:
    """기존 정렬 위에 얹는 미세 보정 4개. 전부 운전자가 보는 방향 기준이다.

    dx_px, dy_px  평행이동. 패널 픽셀. +x 는 운전자 기준 오른쪽, +y 는 아래.
    keystone      상하 사다리꼴. + 면 위쪽 폭이 넓어진다. 화면 맨 아랫줄이 축이라
                  아랫줄 폭과 높이는 그대로다. 가까운 차선을 평행이동으로 먼저
                  맞춘 뒤 이것으로 먼 쪽 벌어짐만 맞추면 앞서 맞춘 것이 안 깨진다.
                  0.01 이면 화면 맨 윗줄 폭이 약 2% 넓어진다.
    roll_deg      좌우 기울기. 화면 중앙을 축으로 돌린다. + 면 시계 방향이라
                  오른쪽이 내려가고 왼쪽이 올라간다.

    보정은 화면 중앙 (cx, cy) 을 원점으로, 화면 반 높이 s 를 단위로 한
    좌표 (u, v) 에서 정의한다. v 는 맨 위가 -1, 맨 아래가 +1 이다. 가로세로에
    같은 s 를 써야 회전이 찌그러지지 않는다.

        K = [[1, 0, 0], [0, 1, -k], [0, k, 1 - k]]  사다리꼴 (w = 1 + k·(v - 1))
        R = 회전 roll_deg                          기울기
        T = 평행이동 (dx/s, dy/s)
        보정 = N⁻¹ · T · R · K · N                 N 은 픽셀 -> (u, v)

    K 는 아랫줄 v = 1 로 옮긴 뒤 투영 행 [0, k, 1] 을 걸고 되돌린 것이라
    v = 1 에서 w = 1 이다. 그 위로 갈수록 w 가 1 보다 작아져 벌어진다.
    투영 성분이라 결과가 여전히 호모그래피이고, 가상상 평면이 아랫변을
    축으로 살짝 숙여진 것과 같은 변형이다.
    """

    dx_px: float = 0.0
    dy_px: float = 0.0
    keystone: float = 0.0
    roll_deg: float = 0.0

    @classmethod
    def from_config(cls, section: dict[str, Any]) -> "TrimParams":
        raw = section.get("trim") or {}
        return cls(**{k: float(raw.get(k, 0.0)) for k in cls.__dataclass_fields__})

    def to_config(self) -> dict[str, float]:
        return {k: round(v, 5) for k, v in asdict(self).items()}

    def clamped(self, width: int, height: int) -> "TrimParams":
        k_max, r_max = TRIM_LIMITS["keystone"], TRIM_LIMITS["roll_deg"]
        return replace(
            self,
            dx_px=float(np.clip(self.dx_px, -width, width)),
            dy_px=float(np.clip(self.dy_px, -height, height)),
            keystone=float(np.clip(self.keystone, -k_max, k_max)),
            roll_deg=float(np.clip(self.roll_deg, -r_max, r_max)),
        )

    @property
    def is_identity(self) -> bool:
        return not any(asdict(self).values())

    def viewer_matrix(self, width: int, height: int) -> np.ndarray:
        """운전자가 보는 방향의 픽셀 좌표에서 쓰는 3x3 보정 행렬."""
        cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
        s = max(1.0, (height - 1) / 2.0)
        to_unit = np.array([[1 / s, 0, -cx / s], [0, 1 / s, -cy / s], [0, 0, 1]])
        from_unit = np.linalg.inv(to_unit)
        k = self.keystone
        keystone = np.array([[1, 0, 0], [0, 1, -k], [0, k, 1 - k]])
        a = math.radians(self.roll_deg)
        # 영상 좌표는 y 가 아래로 자라므로 이 행렬은 화면에서 시계 방향이다
        roll = np.array([[math.cos(a), -math.sin(a), 0],
                         [math.sin(a), math.cos(a), 0],
                         [0, 0, 1]])
        move = np.array([[1, 0, self.dx_px / s], [0, 1, self.dy_px / s], [0, 0, 1]])
        return from_unit @ move @ roll @ keystone @ to_unit

    def panel_matrix(
        self, width: int, height: int, flip_h: bool, flip_v: bool
    ) -> np.ndarray:
        """패널 픽셀 좌표에서 쓰는 보정 행렬.

        반사 광학계가 상을 뒤집으므로 패널에서의 이동은 운전자가 보는
        이동과 방향이 다르다. 물리적 반전 F 로 감싸 F · V · F 로 바꾼다.
        그래야 키를 오른쪽으로 누르면 운전자 눈에도 오른쪽으로 간다.
        반전은 화면 중앙에 대해 대칭이라 사다리꼴 축과 회전 중심이 그대로다.
        """
        flip = np.diag([-1.0 if flip_h else 1.0, -1.0 if flip_v else 1.0, 1.0])
        flip[0, 2] = (width - 1) if flip_h else 0.0
        flip[1, 2] = (height - 1) if flip_v else 0.0
        return flip @ self.viewer_matrix(width, height) @ flip   # F 는 자기 역행렬


class AlignmentMap:
    """카메라 정규화 좌표를 HUD 패널 픽셀로 옮긴다."""

    def __init__(self, config: dict[str, Any]) -> None:
        display = config["display"]
        self.width = int(display["width"])
        self.height = int(display["height"])
        section = ensure_alignment_config(config)
        self.mode = str(section["mode"])
        self.horizon = float(section["camera_horizon_y"])
        self.pan = (int(section["pan_x"]), int(section["pan_y"]))
        self.pairs = list(section["pairs"])
        self.flip_horizontal = bool(display.get("flip_horizontal", False))
        self.flip_vertical = bool(display.get("flip_vertical", False))
        # 반사 광학계가 실제로 상을 뒤집는지. 대응쌍 모드에서는 반전이 행렬에
        # 흡수되어 위 두 값이 꺼지지만, 운전자 기준 방향을 계산하는 쪽
        # (트림, 상태 바 위치) 은 물리적 반전을 알아야 한다.
        self.physical_flip_horizontal = self.flip_horizontal
        self.physical_flip_vertical = self.flip_vertical

        self.horizon_margin = float(section.get("horizon_margin", 0.18))
        self.profiles = list(section.get("profiles", []))
        self.blend = float(section.get("blend", 0.0))
        self.pairs = blended_pairs(self.pairs, self.profiles, self.blend)
        if self.mode == "correspondence" and len(self.pairs) >= MIN_PAIRS:
            self.matrix = solve_alignment(self.pairs)
            self.w_floor = vanishing_floor(
                self.pairs, self.matrix, self.horizon_margin
            )
            # 대응쌍에는 반사로 인한 좌우 반전이 이미 포함되어 있다.
            self.flip_horizontal = False
            self.flip_vertical = False
            self.calibrated = True
        else:
            self.matrix = quad_matrix(
                config["destination_quad_normalized"], self.width, self.height
            )
            self.w_floor = 1e-6
            self.calibrated = False
        self.set_trim(TrimParams.from_config(section))

    def set_trim(self, trim: TrimParams) -> None:
        """미세 보정만 바꾼다. 기본 행렬은 다시 풀지 않는다.

        렌더러를 rebuild 하지 않고 이것만 부르면 되므로 트림 도구가 키를
        누를 때마다 부트 애니메이션이 다시 도는 일이 없다.
        """
        self.trim = trim.clamped(self.width, self.height)
        self.trim_matrix = None
        if not self.trim.is_identity:
            self.trim_matrix = self.trim.panel_matrix(
                self.width, self.height,
                self.physical_flip_horizontal, self.physical_flip_vertical,
            )

    def project(self, points: Any) -> tuple[np.ndarray, np.ndarray]:
        """정규화 좌표 배열을 패널 픽셀과 유효 여부로 돌려준다."""
        array = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        if array.size == 0:
            return np.zeros((0, 2), np.int32), np.zeros((0,), bool)

        stacked = np.hstack([array, np.ones((len(array), 1))])
        moved = stacked @ self.matrix.T
        w = moved[:, 2]
        # w 가 0 에 가까우면 소실선 근처라 좌표가 발산한다.
        valid = w > self.w_floor
        uv = np.zeros((len(array), 2), dtype=np.float64)
        uv[valid] = moved[valid, :2] / w[valid, None]

        if self.flip_horizontal:
            uv[:, 0] = (self.width - 1) - uv[:, 0]
        if self.flip_vertical:
            uv[:, 1] = (self.height - 1) - uv[:, 1]
        uv[:, 0] += self.pan[0]
        uv[:, 1] += self.pan[1]

        # 미세 보정은 최종 패널 좌표 위에 마지막으로 곱한다. 기본 행렬의
        # w 로 소실선 판정을 끝낸 뒤라 그 판정은 트림과 무관하게 유지된다.
        # 사다리꼴은 화면에서 아주 먼 점의 w 를 뒤집을 수 있어 따로 거른다.
        if self.trim_matrix is not None:
            moved = np.hstack([uv, np.ones((len(uv), 1))]) @ self.trim_matrix.T
            w = moved[:, 2]
            valid &= w > 0.05
            uv[valid] = moved[valid, :2] / w[valid, None]

        # 정수 변환 시 넘침을 막기 위해 화면에서 아주 먼 점은 잘라 낸다.
        limit = 10 * max(self.width, self.height)
        valid &= np.all(np.abs(uv) < limit, axis=1)
        return np.rint(uv).astype(np.int32), valid

    def clip_above_horizon(self, lane: Any) -> np.ndarray:
        """지평선 위쪽 점은 버린다. 노면 평면 가정이 깨지는 구간이다.

        경계를 넘나드는 선분은 지평선과의 교점을 끼워 넣어 선이 지평선
        직전에서 끊기지 않게 한다. 점만 걷어 내면 남은 마지막 점과 지평선
        사이가 비어 차선이 짧아 보인다.
        """
        array = np.asarray(lane, dtype=np.float64).reshape(-1, 2)
        if self.horizon <= 0.0 or len(array) == 0:
            return array
        bounds = (-1e9, self.horizon, 1e9, 1e9)
        clipped = _clip_polyline(array.tolist(), bounds)
        return np.asarray(clipped, dtype=np.float64).reshape(-1, 2)


def quad_matrix(
    quad_normalized: Any, width: int, height: int
) -> np.ndarray:
    """단위 사각형을 패널 위 사각형으로 보내는 행렬 (alignment.mode = quad)."""
    source = np.asarray([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    quad = np.asarray(quad_normalized, dtype=np.float32).reshape(4, 2)
    target = np.column_stack([quad[:, 0] * (width - 1), quad[:, 1] * (height - 1)])
    return cv2.getPerspectiveTransform(source, target.astype(np.float32)).astype(
        np.float64
    )


def solve_alignment(pairs: list[dict[str, Any]]) -> np.ndarray:
    """대응쌍에서 카메라 -> 패널 호모그래피를 구한다."""
    if len(pairs) < MIN_PAIRS:
        raise ValueError(f"need at least {MIN_PAIRS} pairs, got {len(pairs)}")
    source = np.asarray([p["camera"] for p in pairs], dtype=np.float64)
    target = np.asarray([p["panel"] for p in pairs], dtype=np.float64)
    matrix, _ = cv2.findHomography(source, target, method=0)
    if matrix is None:
        raise ValueError("could not solve alignment from these points")
    # 부호를 맞춰 유효 영역에서 w 가 양수가 되게 한다.
    center = np.append(source.mean(axis=0), 1.0)
    if float((matrix @ center)[2]) < 0:
        matrix = -matrix
    return matrix / matrix[2, 2] if abs(matrix[2, 2]) > 1e-12 else matrix


def blended_pairs(
    pairs: list[dict[str, Any]], profiles: list[dict[str, Any]], blend: float
) -> list[dict[str, Any]]:
    """눈높이가 다른 두 사람의 보정 사이를 섞는다.

    프로필이 두 개 이상 있으면 패널 쪽 점을 선형 보간한 뒤 행렬을 다시 푼다.
    운전자가 바뀔 때 전체 보정을 반복하지 않기 위한 장치다.
    """
    if len(profiles) < 2 or not pairs:
        return pairs
    first = np.asarray(profiles[0]["panel"], dtype=np.float64)
    second = np.asarray(profiles[1]["panel"], dtype=np.float64)
    if first.shape != second.shape or len(first) != len(pairs):
        return pairs
    ratio = float(np.clip(blend, 0.0, 1.0))
    mixed = first + (second - first) * ratio
    return [
        {"camera": pair["camera"], "panel": mixed[index].tolist()}
        for index, pair in enumerate(pairs)
    ]


def vanishing_floor(
    pairs: list[dict[str, Any]], matrix: np.ndarray, margin: float
) -> float:
    """소실선에 얼마나 가까운 점까지 그릴지 정하는 하한.

    보정에 쓴 표식들의 w 값을 기준 삼아 자동으로 잡는다. 카메라 장착 각도가
    달라져도 지평선 높이를 손으로 다시 넣을 필요가 없다.
    """
    source = np.asarray([p["camera"] for p in pairs], dtype=np.float64)
    stacked = np.hstack([source, np.ones((len(source), 1))])
    values = stacked @ matrix.T
    positive = values[:, 2][values[:, 2] > 0]
    if positive.size == 0:
        return 1e-6
    return max(1e-6, float(positive.min()) * float(margin))


def alignment_error(pairs: list[dict[str, Any]], matrix: np.ndarray) -> list[float]:
    """대응쌍마다 몇 픽셀 어긋나는지 돌려준다."""
    errors = []
    for pair in pairs:
        point = np.append(np.asarray(pair["camera"], dtype=np.float64), 1.0)
        moved = matrix @ point
        if abs(moved[2]) < 1e-9:
            errors.append(float("inf"))
            continue
        projected = moved[:2] / moved[2]
        errors.append(float(np.linalg.norm(projected - np.asarray(pair["panel"]))))
    return errors
