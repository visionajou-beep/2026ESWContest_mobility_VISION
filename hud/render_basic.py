# -*- coding: utf-8 -*-
"""기본 렌더러 — 흰 선 두 개와 글자만 그리는 단순 화면 (ui.style 이 ar_overlay 가 아닐 때).
"""
from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from .align import AlignmentMap, ensure_alignment_config
from .config import ensure_ui_config
from .receiver import _latency_line
from .render import STATE_LOST, normalize_state


class HudRenderer:
    """정규화 좌표를 받아 HUD 한 장을 그린다."""

    def __init__(self, config: dict[str, Any]) -> None:
        display = config["display"]
        self.width = int(display["width"])
        self.height = int(display["height"])
        ensure_alignment_config(config)
        self.mapper = AlignmentMap(config)
        self.ui = ensure_ui_config(config)
        self.rect = (0, 0, self.width, self.height)
        self.static_layer = self._build_static_layer()
        self.state = 0
        self.lane_state = 0
        self.lane_opacity = 1.0

    @property
    def calibrated(self) -> bool:
        """운전자 시점 정렬이 실측으로 잡혀 있는지."""
        return self.mapper.calibrated

    # 좌표 변환 -------------------------------------------------------------

    def _road_points(self, lane: Any) -> tuple[np.ndarray, np.ndarray]:
        """노면 위 차선 좌표를 패널 픽셀로. 지평선 위쪽은 버린다."""
        clipped = self.mapper.clip_above_horizon(lane)
        return self.mapper.project(clipped)

    def _panel_point(self, x: float, y: float) -> tuple[int, int]:
        """화면 고정 요소용. 안전 영역 안의 비율 좌표를 픽셀로 옮긴다."""
        x0, y0, x1, y1 = (float(v) for v in self.ui["safe_area"])
        px = (x0 + x * (x1 - x0)) * (self.width - 1)
        py = (y0 + y * (y1 - y0)) * (self.height - 1)
        return int(round(px)), int(round(py))

    def _draw_clipped_polyline(
        self,
        canvas: np.ndarray,
        points: np.ndarray,
        valid: np.ndarray,
        color: tuple[int, int, int],
        thickness: int,
    ) -> None:
        """화면 밖으로 나가는 부분을 잘라 내며 선을 잇는다."""
        joint = max(1, thickness // 2)
        for index in range(len(points) - 1):
            if not (valid[index] and valid[index + 1]):
                continue
            first = (int(points[index][0]), int(points[index][1]))
            second = (int(points[index + 1][0]), int(points[index + 1][1]))
            inside, clipped_a, clipped_b = cv2.clipLine(self.rect, first, second)
            if not inside:
                continue
            cv2.line(canvas, clipped_a, clipped_b, color, thickness, cv2.LINE_AA)
            if 0 < index < len(points) - 1:
                cv2.circle(canvas, clipped_a, joint, color, -1, cv2.LINE_AA)

    # 정적 요소 -------------------------------------------------------------

    def _build_static_layer(self) -> np.ndarray:
        """매 프레임 다시 그릴 필요가 없는 요소를 한 번만 그려 캐시한다."""
        layer = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        tick = self.ui["center_tick"]
        if bool(tick["enabled"]):
            half = float(tick["width"]) / 2.0
            y = float(tick["y"])
            left = self._panel_point(0.5 - half, y)
            right = self._panel_point(0.5 + half, y)
            cv2.line(
                layer,
                left,
                right,
                tuple(int(v) for v in tick["color_bgr"]),
                int(tick["thickness"]),
                cv2.LINE_AA,
            )
        return layer

    def rebuild(self, config: dict[str, Any]) -> None:
        """보정값이나 레이아웃이 바뀐 뒤 캐시를 다시 만든다."""
        self.__init__(config)

    # 개별 요소 -------------------------------------------------------------

    def _draw_lanes(
        self,
        canvas: np.ndarray,
        lanes: list[list[list[float]]],
        emphasis_side: str | None,
    ) -> None:
        style = self.ui["lane"]
        color = tuple(int(v * self.lane_opacity) for v in style["color_bgr"])
        base = int(style["thickness"])
        strong = int(style["emphasis_thickness"])
        for lane in lanes:
            if len(lane) < 2:
                continue
            bottom_x = float(lane[-1][0])
            thickness = base
            if emphasis_side == "left" and bottom_x <= 0.5:
                thickness = strong
            elif emphasis_side == "right" and bottom_x > 0.5:
                thickness = strong
            points, valid = self._road_points(lane)
            if len(points) < 2:
                continue
            self._draw_clipped_polyline(canvas, points, valid, color, thickness)

    def _chevron_points(
        self, center_x: float, center_y: float, pointing: str
    ) -> np.ndarray:
        """꺾쇠 하나의 세 점을 정규화 좌표로 만든다."""
        style = self.ui["warning"]
        half_w = float(style["chevron_width"]) / 2.0
        half_h = float(style["chevron_height"]) / 2.0
        tip_x = center_x + (half_w if pointing == "right" else -half_w)
        back_x = center_x - (half_w if pointing == "right" else -half_w)
        return np.asarray(
            [
                [back_x, center_y - half_h],
                [tip_x, center_y],
                [back_x, center_y + half_h],
            ],
            dtype=np.float32,
        )

    def _draw_warning(
        self, canvas: np.ndarray, warning: str, elapsed: float
    ) -> None:
        style = self.ui["warning"]
        if not bool(style["enabled"]) or warning == "none":
            return

        side = "left" if warning.endswith("_left") else "right"
        departure = warning.startswith("departure")
        # 이탈은 중앙으로 돌아오라는 뜻이라 안쪽을, 변경은 진행 방향인 바깥쪽을 가리킨다.
        if departure:
            pointing = "right" if side == "left" else "left"
        else:
            pointing = "left" if side == "left" else "right"

        if departure and math.sin(elapsed * 2.0 * math.pi * float(style["blink_hz"])) < 0:
            return

        color = tuple(int(v) for v in style["color_bgr"])
        thickness = int(style["thickness"])
        count = int(style["chevron_count"])
        gap = float(style["chevron_gap"]) + float(style["chevron_width"])
        margin = float(style["side_margin"])
        center_y = float(style["center_y"])
        base_x = margin if side == "left" else 1.0 - margin
        half = (count - 1) / 2.0

        active = -1
        if not departure:
            active = int(elapsed * float(style["flow_hz"]) * count) % count
            if pointing == "left":
                active = count - 1 - active

        for index in range(count):
            center_x = base_x + (index - half) * gap
            points = self._chevron_points(center_x, center_y, pointing)
            weight = thickness
            if active >= 0 and index != active:
                weight = max(2, thickness - 4)
            panel = np.asarray(
                [self._panel_point(px, py) for px, py in points], dtype=np.int32
            )
            cv2.polylines(canvas, [panel], False, color, weight, cv2.LINE_AA)

        if bool(style["label"]):
            text = "DEPARTURE" if departure else "LANE CHANGE"
            self._draw_centered_text(
                canvas,
                text,
                float(style["label_y"]),
                float(style["label_scale"]),
                color,
                2,
            )

    def _draw_centered_text(
        self,
        canvas: np.ndarray,
        text: str,
        y: float,
        scale: float,
        color: tuple[int, int, int],
        thickness: int,
    ) -> None:
        font = cv2.FONT_HERSHEY_DUPLEX
        (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
        anchor_x, anchor_y = self._panel_point(0.5, y)
        origin = (int(anchor_x - text_w / 2), int(anchor_y + text_h / 2))
        cv2.putText(canvas, text, origin, font, scale, color, thickness, cv2.LINE_AA)

    def _draw_debug(self, canvas: np.ndarray, info: dict[str, Any]) -> None:
        style = self.ui["debug"]
        if not bool(style["enabled"]):
            return
        color = tuple(int(v) for v in style["color_bgr"])
        scale = float(style["scale"])
        thickness = int(style["thickness"])
        origin_x, origin_y = style["origin"]
        gap = float(style["line_gap"])
        lines = [
            "STATUS {status}  LANES {lanes}  SEQ {seq}".format(**info),
            "FPS {fps:.1f}  INFER {inference_ms:.1f}ms  AGE {age_ms:.0f}ms".format(**info),
            "RENDER {render_ms:.1f}ms  BAD {dropped}  WARN {warning}".format(**info),
            _latency_line(info),
            f"STATE {self.state}",
            "ALIGN {align}".format(align="driver-calibrated" if self.mapper.calibrated else "UNCALIBRATED (quad fallback)"),
        ]
        for index, line in enumerate(lines):
            point = self._panel_point(float(origin_x), float(origin_y) + gap * index)
            cv2.putText(
                canvas,
                line,
                point,
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                color,
                thickness,
                cv2.LINE_AA,
            )

    # 한 프레임 -------------------------------------------------------------

    def render(
        self,
        *,
        lanes: list[list[list[float]]],
        warning: str = "none",
        elapsed: float = 0.0,
        debug: dict[str, Any] | None = None,
        telemetry: dict[str, Any] | None = None,
        state: int = 0,
        lane_state: int | None = None,
        lane_opacity: float = 1.0,
    ) -> np.ndarray:
        # 지금은 받아서 보관만 한다
        self.state = normalize_state(state)
        # 차선 색과 밝기는 상태 바 state 와 따로 받는다. ThemeRenderer 와
        # 인터페이스를 맞춰 두어야 호출부가 렌더러를 갈아끼울 수 있다.
        self.lane_state = (
            self.state if lane_state is None else normalize_state(lane_state)
        )
        self.lane_opacity = float(min(1.0, max(0.0, lane_opacity)))
        canvas = self.static_layer.copy()
        emphasis = None
        if warning.startswith("departure"):
            emphasis = "left" if warning.endswith("_left") else "right"
        if self.lane_state != STATE_LOST and self.lane_opacity > 1.0 / 255.0:
            self._draw_lanes(canvas, lanes, emphasis)
        self._draw_warning(canvas, warning, elapsed)
        if debug is not None:
            self._draw_debug(canvas, debug)
        return canvas
