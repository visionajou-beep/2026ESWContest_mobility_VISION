# -*- coding: utf-8 -*-
"""패널 진단 패턴 — 레터박스 · 반전 · 선 굵기 · 안전 영역을 한 화면에서 확인한다.
"""
from __future__ import annotations

import cv2
import numpy as np

from .render import DESIGN_H, DESIGN_W, STATE_COLORS, STATE_NORMAL, ThemeRenderer


# ---------------------------------------------------------------------------
# pattern: 패널이 제대로 그려지는지 눈으로 보는 진단 패턴
# ---------------------------------------------------------------------------
#
# 배경이 검정이고 mock 차선이 좌우 대칭이라 모니터를 바꿨을 때 레터박스가
# 있는지, 반전이 걸렸는지 알 수가 없다. 그걸 한 화면에서 판정한다.
#
#   화면 경계(흰 1px)와 렌더 영역 경계(하늘색)가 붙어 있으면 레터박스 없음.
#   벌어진 만큼이 레터박스다. 16:10 패널이면 위아래로 벌어진다.
#   L/R/TOP/BOTTOM 글자와 좌상 → 우하 대각선이 비대칭 기준이다. 글자가
#   거울상이면 그 축으로 반전이 걸려 있다.
#   자홍색 사각형이 안전 영역이다. 반사판 앞 운전 자세에서 네 변이 다
#   보여야 하고, 그 안쪽 녹색 띠가 상태 바가 실제로 그려지는 자리다.

PATTERN_GRID_PX = 100


# BGR. 요소마다 색을 달리 둬야 겹친 자리를 구분할 수 있다. 반사식 제약대로
# 전부 얇은 선과 글리프고, 큰 밝은 면적은 없다.
PATTERN_COLORS = {
    "border": (255, 255, 255),     # 흰색     화면 경계
    "design": (255, 255, 0),       # 하늘색   렌더(디자인 960x540) 영역 경계
    "grid": (70, 70, 70),          # 짙은 회색 격자
    "diagonal": (0, 255, 255),     # 노란색   비대칭 기준 대각선
    "cross": (120, 255, 120),      # 녹색     중심 십자
    "label": (255, 255, 255),      # 흰색     방향 글자
    "info": (0, 170, 255),         # 주황색   반전을 타지 않는 안내 블록
    "safe": (255, 0, 255),         # 자홍색   안전 영역 경계
}


def _pattern_text(
    canvas: np.ndarray,
    text: str,
    center: tuple[float, float],
    scale: float,
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    """문자열을 중심 좌표에 맞춰 그린다."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
    origin = (int(round(center[0] - text_w / 2)), int(round(center[1] + text_h / 2)))
    cv2.putText(canvas, text, origin, font, scale, color, thickness, cv2.LINE_AA)


def _flip_code(flip_h: bool, flip_v: bool) -> int | None:
    """cv2.flip 코드. 반전이 없으면 None."""
    if flip_h and flip_v:
        return -1
    if flip_h:
        return 1
    if flip_v:
        return 0
    return None


def _lane_widths(renderer: ThemeRenderer) -> list[tuple[str, int]]:
    """차선이 실제로 그려지는 굵기 세 가지를 픽셀로."""
    return [
        (name, renderer._ratio_px(renderer.theme[key]))
        for name, key in (
            ("EDGE", "edge_width_ratio"),
            ("ALERT", "alert_edge_width_ratio"),
            ("DIM", "dim_edge_width_ratio"),
        )
    ]


def _draw_pattern_body(
    renderer: ThemeRenderer, canvas: np.ndarray, grid: bool
) -> None:
    """반전을 타는 쪽. 차선과 같은 방향으로 뒤집혀야 하는 것만 여기 그린다."""
    width, height = renderer.width, renderer.height
    scale = renderer.scale
    last_x, last_y = width - 1, height - 1

    # 5. 격자 100px. 가장 어두운 색으로 맨 아래에 깐다.
    if grid:
        for x in range(0, width, PATTERN_GRID_PX):
            cv2.line(canvas, (x, 0), (x, last_y), PATTERN_COLORS["grid"], 1)
        for y in range(0, height, PATTERN_GRID_PX):
            cv2.line(canvas, (0, y), (last_x, y), PATTERN_COLORS["grid"], 1)

    # 2. 렌더 영역 경계. 디자인 좌표 960x540 이 패널에서 차지하는 사각형이다.
    #    ThemeRenderer 가 차선을 올릴 때 쓰는 _dx/_dy 를 그대로 쓴다.
    dx0, dy0 = renderer._dp(0.0, 0.0)
    dx1 = min(last_x, int(round(renderer._dx(DESIGN_W))) - 1)
    dy1 = min(last_y, int(round(renderer._dy(DESIGN_H))) - 1)
    cv2.rectangle(canvas, (dx0, dy0), (dx1, dy1), PATTERN_COLORS["design"], 2)

    # 4. 비대칭 기준선. 디자인 좌표의 좌상 → 우하.
    cv2.line(canvas, (dx0, dy0), (dx1, dy1), PATTERN_COLORS["diagonal"], 2,
             cv2.LINE_AA)

    # 6. 중심 십자. 패널 중심이지 디자인 중심이 아니다. 둘이 어긋나면
    #    렌더 영역 사각형의 대각선 교점과 벌어진 만큼이 보인다.
    cx, cy = last_x // 2, last_y // 2
    arm = int(40 * scale)
    cv2.line(canvas, (cx - arm, cy), (cx + arm, cy), PATTERN_COLORS["cross"], 1,
             cv2.LINE_AA)
    cv2.line(canvas, (cx, cy - arm), (cx, cy + arm), PATTERN_COLORS["cross"], 1,
             cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), int(10 * scale), PATTERN_COLORS["cross"], 1,
               cv2.LINE_AA)

    # 3. 방향 글자. 디자인 좌표 기준이라 반전이 걸리면 거울상으로 보인다.
    #    TOP/BOTTOM 은 디자인 위아래 끝에 바짝 붙인다. 가운데로 내려오면
    #    반전됐을 때 아래쪽 안내 블록 자리로 들어간다.
    big = 1.1 * scale
    label = PATTERN_COLORS["label"]
    _pattern_text(canvas, "L", renderer._dp(40.0, DESIGN_H / 2), big, label, 3)
    _pattern_text(canvas, "R", renderer._dp(DESIGN_W - 40.0, DESIGN_H / 2), big,
                  label, 3)
    _pattern_text(canvas, "TOP", renderer._dp(DESIGN_W / 2, 28.0), big, label, 3)
    _pattern_text(canvas, "BOTTOM", renderer._dp(DESIGN_W / 2, DESIGN_H - 28.0),
                  big, label, 3)

    # 차선 굵기 샘플. 색은 state 0 차선과 같게 두고 굵기만 바꿔야 굵기
    # 하나만 보고 판단할 수 있다. 선 끝 모양도 폴리라인과 같게 LINE_AA.
    # 패널 좌표의 가장자리 쪽 세로 중앙 띠에 둔다. 반전이 걸리면 좌우로
    # 자리를 옮길 뿐 안내 블록이 있는 위쪽 띠로는 내려오지 않는다.
    x0 = int(round(width * 0.08))
    x1 = int(round(width * 0.26))
    for index, (name, pixels) in enumerate(_lane_widths(renderer)):
        y = int(round(height * (0.38 + 0.12 * index)))
        cv2.line(canvas, (x0, y), (x1, y), STATE_COLORS[STATE_NORMAL], pixels,
                 cv2.LINE_AA)
        _pattern_text(canvas, f"{name} {pixels}px",
                      ((x0 + x1) / 2.0, y - 24.0 * scale), 0.42 * scale, label, 2)


def _draw_pattern_frame(renderer: ThemeRenderer, canvas: np.ndarray) -> None:
    """반전을 타지 않는 쪽.

    1. 화면 경계는 패널 전체 사각형이라 반전해도 제자리다. 반전 뒤에 그려
       1px 이 뒤집기 과정에서 잘리지 않게 한다.
    2. 안내 글자는 반전되면 거울상이 되어 읽을 수가 없다. 패턴의 다른
       글자와 구분되도록 색을 따로 쓰고 NOT FLIPPED 라고 못박는다.
    """
    width, height = renderer.width, renderer.height
    scale = renderer.scale
    mapper = renderer.mapper

    # 1. 화면 경계 1px. 네 변이 다 보이면 오버스캔 없음.
    cv2.rectangle(canvas, (0, 0), (width - 1, height - 1),
                  PATTERN_COLORS["border"], 1)

    # 7. 안전 영역. 네 변에서 같은 여백이라 반전과 무관하게 제자리다.
    #    반사판 앞에서 운전 자세로 이 사각형 네 변이 다 보이는지 본다.
    #    아랫변이 안 보이면 theme.state_bar_margin_ratio 를 키운다.
    margin = renderer.safe_margin_px
    safe = PATTERN_COLORS["safe"]
    if margin > 0:
        cv2.rectangle(canvas, (margin, margin),
                      (width - 1 - margin, height - 1 - margin), safe, 2)

    # 8. 상태 바 실제 위치. 렌더러가 쓰는 행 범위를 그대로 칠한다.
    #    상태 바는 물리적 반전을 따르고 패턴 본체는 행렬에 남은 반전을
    #    따르므로, 본체에 그리면 대응쌍 보정 모드에서 자리가 어긋난다.
    bar_top, bar_bottom = renderer._state_bar_rows
    canvas[bar_top:bar_bottom] = STATE_COLORS[STATE_NORMAL]
    # 글자는 바에서 안전 영역 안쪽으로 띄운다. 가운데는 TOP/BOTTOM
    # 글자 자리라 왼쪽에 둔다.
    label_y = (bar_bottom + 22 * scale if bar_top < height / 2
               else bar_top - 22 * scale)
    _pattern_text(canvas, f"STATE BAR  safe margin {margin}px",
                  (width * 0.2, label_y), 0.42 * scale, safe, 2)

    # 설정값과 패턴이 실제로 탄 반전을 따로 적는다. 대응쌍 보정이 잡혀
    # 있으면 반전이 행렬에 흡수되어 둘이 갈라진다.
    def onoff(flag: bool) -> str:
        return "ON " if flag else "off"

    config_flip = (mapper.physical_flip_horizontal, mapper.physical_flip_vertical)
    applied_flip = (mapper.flip_horizontal, mapper.flip_vertical)
    widths = "  ".join(f"{name.lower()} {px}px" for name, px in _lane_widths(renderer))

    lines = [
        "TEST PATTERN -- THIS BLOCK IS NOT FLIPPED",
        f"PANEL {width}x{height}   DESIGN {int(DESIGN_W)}x{int(DESIGN_H)}"
        f" -> scale {renderer.scale:.3f}"
        f"  letterbox {renderer.offset_x:.0f},{renderer.offset_y:.0f}px",
        f"FLIP config  H {onoff(config_flip[0])} V {onoff(config_flip[1])}"
        f"    pattern rode  H {onoff(applied_flip[0])} V {onoff(applied_flip[1])}",
        f"ALIGN {'correspondence (flip absorbed in matrix)' if mapper.calibrated else 'quad (uncalibrated)'}",
        f"LANE WIDTH  {widths}",
        f"SAFE margin {renderer.safe_margin_px}px"
        f" (theme.state_bar_margin_ratio {float(renderer.theme['state_bar_margin_ratio']):.3f})"
        f"   state bar rows {bar_top}-{bar_bottom - 1}",
        "mirrored L/R or TOP/BOTTOM above = that axis is flipped",
    ]
    # 패턴이 비워 둔 위쪽 띠. 굵기 샘플(0.38H 아래)과 TOP/BOTTOM 글자
    # (디자인 끝단) 사이라 어느 쪽으로 반전되든 겹치지 않는다.
    text_scale = 0.42 * scale
    gap = int(21 * scale)
    top = int(height * 0.14)
    for index, line in enumerate(lines):
        _pattern_text(canvas, line, (width / 2.0, float(top + index * gap)),
                      text_scale, PATTERN_COLORS["info"], 2)


def build_test_pattern(renderer: ThemeRenderer, *, grid: bool = True) -> np.ndarray:
    """진단 패턴 한 장. 매 프레임 다시 그릴 이유가 없어 한 번만 만든다."""
    canvas = np.zeros((renderer.height, renderer.width, 3), np.uint8)
    _draw_pattern_body(renderer, canvas, grid)

    # 차선은 AlignmentMap.project() 안에서 x -> (w-1)-x, y -> (h-1)-y 를
    # 받는다. 프레임 전체 cv2.flip 이 그와 완전히 같은 변환이므로 패턴도
    # 차선과 같은 방향으로 뒤집힌다. pan 과 trim 은 태우지 않는다. 화면
    # 경계와 렌더 영역을 재는 패턴이라 평행이동이 끼면 기준이 흔들린다.
    code = _flip_code(renderer.mapper.flip_horizontal, renderer.mapper.flip_vertical)
    if code is not None:
        canvas = cv2.flip(canvas, code)

    _draw_pattern_frame(renderer, canvas)
    return canvas
