"""소실점 검사의 커브 보정 시험.

소실점은 곧은 도로의 선이 모이는 점이라, 커브에서는 진짜 선도 소실점을 비켜 간다.
같은 프레임의 다른 선과 함께 비켜 가면 커브로 보고 버리지 않는다. 화살표 · 연석 같은 표시는 혼자 비켜 간다.
"""

from __future__ import annotations

import numpy as np

from postprocess import EgoLaneTracker
from postprocess.config import VP_TOGETHER
from postprocess.lane import STATE_OK

FPS = 30.0
VP = (0.5, 0.45)


def strip(x_bottom: float, x_at_vp: float, *, y0: float = 0.55, w: float = 0.008) -> np.ndarray:
    """(x_bottom, 1.0) 에서 소실점 높이의 x_at_vp 를 향하는 곧은 띠, 위 끝 y0."""
    xt = x_bottom + (x_at_vp - x_bottom) * (1.0 - y0) / (1.0 - VP[1])
    return np.asarray([[x_bottom - w, 1.0], [xt - w, y0], [xt + w, y0], [x_bottom + w, 1.0]], dtype=np.float32)


def run(frame, n=15):
    tr = EgoLaneTracker(FPS, hold_s=0.15, miss_s=0.30)
    tr.vp, tr._vp_n = VP, 10_000            # 소실점을 이미 배운 상태에서 시작
    for _ in range(n):
        out = tr.update(frame())
    return out


def curve():
    # 두 선 모두 소실점 높이에서 0.11 오른쪽을 향한다 (검사 기준 0.10 초과) — 오른쪽으로 도는 커브
    return [strip(0.10, VP[0] + 0.11), strip(0.90, VP[0] + 0.11)]


def test_lines_missing_together_are_a_curve():
    out = run(curve)
    assert out.diag["vp"] == 0 and out.diag.get("vp_together_kept") == 2
    assert out.state == STATE_OK
    assert out.display_line(out.ego_left) is not None and out.display_line(out.ego_right) is not None


def test_a_mark_missing_alone_is_still_dropped():
    # 두 선은 소실점으로 모이고, 왼쪽 선 옆의 자국 하나만 0.15 비켜 간다
    frame = lambda: [strip(0.10, VP[0]), strip(0.90, VP[0]), strip(0.30, VP[0] - 0.15, y0=0.62)]
    out = run(frame)
    assert out.diag["vp"] == 1 and not out.diag.get("vp_together_kept")


def test_pieces_of_one_mark_do_not_vouch_for_each_other():
    # 같은 자리(MIN_LANE_GAP 안)의 두 조각이 같이 비켜 간다 — 한 자국이지 커브가 아니다
    frame = lambda: [strip(0.10, VP[0]), strip(0.90, VP[0]),
                     strip(0.30, VP[0] - 0.15, y0=0.62), strip(0.31, VP[0] - 0.15, y0=0.62)]
    out = run(frame)
    assert out.diag["vp"] == 2 and not out.diag.get("vp_together_kept")


def test_misses_farther_apart_than_the_tolerance_do_not_count():
    assert VP_TOGETHER == 0.03
    frame = lambda: [strip(0.10, VP[0] + 0.11), strip(0.90, VP[0] + 0.16)]   # 0.05 차 > 0.03
    assert run(frame).diag["vp"] == 2
