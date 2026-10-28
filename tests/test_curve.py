"""급커브 시험 — 좌우는 차 가까이에서 판단한다.

선의 위치를 기준행이 아니라 차 가까이(마스크 아래 끝 바로 위)에서 읽고, 곧은 도로처럼 기준행으로 옮긴다.
급커브에서는 선이 차와 기준행 사이에서 진행 경로를 넘어가므로, 기준행에서 읽으면 차 왼쪽의 선이 오른쪽 선이 된다.
"""

from __future__ import annotations

import numpy as np
import pytest

import postprocess.tracker as E
from postprocess import EgoLaneTracker
from postprocess.config import CROSS_MARGIN, REF_ROW_BELOW_HORIZON
from postprocess.geometry import vp_together, x_at_row

FPS = 30.0
VP = (0.5, 0.45)


def ribbon(xs_of_y, y0: float, y1: float = 1.0, w: float = 0.008, n: int = 12) -> np.ndarray:
    ys = np.linspace(y0, y1, n)
    xs = np.array([xs_of_y(y) for y in ys])
    return np.asarray(np.r_[np.c_[xs - w, ys], np.c_[xs + w, ys][::-1]], dtype=np.float32)


def straight(x_bottom: float, y0: float = 0.55) -> np.ndarray:
    return ribbon(lambda y: VP[0] + (x_bottom - VP[0]) * (y - VP[1]) / (1.0 - VP[1]), y0)


def bend_right(y):
    # 차 가까이에서는 왼쪽(아래 x 0.30), 위로 갈수록 오른쪽으로 휜다 — 기준행(0.645)에서는 경로(0.5)의 오른쪽
    t = (1.0 - y) / (1.0 - 0.55)
    return 0.30 + 0.42 * t ** 1.6


@pytest.fixture
def no_vp_test(monkeypatch):
    monkeypatch.setattr(E, "VP_MISS_MAX", 9.0)      # 소실점 검사는 이 테스트의 대상이 아니다


def run(frame, n=12):
    tr = EgoLaneTracker(FPS, hold_s=0.15, miss_s=0.30)
    tr.vp, tr._vp_n = VP, 10_000
    for _ in range(n):
        out = tr.update(frame())
    return out, tr


def test_a_line_that_bends_across_the_path_keeps_its_side_near_the_car(no_vp_test):
    frame = lambda: [ribbon(bend_right, 0.55), straight(0.90)]
    on, _ = run(frame)
    curved_ref = x_at_row(frame()[0], on.ref_row)
    assert curved_ref > on.center_x + CROSS_MARGIN             # 기준행에서 읽으면 경로 오른쪽이다
    assert on.ego_left is not None and on.ego_right is not None
    assert on.ego_left.x < on.center_x < on.ego_right.x
    assert abs(x_at_row(on.ego_left.poly, 0.9) - bend_right(0.9)) < 0.01   # 왼쪽 선은 그 휜 선


def test_on_a_straight_road_the_position_is_the_reference_row_reading():
    on, _ = run(lambda: [straight(0.15), straight(0.85)])
    assert abs(on.ego_left.x - x_at_row(straight(0.15), on.ref_row)) < 2e-3
    assert abs(on.ego_right.x - x_at_row(straight(0.85), on.ref_row)) < 2e-3


def test_a_mask_ending_above_the_near_rows_keeps_the_reference_reading():
    tr = EgoLaneTracker(FPS)
    tr.vp, tr._vp_n = VP, 10_000
    tr.ref_row = tr._reference_row()
    q = straight(0.20, y0=0.55)[:]
    q = q[q[:, 1] <= tr.ref_row + 0.01]                          # 기준행 바로 아래에서 끝나는 조각
    assert tr._near_equiv(q, 0.123, 0.5) == 0.123


# 차 가까이에서 읽을 때의 예외 경우 · 커브 보정의 안전장치

REF = VP[1] + REF_ROW_BELOW_HORIZON * (1.0 - VP[1])


def test_keep_tracked_does_not_spare_a_slanted_mark_read_near_the_car():
    # 두 선을 따라가다가 왼쪽 선 옆에 기운 자국이 나타난다 (차 가까이에서 읽어 옮기면 왼쪽 선과 0.05 안)
    tr = EgoLaneTracker(FPS, hold_s=0.15, miss_s=0.30)
    tr.vp, tr._vp_n = VP, 10_000
    for _ in range(15):
        tr.update([straight(0.15), straight(0.85)])
    left_id = tr.committed["ego_left"]
    mark = ribbon(lambda y: 0.47 - 1.2 * (y - REF), 0.70, 0.95)
    for _ in range(20):
        out = tr.update([straight(0.15), straight(0.85), mark])
        assert out.diag["vp"] >= 1                                   # 자국은 소실점 검사에서 떨어진다
        assert out.ego_left.track_id == left_id
        assert float(out.ego_left.poly[:, 1].min()) < 0.6            # 왼쪽은 진짜 선의 폴리곤


def test_one_curved_line_in_two_pieces_is_not_both_ego_lines(no_vp_test):
    near_piece = ribbon(bend_right, 0.66, 1.0)                       # 차까지 내려옴 — 차 가까이에서 읽으면 경로 왼쪽
    far_piece = ribbon(bend_right, 0.50, 0.64)                       # 기준행 위에서 끝남 — 기준행에서 읽으면 경로 오른쪽
    out, _ = run(lambda: [near_piece, far_piece], n=20)
    assert out.ego_left is None or out.ego_right is None


def test_vp_together_safeguards():
    tog = vp_together
    assert tog(0.40, 0.12, [(0.62, 0.12)], 0.03, 0.5)                # 경로 반대쪽 짝: 커브
    assert not tog(0.30, 0.12, [(0.40, 0.12)], 0.03, 0.5)            # 같은 쪽 짝 (옆 차로 화살표 · 빗금)
    assert not tog(0.40, 0.12, [(0.62, 0.12), (0.70, 0.01)], 0.03, 0.5)   # 소실점으로 모이는 선이 있다 — 곧은 도로
    assert tog(0.40, 0.12, [(0.62, 0.149)], 0.03, 0.5) and not tog(0.40, 0.12, [(0.62, 0.151)], 0.03, 0.5)
    assert not tog(0.40, 0.12, [(0.459, 0.12)], 0.03, 0.43) and tog(0.40, 0.12, [(0.461, 0.12)], 0.03, 0.43)


def test_the_two_strokes_of_a_double_line_under_the_car_are_not_the_ego_pair():
    # 차가 이중 황색선을 밟고 있을 때, 오른쪽 획을 따라가던 중 왼쪽 획이 경로 위에 새로 나타나도
    # 둘을 자차 좌 · 우 쌍으로 보지 않는다 — 두 획은 한 선이지 폭 0.5 m 차로가 아니다
    tr = EgoLaneTracker(FPS, hold_s=0.15, miss_s=0.30)
    tr.vp, tr._vp_n = VP, 10_000
    for _ in range(15):
        tr.update([straight(0.60), straight(0.95)])
    for _ in range(20):
        out = tr.update([straight(0.54), straight(0.60), straight(0.95)])
        if out.ego_left is not None and out.ego_right is not None:
            assert out.ego_right.x - out.ego_left.x > 0.06
