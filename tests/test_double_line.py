"""이중선 · 셋째 선 시험.

셋째 선: 차로변경 때 넘어가려는 자차 선에서 한 차로(0.7 ~ 1.3 차로폭) 옆 선만 그린다 (벽 · 빗금 끝 · 방호벽 너머 선은 아니다).
이중선: 두 획이 프레임마다 번갈아 한 마스크로 올 때 안쪽 획(차에 가까운 쪽)을 자차 선으로 쓴다.
"""

from __future__ import annotations

import numpy as np

from postprocess import EgoLaneTracker

FPS = 30.0
VP = (0.5, 0.45)


def straight(x_bottom: float, y0: float = 0.55, w: float = 0.008) -> np.ndarray:
    ys = np.linspace(y0, 1.0, 12)
    xs = VP[0] + (x_bottom - VP[0]) * (ys - VP[1]) / (1.0 - VP[1])
    return np.asarray(np.r_[np.c_[xs - w, ys], np.c_[xs + w, ys][::-1]], dtype=np.float32)


def tracker():
    tr = EgoLaneTracker(FPS, hold_s=0.15, miss_s=0.30)
    tr.vp, tr._vp_n = VP, 10_000
    return tr


def drift_left(adj_offset: float):
    """차가 왼쪽 선 쪽으로 붙어 간다 (선들이 화면에서 오른쪽으로 이동). 그 바깥 선은 왼쪽 선에서 adj_offset 떨어져 있다 (화면 아래 기준)."""
    tr = tracker()
    outs = []
    for k in range(60):
        lb = 0.20 + min(k, 40) / 40 * (0.42 - 0.20)
        outs.append(tr.update([straight(lb), straight(lb + 0.60), straight(lb - adj_offset)]))
    return outs


def test_x_one_lane_over_is_drawn():
    assert any(o.extra is not None for o in drift_left(0.60)[40:])


def test_x_on_a_line_two_lanes_out_is_not_drawn():
    # 두 차로 바깥(벽 쪽) 선은 셋째 선으로 그리지 않는다
    assert all(o.extra is None for o in drift_left(1.10))


def alternating(n: int = 40):
    """왼쪽 이중선: 안쪽 획(아래 0.28)과 바깥 획(아래 0.17)이 번갈아 한 마스크로 온다."""
    tr = tracker()
    xs = []
    for k in range(n):
        left = straight(0.28) if k % 3 != 1 else straight(0.17)
        out = tr.update([left, straight(0.85)])
        xs.append(out.ego_left.x)
    return np.array(xs), tr


def test_alternating_strokes_keep_the_inner_line():
    inner = 0.5 + (0.28 - 0.5) * (0.6447 - 0.45) / 0.55
    xs, _ = alternating()
    assert np.max(np.abs(xs[10:] - inner)) < 0.006                          # 바깥 획으로 튀지 않고 안쪽 획에 머문다


def test_one_jump_is_not_a_double_line():
    # 한 번 옮겨 가서 머문 선은 이중선으로 배우지 않는다
    tr = tracker()
    for k in range(40):
        tr.update([straight(0.20 if k < 20 else 0.26), straight(0.85)])
    left = tr._by_id[tr.committed["ego_left"]]
    assert left.stroke is None
