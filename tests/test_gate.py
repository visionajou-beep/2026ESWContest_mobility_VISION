"""자차 선 거리 제한 — 진행 경로에서 한 차로폭보다 먼 선은 자차 선이 되지 않는다.

모델이 진짜 자차 선을 놓쳤을 때 옆 차로 선을 대신 그리면 틀린 선이 된다.
거리 제한을 두면 그 쪽은 안 그리는 쪽(허용되는 실패)으로 간다.
"""

from __future__ import annotations

import numpy as np

from postprocess import EgoLaneTracker
from postprocess.geometry import assign_roles, x_at_row

FPS = 30.0


def strip(x: float, *, y0: float = 0.50, y1: float = 0.82, w: float = 0.008) -> np.ndarray:
    return np.asarray([[x - w, y0], [x - w, y1], [x + w, y1], [x + w, y0]], dtype=np.float32)


def run(tracker: EgoLaneTracker, frames):
    return [tracker.update([strip(x) for x in xs]) for xs in frames]


def roles(polys, max_dist=9.0, pair_w=0.32):
    """추적기처럼 부른다: 기준행 0.65 위치, 쪽은 모름(0). max_dist 9.0 = 거리 제한 없음."""
    return assign_roles(polys, [x_at_row(p, 0.65) for p in polys], [0] * len(polys), 0.5, 0.65,
                        max_dist=max_dist, pair_w=pair_w)


def test_max_dist_leaves_a_far_line_undrawn_instead_of_promoting_it():
    polys = [strip(0.05), strip(0.65)]          # 먼 왼쪽 선 + 진짜 오른쪽 선, 왼쪽 자차 선은 없음
    free = roles(polys)
    assert free.get("ego_left") == 0             # 거리 제한이 없으면 먼 선이 자차 선이 된다
    gated = roles(polys, max_dist=0.32)
    assert "ego_left" not in gated
    assert gated.get("ego_right") == 1


def test_max_dist_keeps_a_normal_ego_pair():
    polys = [strip(0.35), strip(0.65), strip(0.05), strip(0.95)]
    gated = roles(polys, max_dist=0.32)
    assert gated["ego_left"] == 0 and gated["ego_right"] == 1
    # 옆 차로 선은 자차 선이 되지 않지만, 차로변경 때 쓰도록 옆 선으로는 알려 준다
    assert gated["adj_left"] == 2 and gated["adj_right"] == 3
    # 한쪽 자차 선이 없으면 거리 제한 밖의 선은 그 자리를 채우지 않는다
    lone = roles([strip(0.65), strip(0.05)], max_dist=0.32)
    assert "ego_left" not in lone and "adj_left" not in lone


def test_tracker_gate_is_on():
    """추적기는 거리 제한(1 차로폭)을 항상 쓴다."""
    out_gated = run(EgoLaneTracker(FPS), [[0.05, 0.65]] * 10)
    assert out_gated[-1].ego_left is None            # 경로에서 0.45 — 자차 선이 아니다
    assert out_gated[-1].ego_right is not None


def test_gate_keeps_the_far_boundary_of_a_lane_change_pair():
    """차가 왼쪽 선(0.49) 위에 있고 오른쪽 선이 한 차로폭(0.83) 떨어진 경우 — 차로변경 중.
    거리 제한만 보면 0.83 은 빠지지만, 왼쪽 선과 한 차로폭 짝이므로 남긴다."""
    polys = [strip(0.49), strip(0.83)]
    paired = roles(polys, max_dist=0.32, pair_w=0.32)
    assert paired.get("ego_left") == 0 and paired.get("ego_right") == 1
    # 짝이 없는 먼 선 하나는 그래도 뺀다
    lone = roles([strip(0.90)], max_dist=0.32, pair_w=0.32)
    assert "ego_right" not in lone
