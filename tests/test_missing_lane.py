"""한쪽 선이 안 보일 때의 시험 — 한 줄 유지와 반대쪽 역산.

한 줄 유지: 한쪽을 잃어도, 직전까지 한 차로폭 짝의 한쪽이었고 그 뒤 옆으로 튀지 않은 선은 최대 1.2 초 그린다 (State 1).
혼자 남은 선은 짝 간격으로 확인할 수 없으므로 이 조건을 만족하는 선만 남긴다.

역산: 한 줄만 그릴 때 반대쪽 선을 마지막으로 함께 본 쌍에서 옮겨 그린다.
"""

from __future__ import annotations

import numpy as np

from postprocess import EgoLaneTracker
from postprocess.config import INFER_SEC, ONE_SIDE_SEC
from postprocess.lane import STATE_HOLD, STATE_LOST

FPS = 30.0


def strip(x: float, *, y0: float = 0.50, y1: float = 1.0, w: float = 0.008) -> np.ndarray:
    return np.asarray([[x - w, y0], [x - w, y1], [x + w, y1], [x + w, y0]], dtype=np.float32)


def lose_left(n_lost: int = 15, right_at=lambda k: 0.65):
    tr = EgoLaneTracker(FPS, hold_s=0.15, miss_s=0.30)
    for _ in range(12):
        tr.update([strip(0.35), strip(0.65)])
    return [tr.update([strip(right_at(k))]) for k in range(n_lost)]


def test_checked_line_stays_drawn_when_the_other_side_is_lost():
    out = lose_left()[-1]                             # 왼쪽 선이 0.5 초 없다
    assert out.state == STATE_HOLD
    assert out.display_line(out.ego_right) is not None and out.display_line(out.ego_left) is None


def test_only_for_the_allowed_time_since_the_last_checked_pair():
    assert ONE_SIDE_SEC == 1.2
    outs = lose_left(n_lost=60)                       # 한 줄로 2.0 초 > 1.2 초
    assert outs[25].state == STATE_HOLD and outs[-1].state == STATE_LOST


def test_a_line_seen_alone_from_the_start_is_not_drawn():
    tr = EgoLaneTracker(FPS, hold_s=0.15, miss_s=0.30)
    for _ in range(20):
        out = tr.update([strip(0.65)])                # 한 번도 짝으로 확인된 적 없는 선
    assert out.state == STATE_LOST


def test_a_line_that_moved_sideways_since_the_pair_is_not_drawn():
    # 차로폭 0.32 기준 0.35 x 0.32 = 0.11 까지 허용. 오른쪽 선이 바깥으로 0.15 움직인다
    outs = lose_left(n_lost=20, right_at=lambda k: 0.65 + 0.0075 * k)
    assert outs[11].state == STATE_HOLD                # 0.08 이동 — 아직 그린다
    assert outs[-1].state == STATE_LOST                # 0.14 이동 — 더 이상 짝 확인 선이 아니다


# ---- 역산 ----

VP = (0.5, 0.45)


def ray(x_bottom: float, *, y0: float = 0.56, bend: float = 0.0, w: float = 0.006) -> np.ndarray:
    """소실점에서 내려오는 차선 하나. bend = 아래로 갈수록 바깥으로 휘는 양 (화면 바닥 기준)."""
    ys = np.linspace(y0, 1.0, 12)
    t = (ys - VP[1]) / (1.0 - VP[1])
    xs = VP[0] + (x_bottom - VP[0]) * t + bend * ((ys - y0) / (1.0 - y0)) ** 2
    left = np.stack([xs - w, ys], 1)
    right = np.stack([xs + w, ys], 1)[::-1]
    return np.concatenate([left, right]).astype(np.float32)


def run_pair(n_pair: int = 40, n_alone: int = 15, bend_after: float = 0.0):
    tr = EgoLaneTracker(FPS, hold_s=0.15, miss_s=0.30)
    for _ in range(n_pair):
        tr.update([ray(0.15), ray(0.85)])
    outs = [tr.update([ray(0.85, bend=bend_after * min(1.0, (k + 1) / 5))]) for k in range(n_alone)]
    return tr, outs


def x_at(line: np.ndarray, y: float) -> float:
    line = line[np.argsort(line[:, 1])]
    return float(np.interp(y, line[:, 1], line[:, 0]))


def test_the_lost_side_is_derived_from_the_last_pair():
    _, outs = run_pair()
    out = outs[-1]                                    # 왼쪽 선이 0.5 초 없다
    assert out.state == STATE_HOLD and out.inferred == "left"
    line = out.display_line(out.ego_left)
    assert line is not None and out.ego_left.inferred and not out.ego_right.inferred
    true_x = VP[0] + (0.15 - VP[0]) * (0.9 - VP[1]) / (1.0 - VP[1])
    assert abs(x_at(line, 0.9) - true_x) < 0.01


def test_output_only_state_and_miss_timer_unchanged():
    # 역산 선은 출력에만 들어가고, 못 본 시간 · State 판정에는 쓰지 않는다
    tr, outs = run_pair()
    assert outs[-1].diag["miss_left"] > 0.3 and not outs[-1].fresh_left


def test_only_within_infer_sec_of_the_last_pair():
    _, outs = run_pair(n_alone=34)
    k_last = int(INFER_SEC * FPS) - 2
    assert outs[k_last].inferred == "left"
    assert outs[-1].inferred == "" and outs[-1].state == STATE_HOLD   # 1.13 초 — 한 줄만, 역산 없음


def test_a_line_that_bends_away_on_its_own_is_not_used():
    _, outs = run_pair(bend_after=0.06)                    # 오른쪽 선이 혼자 휘어 나간다 (우회전 · 출구)
    assert outs[-1].inferred == ""
    assert outs[-1].diag.get("infer_shape_resid", 0.0) > 0.005
