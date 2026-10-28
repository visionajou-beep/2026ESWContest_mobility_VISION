"""자차 차선 추적기의 기본 동작 시험 (GPU 없이).

정상 주행, 와이퍼에 가린 순간, 유지 시간보다 긴 점선 틈, 한 프레임 튀는 마스크, 차로변경 같은 주행 상황을
폴리곤으로 만들어 넣는다. 추적기는 정규화 폴리곤만 받으므로 모델이나 영상 없이 시험할 수 있다.
"""

from __future__ import annotations

import numpy as np
import pytest

from postprocess import EgoLaneTracker
from postprocess.config import HOLD_SEC, MATCH_GATE
from postprocess.geometry import is_longitudinal
from postprocess.lane import STATE_HOLD, STATE_LOST, STATE_OK

FPS = 30.0


def strip(x: float, *, y0: float = 0.50, y1: float = 0.82,
          w: float = 0.008) -> np.ndarray:
    """세로 차선 하나 — `x` 를 중심으로 한 얇은 띠."""
    return np.asarray([[x - w, y0], [x - w, y1], [x + w, y1], [x + w, y0]],
                      dtype=np.float32)


def run(tracker: EgoLaneTracker, frames: list[list[float]]):
    """프레임마다 차선 위치 목록을 넣고 결과를 모은다."""
    return [tracker.update([strip(x) for x in xs]) for xs in frames]


# --------------------------------------------------------------------------
# 선 번호 유지
# --------------------------------------------------------------------------

def test_steady_lane_keeps_one_identity():
    out = run(EgoLaneTracker(FPS), [[0.35, 0.65]] * 30)
    left_ids = {o.ego_left.track_id for o in out}
    right_ids = {o.ego_right.track_id for o in out}
    assert len(left_ids) == 1 and len(right_ids) == 1
    assert left_ids != right_ids
    assert all(o.state == STATE_OK for o in out)
    assert out[-1].ego_left.x == pytest.approx(0.35, abs=1e-3)
    assert out[-1].ego_right.x == pytest.approx(0.65, abs=1e-3)


def test_identity_survives_slow_lateral_drift():
    """프레임마다 조금씩 옆으로 움직이는 선은 같은 선으로 유지된다."""
    frames = [[0.35 + 0.004 * i, 0.65 + 0.004 * i] for i in range(20)]
    out = run(EgoLaneTracker(FPS), frames)
    assert len({o.ego_left.track_id for o in out}) == 1


# --------------------------------------------------------------------------
# 유지
# --------------------------------------------------------------------------

def test_wiper_length_dropout_is_held_and_reclaimed():
    """왼쪽 선이 세 프레임(0.1 초) 안 보여도 화면에서 사라지지 않는다."""
    tracker = EgoLaneTracker(FPS)
    before = run(tracker, [[0.35, 0.65]] * 10)
    during = run(tracker, [[0.65]] * 3)
    after = run(tracker, [[0.35, 0.65]] * 5)

    tid = before[-1].ego_left.track_id
    assert all(o.ego_left is not None and o.ego_left.held for o in during)
    assert all(o.ego_left.drawable(HOLD_SEC) for o in during)
    assert all(o.ego_left.track_id == tid for o in during)
    assert all(o.state == STATE_HOLD for o in during)
    # 다시 보이면 새 선이 아니라 같은 선이다
    assert after[0].ego_left.track_id == tid
    assert after[0].state == STATE_OK


def test_dropout_beyond_the_miss_window_reaches_state_lost():
    """두 선을 모두 잃으면 State 2 가 된다 (한쪽만 잃는 경우는 test_missing_lane.py)."""
    tracker = EgoLaneTracker(FPS)
    run(tracker, [[0.35, 0.65]] * 10)
    during = run(tracker, [[]] * 15)              # 0.5 초 — 유지 시간과 State 2 시간을 모두 넘는다

    assert during[0].state == STATE_HOLD
    assert during[-1].state == STATE_LOST
    # 유지 시간이 지난 선은 더 이상 그리지 않는다
    late = during[-1].ego_left
    assert late is None or not late.drawable(HOLD_SEC)


def test_hold_does_not_hand_the_role_to_the_next_line_over():
    """자차 선을 유지하는 동안 옆 차로 선이 자차 선이 되지 않는다."""
    tracker = EgoLaneTracker(FPS)
    before = run(tracker, [[0.10, 0.35, 0.65]] * 10)
    during = run(tracker, [[0.10, 0.65]] * 3)
    assert all(o.ego_left.track_id == before[-1].ego_left.track_id for o in during)
    assert all(o.ego_left.x == pytest.approx(0.35, abs=1e-3) for o in during)


# --------------------------------------------------------------------------
# 역할 교체 지연
# --------------------------------------------------------------------------

def test_merging_twins_keeps_the_longer_piece():
    """한 선이 긴 마스크와 짧은 조각으로 함께 나오면 긴 쪽을 남긴다."""
    tracker = EgoLaneTracker(FPS)
    long_piece = strip(0.56, y0=0.72, y1=0.87)
    stub = strip(0.52, y0=0.70, y1=0.74)
    for _ in range(4):
        out = tracker.update([strip(0.20), long_piece, stub])

    assert out.ego_right is not None
    span = out.ego_right.poly[:, 1].max() - out.ego_right.poly[:, 1].min()
    assert span > 0.10                       # 조각이 아니라 선


def test_single_frame_intruder_does_not_steal_the_role():
    tracker = EgoLaneTracker(FPS)
    before = run(tracker, [[0.35, 0.65]] * 10)
    spike = run(tracker, [[0.44, 0.35, 0.65]])     # 한 프레임, 경로에 더 가까운 마스크
    after = run(tracker, [[0.35, 0.65]] * 5)

    tid = before[-1].ego_left.track_id
    assert spike[0].ego_left.track_id == tid
    assert after[-1].ego_left.track_id == tid


def test_a_persistent_new_line_is_eventually_accepted():
    """역할 교체는 늦출 뿐 막지는 않는다."""
    tracker = EgoLaneTracker(FPS)
    before = run(tracker, [[0.35, 0.65]] * 10)
    after = run(tracker, [[0.44, 0.35, 0.65]] * 20)   # 0.67 초 — 0.25 초를 충분히 넘는다
    assert after[0].ego_left.track_id == before[-1].ego_left.track_id
    assert after[-1].ego_left.x == pytest.approx(0.44, abs=1e-3)


# --------------------------------------------------------------------------
# 실제 시간 — 처리 속도가 30 fps 보다 느려도 같은 초를 지킨다
# --------------------------------------------------------------------------

def run_timed(tracker: EgoLaneTracker, frames: list[list[float]], times: list[float]):
    return [tracker.update([strip(x) for x in xs], t=t) for xs, t in zip(frames, times)]


def _signature(o):
    def b(x):
        return None if x is None else (x.track_id, x.x, x.age_s, x.held,
                                       None if x.line is None else x.line.tobytes())
    return (o.frame, o.time_s, o.state, o.event, o.intent, o.offset, b(o.ego_left), b(o.ego_right), b(o.extra))


def test_timestamps_at_the_nominal_rate_change_nothing():
    """정확히 30 fps 시각을 넣은 결과는 시각 없이 돌린 결과와 같다 (시각의 기준점과 상관없이)."""
    frames = ([[0.02, 0.22, 0.42, 0.62]] * 10 + [[0.22, 0.42, 0.62]] * 4
              + [[x + 0.004 * i for x in (0.02, 0.22, 0.42, 0.62)] for i in range(1, 46)])
    plain = run(EgoLaneTracker(FPS), frames)
    for origin in (0.0, 1234.5):
        timed = run_timed(EgoLaneTracker(FPS), frames, [origin + k / FPS for k in range(len(frames))])
        assert [_signature(o) for o in timed] == [_signature(o) for o in plain]


def _seconds_to_state_lost(rate: float, timed: bool) -> float:
    tracker = EgoLaneTracker(FPS)
    times = [k / rate for k in range(40)]
    frames = [[0.35, 0.65]] * 10 + [[]] * 30    # 두 선을 모두 잃는다
    out = run_timed(tracker, frames, times) if timed else run(tracker, frames)
    k = next(i for i, o in enumerate(out) if o.state == STATE_LOST)
    return times[k] - times[9]                  # 선을 마지막으로 본 뒤


def test_state_lost_comes_after_the_miss_window_in_seconds_at_any_rate():
    """State 2 는 처리 속도와 상관없이 0.3 초 뒤에 온다 (프레임 수로 세면 15 fps 에서 0.6 초가 된다)."""
    ref = _seconds_to_state_lost(30.0, timed=True)
    assert ref == pytest.approx(0.3, abs=1 / 30 + 1e-9)
    assert abs(_seconds_to_state_lost(15.0, timed=True) - ref) <= 1 / 15 + 1e-9
    assert _seconds_to_state_lost(15.0, timed=False) > 1.8 * ref     # 프레임 수로 센 경우


def _seconds_to_accept(rate: float, timed: bool) -> float:
    tracker = EgoLaneTracker(FPS)
    times = [k / rate for k in range(30)]
    frames = [[0.35, 0.65]] * 10 + [[0.44, 0.35, 0.65]] * 20
    out = run_timed(tracker, frames, times) if timed else run(tracker, frames)
    k = next(i for i, o in enumerate(out) if o.ego_left.x == pytest.approx(0.44, abs=1e-3))
    return times[k] - times[10]                 # 새 선이 처음 나타난 뒤


def test_role_hysteresis_is_in_seconds_at_any_rate():
    ref = _seconds_to_accept(30.0, timed=True)
    assert abs(_seconds_to_accept(15.0, timed=True) - ref) <= 1 / 15 + 1e-9
    assert _seconds_to_accept(15.0, timed=False) > 1.8 * ref


def _drawn_fraction_after(rate: float, timed: bool, seconds: float = 4 / 30) -> float:
    """왼쪽 선이 옆으로 0.04 옮겨졌을 때(같은 선), `seconds` 뒤 그린 선이 얼마나 따라왔는지."""
    tracker = EgoLaneTracker(FPS)
    n0 = 10
    n = n0 + int(round(seconds * rate)) + 1
    times = [k / rate for k in range(n)]
    frames = [[0.35, 0.65]] * n0 + [[0.39, 0.65]] * (n - n0)
    out = run_timed(tracker, frames, times) if timed else run(tracker, frames)
    return (float(out[-1].ego_left.line[:, 0].mean()) - 0.35) / 0.04


def test_line_smoothing_keeps_its_lag_in_seconds_at_a_lower_rate():
    """평활 가중치 0.7: 1/30 초 뒤 1 - 0.3^2 = 0.91 만큼 따라온다 — 15 fps 에서도 같은 시간 기준."""
    ref = _drawn_fraction_after(30.0, timed=True, seconds=1 / 30)
    assert ref == pytest.approx(0.91, abs=0.01)
    assert abs(_drawn_fraction_after(15.0, timed=True, seconds=1 / 30) - ref) < 0.02
    assert abs(_drawn_fraction_after(15.0, timed=False, seconds=1 / 30) - ref) > 0.12   # 프레임 수로 센 경우


def test_a_stall_expires_old_lines_at_once():
    """0.5 초 동안 프레임이 없으면 그 전에 본 선은 바로 안 그린다."""
    tracker = EgoLaneTracker(FPS)
    run_timed(tracker, [[0.35, 0.65]] * 10, [k / FPS for k in range(10)])
    out = run_timed(tracker, [[0.65]], [9 / FPS + 0.5])[0]
    # 왼쪽은 0.5 초 전 선이라 안 그리고, 오른쪽은 이번 프레임에 본 짝 확인 선이라 State 1 로 남는다
    assert out.state == STATE_HOLD
    assert out.ego_left is None or not out.ego_left.drawable(HOLD_SEC)
    assert out.display_line(out.ego_left) is None and out.display_line(out.ego_right) is not None


@pytest.mark.parametrize("rate", [15.0, 20.0, 25.0, 30.0])
def test_a_line_seen_once_does_not_take_the_role_at_any_rate(rate):
    """한 프레임만 나온 마스크는 어떤 처리 속도에서도 자차 역할을 가져가지 못한다."""
    tracker = EgoLaneTracker(FPS)
    times = [k / rate for k in range(30)]
    out = run_timed(tracker, [[0.35, 0.65]] * 10 + [[0.44, 0.35, 0.65]] + [[0.35, 0.65]] * 19, times)
    tid = out[9].ego_left.track_id
    assert all(o.ego_left.track_id == tid for o in out[10:])


def test_one_frame_after_a_stall_is_not_a_quarter_second_of_agreement():
    """0.5 초 멈춘 뒤의 한 프레임은 시간이 길어도 관측 한 번으로 센다 (0.25 초 지연을 혼자 넘지 않는다)."""
    tracker = EgoLaneTracker(FPS)
    before = run_timed(tracker, [[0.35, 0.65]] * 10, [k / FPS for k in range(10)])
    t = 9 / FPS + 0.5
    spike = run_timed(tracker, [[0.44, 0.35, 0.65]], [t])
    after = run_timed(tracker, [[0.35, 0.65]] * 5, [t + (k + 1) / FPS for k in range(5)])
    tid = before[-1].ego_left.track_id
    assert spike[0].ego_left.track_id == tid
    assert after[-1].ego_left.track_id == tid


def test_time_never_runs_backwards():
    """같거나 이전 시각이 들어오면 길이 0 인 프레임으로 본다."""
    tracker = EgoLaneTracker(FPS)
    out = run_timed(tracker, [[0.35, 0.65]] * 3, [1.0, 1.0, 0.9])
    assert [o.time_s for o in out] == [0.0, 0.0, 0.0]
    assert all(o.state == STATE_OK for o in out)


# --------------------------------------------------------------------------
# 차로변경
# --------------------------------------------------------------------------

def test_lane_change_hands_the_crossed_line_to_the_other_role():
    """왼쪽으로 차로를 바꾸면 선들이 오른쪽으로 밀리고, 넘은 선은 번호를 지킨 채 오른쪽 선이 된다.

    선이 차 밑을 지나는 순간에도 같은 선을 새 차로의 오른쪽 선으로 계속 그려야 한다.
    """
    tracker = EgoLaneTracker(FPS)
    start = run(tracker, [[0.05, 0.35, 0.65]] * 10)
    crossing_id = start[-1].ego_left.track_id

    frames = [[0.05 + 0.006 * i, 0.35 + 0.006 * i, 0.65 + 0.006 * i]
              for i in range(1, 40)]
    out = run(tracker, frames)

    events = [o for o in out if o.event]
    assert events and events[0].event == "change_left"
    # 셋째 선은 넘기 전에 나온다
    assert any(o.intent == "left" and o.extra is not None
               for o in out[: out.index(events[0])])
    after = out[-1]
    assert after.ego_right.track_id == crossing_id     # 같은 번호로 넘겨받음
    assert after.ego_left.track_id != crossing_id
    assert after.ego_left.x < after.ego_right.x


def test_intent_clears_once_the_lane_change_is_over():
    """방금 넘은 선 가까이 있는 것을 되돌아가려는 의도로 읽지 않는다."""
    tracker = EgoLaneTracker(FPS)
    run(tracker, [[0.05, 0.35, 0.65]] * 10)
    out = run(tracker, [[0.05 + 0.006 * i, 0.35 + 0.006 * i, 0.65 + 0.006 * i]
                        for i in range(1, 40)])
    assert out[-1].intent != "right"


def test_drifting_toward_a_boundary_raises_the_intent_flag():
    """왼쪽 선 가까이 붙어 가면 그쪽 셋째 선이 나온다."""
    tracker = EgoLaneTracker(FPS)
    out = run(tracker, [[0.15, 0.47, 0.79]] * 20)
    assert out[0].intent == ""                         # 첫 프레임에는 아니다
    assert out[-1].intent == "left"
    assert out[-1].extra is not None
    assert out[-1].extra.x == pytest.approx(0.15, abs=1e-3)


def _left_change_with_a_far_line():
    """차선 네 개, 왼쪽으로 차로를 바꾼다. 넘기 전 셋째 선은 0.22 선이고, 넘는 순간 역할이 한 차로씩 밀리면
    '왼쪽 옆 선'은 한 차로 더 바깥의 0.02 선이 된다."""
    tracker = EgoLaneTracker(FPS)
    run(tracker, [[0.02, 0.22, 0.42, 0.62]] * 10)
    out = run(tracker, [[x + 0.004 * i for x in (0.02, 0.22, 0.42, 0.62)] for i in range(1, 46)])
    return tracker, out


def test_the_third_line_does_not_jump_one_lane_out_after_the_crossing():
    """선을 넘은 뒤에는 바로 두 선으로 돌아간다 (셋째 선이 한 차로 바깥으로 옮겨 가지 않는다)."""
    _, out = _left_change_with_a_far_line()
    k = next(i for i, o in enumerate(out) if o.event)
    before = [o for o in out[:k] if o.intent == "left" and o.extra is not None]
    assert before, "넘기 전에 셋째 선이 있어야 한다"
    pinned = before[-1].extra.track_id
    assert out[k].event == "change_left"
    assert out[-1].ego_left.track_id == pinned          # 셋째 선이던 선이 새 왼쪽 선이 된다
    for o in out[k:]:
        assert o.extra is None, f"frame {o.frame}: third line {o.extra.track_id} at x={o.extra.x:.3f}"


def test_sitting_against_the_crossed_line_raises_no_new_third_line():
    """넘은 직후 그 선에 붙어 멈춰 있어도 되돌아가려는 의도로 보지 않는다."""
    tracker = EgoLaneTracker(FPS)
    run(tracker, [[0.02, 0.22, 0.42, 0.62]] * 10)
    moving = run(tracker, [[x + 0.004 * i for x in (0.02, 0.22, 0.42, 0.62)] for i in range(1, 32)])
    assert any(o.event for o in moving)
    last = [x + 0.004 * 31 for x in (0.02, 0.22, 0.42, 0.62)]
    stopped = run(tracker, [last] * 40)
    k = next(i for i, o in enumerate(moving) if o.event)
    for o in moving[k:] + stopped:
        assert o.intent == "" and o.extra is None


def _perspective_ribbon(k: float, *, vp=(0.5, 0.45), w: float = 0.01, sliver: float = 0.004,
                        y0: float = 0.55, n: int = 46) -> np.ndarray:
    """원근이 있는 차선 x = vp_x + k * (y - vp_y). 화면 밖으로 나가는 곳에서는 모델 마스크처럼
    화면 가장자리를 따라 `sliver` 폭의 띠로 바닥까지 내려간다."""
    ys = np.linspace(y0, 1.0, n)
    xc = vp[0] + k * (ys - vp[1])
    lo = np.clip(xc - w, 0.0, 1.0)
    hi = np.clip(xc + w, 0.0, 1.0)
    lo = np.minimum(lo, 1.0 - sliver)
    hi = np.maximum(hi, sliver)
    return np.asarray(list(zip(lo, ys)) + list(zip(hi[::-1], ys[::-1])), dtype=np.float32)


def test_diag_says_why_masks_did_not_count():
    """진단 값(diag)은 필터별로 버린 마스크 수와 각 쪽을 못 본 시간을 알려 준다 (판정에는 쓰지 않는다)."""
    tracker = EgoLaneTracker(FPS)
    flat = np.asarray([[0.2, 0.70], [0.8, 0.70], [0.8, 0.72], [0.2, 0.72]], np.float32)   # 정지선
    low = strip(0.40, y0=0.75, y1=0.80)     # 기준행(0.65)보다 아래에만 있고 너무 짧아 늘여 읽지 못한다
    short = strip(0.55, y0=0.62, y1=0.70)   # 기준행을 지나지만 길이 0.08 (< 0.10)
    out = None
    for _ in range(3):
        out = tracker.update([strip(0.30), strip(0.70), flat, low, short])
    d = out.diag
    assert d["masks"] == 5
    assert (d["slope"], d["reach"], d["short"], d["vp"]) == (1, 1, 1, 0)
    assert out.observed == 2
    assert d["miss_left"] == 0.0 and d["miss_right"] == 0.0
    for _ in range(12):                                   # 오른쪽 선이 12 프레임(0.4 초) 없다
        out = tracker.update([strip(0.30)])
    assert out.state == STATE_HOLD                        # 왼쪽은 짝 확인 선이라 State 1
    assert out.diag["miss_right"] > out.diag["miss_limit"] >= out.diag["miss_left"]


def test_a_line_whose_near_end_retreats_while_moving_draws_no_hook():
    """옆으로 움직이면서 아래 끝이 위로 올라가는 선을 그릴 때, 가까운 쪽 끝에 갈고리 모양이 생기지 않는다.

    왼쪽 선이 프레임마다 0.02 씩 왼쪽으로 가며 아래 끝이 0.95 에서 0.80 으로 올라간다.
    그린 선의 이웃한 행 사이에 계단이 없어야 한다.
    """
    tracker = EgoLaneTracker(FPS)
    for _ in range(12):
        tracker.update([strip(0.32, y1=0.95), strip(0.70, y1=0.95)])
    worst = 0.0
    for k in range(1, 6):
        out = tracker.update([strip(0.32 - 0.02 * k, y1=0.95 - 0.03 * k), strip(0.70, y1=0.95)])
        x = out.ego_left.line[:, 0]
        worst = max(worst, float(np.max(np.abs(np.diff(x)))))
    assert worst < 0.003, f"a step of {worst:.4f} between neighbouring rows"


def test_a_line_leaving_the_frame_side_does_not_run_down_the_border():
    """화면 옆으로 나가는 선은 화면 가장자리를 따라 내려가게 그리지 않고, 도색이 화면을 벗어나는 곳에서 끝낸다."""
    tracker = EgoLaneTracker(FPS)
    left, right = _perspective_ribbon(-1.18), _perspective_ribbon(0.818)
    out = [tracker.update([left, right]) for _ in range(30)][-1]
    line = out.ego_left.line
    assert line is not None
    assert line[:, 0].min() > 0.01, "the centreline slid onto the left border"
    exit_row = 0.45 + (0.5 - 0.01) / 1.18            # 띠의 바깥 변이 x = 0 에 닿는 행
    assert line[:, 1].max() <= exit_row + 0.02
    assert out.ego_right.line[:, 1].max() > 0.95        # 화면 안의 선은 바닥까지 간다


# --------------------------------------------------------------------------
# 모양 필터
# --------------------------------------------------------------------------

def test_stop_line_is_ignored():
    """도로를 가로지르는 선은 차선 경계가 아니다."""
    tracker = EgoLaneTracker(FPS)
    stop_line = np.asarray([[0.20, 0.64], [0.80, 0.64], [0.80, 0.66], [0.20, 0.66]],
                           dtype=np.float32)
    out = tracker.update([strip(0.35), strip(0.65), stop_line])
    assert out.observed == 2


# --------------------------------------------------------------------------
# 기준행은 화면이 아니라 소실점을 따른다
# --------------------------------------------------------------------------

def taper(x_bottom: float, *, vp=(0.5, 0.63), y_bottom: float = 0.98,
          y_top: float = 0.75, w: float = 0.006) -> np.ndarray:
    """차에서 `vp` 쪽으로 원근 있게 그린 차선."""
    vx, vy = vp
    t = (y_top - y_bottom) / (vy - y_bottom)
    x_top = x_bottom + t * (vx - x_bottom)
    return np.asarray([[x_bottom - w, y_bottom], [x_top - w, y_top],
                       [x_top + w, y_top], [x_bottom + w, y_bottom]],
                      dtype=np.float32)


def test_reference_row_drops_below_a_low_horizon():
    """소실점이 0.63 에 있는 카메라는 0.65 행에서 읽으면 안 된다 (소실점 바로 아래라 모든 선이 한 점에 모인다)."""
    tracker = EgoLaneTracker(FPS)
    for _ in range(12):   # 소실점을 배우는 시간 이후
        out = tracker.update([taper(0.20), taper(0.80)])

    assert tracker.vp is not None
    assert tracker.vp[1] == pytest.approx(0.63, abs=0.02)
    assert 0.74 < out.ref_row < 0.79
    # 자차 좌 · 우가 한 차로폭만큼 떨어져 읽힌다
    assert out.ego_right.x - out.ego_left.x > 0.15


def test_bdd100k_geometry_keeps_the_calibrated_row():
    """BDD100K 의 소실점 높이에서는 기준행이 0.65 가 된다."""
    tracker = EgoLaneTracker(FPS)
    for _ in range(5):
        out = tracker.update([taper(0.15, vp=(0.5, 0.4586), y_top=0.60),
                              taper(0.85, vp=(0.5, 0.4586), y_top=0.60)])
    assert out.ref_row == pytest.approx(0.65, abs=0.01)


def test_mask_reaching_the_horizon_is_still_a_side_boundary():
    """소실점까지 이어진 마스크는 거기서 경로에 걸치지만, 차 가까이에서 보면 분명히 한쪽 경계다."""
    left = taper(0.20, y_top=0.60)          # 소실점(0.63) 위까지 올라간다
    assert not is_longitudinal(left, 0.5, row=0.0)      # 전체 길이로 보면 걸친다
    assert is_longitudinal(left, 0.5, row=0.77)         # 차 가까이에서 보면 왼쪽

    tracker = EgoLaneTracker(FPS)
    for _ in range(12):   # 소실점을 배우는 시간 이후
        out = tracker.update([left, taper(0.80, y_top=0.60)])
    assert out.ego_left is not None and out.ego_right is not None
    assert out.ego_left.x < out.center_x < out.ego_right.x
    assert out.state == STATE_OK


# --------------------------------------------------------------------------
# 그 밖의 선택 규칙
# --------------------------------------------------------------------------

def test_match_gate_is_smaller_than_a_lane():
    """같은 선으로 잇는 범위는 옆 차로 선까지 닿지 않는다."""
    assert MATCH_GATE < 0.32 / 2


def test_straddled_line_keeps_its_side_and_hands_over():
    """오른쪽 선이 차 밑으로 지나가는 동안(오른쪽 차로변경) 계속 오른쪽 선으로 그리고,
    분명히 왼쪽으로 넘어가면 차로변경을 알리고 새 왼쪽 선으로 넘긴다."""
    from postprocess import EgoLaneTracker
    fps = 30.0
    tr = EgoLaneTracker(fps)
    def strip(x):
        return np.asarray([[x - 0.008, 0.50], [x - 0.008, 1.0], [x + 0.008, 1.0], [x + 0.008, 0.50]], dtype=np.float32)
    for _ in range(15):
        out = tr.update([strip(0.35), strip(0.65)])
    rid = out.ego_right.track_id
    xs = np.linspace(0.65, 0.40, 60)            # 오른쪽 선이 가운데(0.5)를 지나간다
    seen_astride = False
    events = []
    for x in xs:
        out = tr.update([strip(x - 0.30), strip(x), strip(x + 0.30)])
        if abs(x - 0.5) < 0.02:
            seen_astride = True
            assert out.ego_right is not None and out.ego_right.track_id == rid, "line dropped while straddled"
        if out.event:
            events.append(out.event)
    assert seen_astride
    assert "change_right" in events
    assert out.ego_left is not None and out.ego_left.track_id == rid


def test_min_span_ignores_short_stubs():
    from postprocess import EgoLaneTracker
    stub = np.asarray([[0.44, 0.76], [0.44, 0.82], [0.46, 0.82], [0.46, 0.76]], dtype=np.float32)
    tr = EgoLaneTracker(30.0)
    out = None
    for _ in range(5):
        out = tr.update([stub])
    assert out.ego_left is None and out.ego_right is None


def test_min_span_keeps_a_dashed_line_made_of_short_dashes():
    from postprocess import EgoLaneTracker
    def dash(x, y0, y1):
        return np.asarray([[x - 0.008, y0], [x - 0.008, y1], [x + 0.008, y1], [x + 0.008, y0]], dtype=np.float32)
    tr = EgoLaneTracker(30.0)
    out = None
    for _ in range(5):
        out = tr.update([dash(0.35, 0.60, 0.67), dash(0.35, 0.68, 0.76), dash(0.65, 0.50, 1.0)])
    assert out.ego_left is not None                      # 한 줄의 0.07 조각 두 개 = 0.16 짜리 선 하나


def test_line_first_seen_astride_fills_the_empty_side():
    from postprocess import EgoLaneTracker
    def strip(x):
        return np.asarray([[x - 0.008, 0.50], [x - 0.008, 1.0], [x + 0.008, 1.0], [x + 0.008, 0.50]], dtype=np.float32)
    tr = EgoLaneTracker(30.0)
    out = None
    for _ in range(15):
        out = tr.update([strip(0.38)])                    # 한동안 왼쪽 선만 있다
    for _ in range(10):
        out = tr.update([strip(0.38), strip(0.49)])       # 차 바로 밑(경로 0.5)에 선이 나타난다
    assert out.ego_left is not None and abs(out.ego_left.x - 0.38) < 0.01
    assert out.ego_right is not None and abs(out.ego_right.x - 0.49) < 0.01


def test_mask_not_pointing_at_the_vanishing_point_is_ignored():
    """두 선으로 소실점이 (0.5, 0.4) 로 정해진 뒤, 소실점에서 0.2 비켜 가는 사선 조각은 자차 선이 되지 않는다."""
    from postprocess import EgoLaneTracker
    def taper(x_bottom, x_top, y_top=0.45):
        return np.asarray([[x_bottom - 0.01, 1.0], [x_top - 0.003, y_top], [x_top + 0.003, y_top], [x_bottom + 0.01, 1.0]], dtype=np.float32)
    left, right = taper(0.20, 0.49), taper(0.80, 0.51)
    stray = np.asarray([[0.30, 0.99], [0.42, 0.80], [0.44, 0.80], [0.32, 0.99]], dtype=np.float32)  # 소실점 높이에서 x 약 0.9 를 가리킨다
    tr = EgoLaneTracker(30.0)
    out = None
    for _ in range(15):
        out = tr.update([left, right, stray])
    # 위치(x)는 소실점 추정에 섞인 조각의 영향을 받으므로, 실제 왼쪽 선을 골랐는지는 폴리곤으로 확인한다
    assert out.ego_left is not None and np.array_equal(out.ego_left.poly, left)
    assert out.ego_right is not None and np.array_equal(out.ego_right.poly, right)


# --------------------------------------------------------------------------
# 급커브: 소실점 검사는 새 후보만 거르고, 따라가던 자차 선은 버리지 않는다
# --------------------------------------------------------------------------

VP_T = (0.5, 0.45)
REF_T = VP_T[1] + 0.354 * (1.0 - VP_T[1])        # 이 소실점에서 추적기의 기준행


def line_ribbon(x_ref: float, k: float, *, y0: float = 0.50, y1: float = 0.95, w: float = 0.006) -> np.ndarray:
    """곧은 차선 x = x_ref + k * (y - REF_T)."""
    ys = np.linspace(y0, y1, 24)
    xs = x_ref + k * (ys - REF_T)
    return np.asarray(list(zip(xs - w, ys)) + list(zip(xs[::-1] + w, ys[::-1])), dtype=np.float32)


def _sharp_curve(keep_tracked: bool, intruder: bool = False):
    """왼쪽 선이 기준행을 축으로 돌아 양 끝이 소실점을 0.0 -> 0.16 만큼 비켜 간다 (급커브)."""
    tracker = EgoLaneTracker(FPS)
    xl, xr = VP_T[0] - 1.0 * (REF_T - VP_T[1]), VP_T[0] + 1.0 * (REF_T - VP_T[1])
    for _ in range(15):
        before = tracker.update([line_ribbon(xl, -1.0), line_ribbon(xr, 1.0)])
    outs = []
    for i in range(1, 21):            # i = 10 부터 검사 기준 0.10 을 넘는다 (11 프레임, 0.37 초)
        polys = [line_ribbon(xl, -1.0 - 0.055 * i), line_ribbon(xr, 1.0)]
        if intruder:
            polys.append(line_ribbon(0.47, -2.2, y0=0.70))      # 경로 가까이에 처음 나타난 사선 표시
        outs.append(tracker.update(polys))
    return before, outs


def test_keep_tracked_follows_the_ego_line_through_the_bend():
    before, outs = _sharp_curve(keep_tracked=True)
    assert all(o.fresh_left and o.ego_left.track_id == before.ego_left.track_id for o in outs)
    assert outs[-1].state == STATE_OK
    assert outs[-1].diag["vp_kept"] >= 1


def test_keep_tracked_still_drops_a_new_stray_mark():
    before, outs = _sharp_curve(keep_tracked=True, intruder=True)
    assert outs[-1].diag["vp"] >= 1                              # 사선 표시는 버린다
    ids = {before.ego_left.track_id, before.ego_right.track_id}
    assert all(o.ego_left.track_id in ids and o.ego_right.track_id in ids for o in outs)


# --------------------------------------------------------------------------
# 넓은 마스크 · 먼 끝 꺾임
# --------------------------------------------------------------------------

XL_T = VP_T[0] - 1.0 * (REF_T - VP_T[1])      # 소실점을 지나는 자차 좌 · 우 선
XR_T = VP_T[0] + 1.0 * (REF_T - VP_T[1])


def _persp(x_ref, k, extra=lambda ys: 0.0, right_extra=lambda ys: 0.0, w=0.005):
    ys = np.linspace(0.50, 0.95, 46)
    xc = x_ref + k * (ys - REF_T) + extra(ys)
    return np.asarray(list(zip(xc - w, ys)) + list(zip((xc + w + right_extra(ys))[::-1], ys[::-1])), dtype=np.float32)


def _warm():
    tracker = EgoLaneTracker(FPS)
    for _ in range(15):
        tracker.update([_persp(XL_T, -1.0), _persp(XR_T, 1.0)])          # 소실점과 도색 폭을 배운다
    return tracker


def test_shape_guard_does_not_draw_through_a_merged_wide_mask():
    """y 0.80 아래에서 왼쪽 마스크가 옆 선까지 삼켜 오른쪽 변이 0.25 넓어진다. 넓은 행은 쓰지 않으므로 선이 끌려가지 않는다."""
    merged = _persp(XL_T, -1.0, right_extra=lambda ys: np.where(ys > 0.80, (ys - 0.80) / 0.15 * 0.25, 0.0))
    out = _warm().update([merged, _persp(XR_T, 1.0)])
    ln = out.ego_left.line
    true_x = XL_T - 1.0 * (0.90 - REF_T)
    assert float(np.interp(0.90, ln[:, 1], ln[:, 0])) == pytest.approx(true_x, abs=0.02)


def test_far_end_kink_is_cut_and_a_near_one_is_not():
    """먼 쪽 절반의 꺾임에서는 선을 자르고(선 끝도 옮긴다), 가까운 쪽 절반은 자르지 않는다."""
    from postprocess.config import LINE_ROWS
    from postprocess.smooth import ShapeStabilizer
    sh = ShapeStabilizer()
    sh.vp_y, sh.ref_row, sh.lane_w = VP_T[1], REF_T, 0.32
    sel = (LINE_ROWS >= 0.50) & (LINE_ROWS <= 0.95)
    base = XL_T - 1.0 * (LINE_ROWS - REF_T)
    far = base + np.where(LINE_ROWS < 0.56, 0.02 * (0.56 - LINE_ROWS) / 0.01, 0.0)     # y 0.56 위의 꺾임
    st = {"x": far.copy(), "y_top": 0.50, "y_bot": 0.95}
    out = sh._trim_far_kink(st, sel.copy())
    assert LINE_ROWS[out].min() >= 0.55 and st["y_top"] >= 0.55
    near = base + np.where(LINE_ROWS > 0.85, 0.02 * (LINE_ROWS - 0.85) / 0.01, 0.0)    # 가까운 쪽 절반의 휨
    st = {"x": near.copy(), "y_top": 0.50, "y_bot": 0.95}
    assert np.array_equal(sh._trim_far_kink(st, sel.copy()), sel)


# --------------------------------------------------------------------------
# 봤지만 다듬은 선을 못 만든 쪽은 못 본 것으로 센다
# --------------------------------------------------------------------------

def _no_left_line(tr: EgoLaneTracker) -> None:
    """왼쪽 선의 다듬기가 실패하게 만든다 (이중선이 합쳐진 마스크처럼)."""
    orig = tr._shape.update

    def upd(role, b, *a, **k):
        if role == "ego_left":
            return None                                    # b.line 이 None 으로 남는다
        return orig(role, b, *a, **k)
    tr._shape.update = upd


def test_a_side_seen_but_not_drawn_counts_as_missed():
    tr = EgoLaneTracker(FPS, hold_s=0.15, miss_s=0.30)
    run(tr, [[0.35, 0.65]] * 12)
    _no_left_line(tr)
    outs = run(tr, [[0.35, 0.65]] * 10)                    # 왼쪽은 매 프레임 보이지만 그리지 못한다
    first, last = outs[0], outs[-1]
    assert first.fresh_left and first.display_line(first.ego_left) is None
    assert first.state == STATE_HOLD and first.diag["undrawn_left"]
    assert last.diag["miss_left"] == pytest.approx(10 / FPS)
    assert last.diag["miss_left"] > last.diag["miss_limit"]
    # 왼쪽은 못 본 것으로 센다. 오른쪽은 짝 확인 선이라 State 1 이고, 왼쪽은 그리지 않는다
    assert last.state == STATE_HOLD and last.display_line(last.ego_left) is None


def test_drawn_again_resets_the_timer():
    tr = EgoLaneTracker(FPS, hold_s=0.15, miss_s=0.30)
    run(tr, [[0.35, 0.65]] * 12)
    orig = tr._shape.update
    _no_left_line(tr)
    run(tr, [[0.35, 0.65]] * 5)
    tr._shape.update = orig
    out = tr.update([strip(0.35), strip(0.65)])
    assert out.display_line(out.ego_left) is not None
    assert out.state == STATE_OK and out.diag["miss_left"] == 0.0 and not out.diag["undrawn_left"]


def test_undrawn_time_carries_into_an_unseen_gap():
    """0.2 초 못 그리고 이어서 0.13 초 안 보이면 왼쪽은 0.33 초 동안 없던 것 — 왼쪽을 잃는다
    (오른쪽은 짝 확인 선으로 남아 State 1)."""
    tr = EgoLaneTracker(FPS, hold_s=0.15, miss_s=0.30)
    run(tr, [[0.35, 0.65]] * 12)
    _no_left_line(tr)
    run(tr, [[0.35, 0.65]] * 6)
    outs = run(tr, [[0.65]] * 4)
    assert outs[-1].diag["miss_left"] == pytest.approx(10 / FPS)
    assert outs[-1].diag["miss_left"] > outs[-1].diag["miss_limit"]
    assert outs[-1].state == STATE_HOLD                    # 오른쪽은 짝 확인 선
    assert outs[-1].display_line(outs[-1].ego_left) is None
