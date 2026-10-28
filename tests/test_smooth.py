"""선 다듬기 시험 — 마스크가 흔들려도 그리는 선은 흔들리지 않아야 한다.

점선은 프레임마다 1 ~ 3 개 조각으로 오고 먼 끝이 매번 달라진다. 그리는 것은 마스크 원본이 아니라
다듬은 중심선(`Boundary.line`)이다.
"""

from __future__ import annotations

import numpy as np
import pytest

from postprocess import EgoLaneTracker
from postprocess.config import LINE_MAX_EXT_STEP

FPS = 30.0


def strip(x: float, *, y0: float = 0.50, y1: float = 0.82, w: float = 0.008) -> np.ndarray:
    return np.asarray([[x - w, y0], [x - w, y1], [x + w, y1], [x + w, y0]], dtype=np.float32)


def test_line_follows_a_steady_boundary():
    tracker = EgoLaneTracker(FPS)
    out = None
    for _ in range(10):
        out = tracker.update([strip(0.35), strip(0.65)])
    left = out.ego_left
    assert left.line is not None and len(left.line) >= 2
    ys, xs = left.line[:, 1], left.line[:, 0]
    assert ys.min() == pytest.approx(0.50, abs=0.011) and ys.max() == pytest.approx(0.82, abs=0.011)
    assert np.allclose(xs, 0.35, atol=0.005)


def test_far_end_flicker_is_smoothed():
    """같은 선의 긴 마스크와 짧은 마스크가 번갈아 와도 그린 선의 끝이 튀지 않는다."""
    tracker = EgoLaneTracker(FPS)
    tops = []
    for i in range(40):
        left = strip(0.35, y0=0.50 if i % 2 == 0 else 0.70)   # 먼 끝이 0.50 과 0.70 을 오간다
        out = tracker.update([left, strip(0.65)])
        if out.ego_left.line is not None:
            tops.append(float(out.ego_left.line[:, 1].min()))
    steps = np.abs(np.diff(tops))
    assert steps.max() <= LINE_MAX_EXT_STEP + 1e-6           # 정해진 속도보다 빨리 움직이지 않는다
    assert np.mean(np.abs(np.diff([0.50, 0.70] * 20))) > 0.1  # 마스크의 먼 끝은 매 프레임 0.20 씩 튀었다


def test_fragments_of_one_line_are_joined():
    """한 선의 점선 조각 두 개는 가까운 조각만이 아니라 둘을 잇는 선 하나가 된다."""
    tracker = EgoLaneTracker(FPS)
    near = strip(0.35, y0=0.72, y1=0.82)
    far = strip(0.35, y0=0.50, y1=0.60)
    out = None
    for _ in range(5):
        out = tracker.update([near, far, strip(0.65)])
    line = out.ego_left.line
    assert line is not None
    assert line[:, 1].min() <= 0.51 and line[:, 1].max() >= 0.81


def test_new_identity_starts_a_new_line():
    tracker = EgoLaneTracker(FPS)
    for _ in range(10):
        out = tracker.update([strip(0.35), strip(0.65)])
    tid = out.ego_left.track_id
    for _ in range(20):                                   # 선이 사라지고 다른 선이 나타난다
        out = tracker.update([strip(0.65)])
    for _ in range(12):                                   # 0.10 떨어진 선 — 같은 선 범위 밖이라 새 선
        out = tracker.update([strip(0.25), strip(0.65)])
    assert out.ego_left.track_id != tid
    assert np.allclose(out.ego_left.line[:, 0], 0.25, atol=0.01)   # 예전 0.35 와 섞이지 않는다


def curved_strip(x0: float, *, bend: float = 0.15, y0: float = 0.50, y1: float = 0.82, w: float = 0.008) -> np.ndarray:
    """중심이 x0 + bend*(y1-y)^2 인 휘는 차선."""
    ys = np.linspace(y1, y0, 9)
    cx = x0 + bend * (y1 - ys) ** 2
    left = np.stack([cx - w, ys], 1)
    right = np.stack([cx + w, ys], 1)[::-1]
    return np.vstack([left, right]).astype(np.float32)


def test_line_follows_a_curve():
    """휘는 선은 휘게 그린다 — 모든 행에서 중심선이 도색 안에 있다."""
    tracker = EgoLaneTracker(FPS)
    out = None
    for _ in range(10):
        out = tracker.update([curved_strip(0.35), strip(0.65)])
    line = out.ego_left.line
    assert line is not None
    expected = 0.35 + 0.15 * (0.82 - line[:, 1]) ** 2
    assert np.abs(line[:, 0] - expected).max() < 0.004          # 모든 행에서 반폭 0.008 안
    assert expected.max() - expected.min() > 0.012              # 실제로 휜 선이다


def test_line_carries_the_paint_width():
    tracker = EgoLaneTracker(FPS)
    out = None
    for _ in range(5):
        out = tracker.update([strip(0.35, w=0.012), strip(0.65)])
    hw = out.ego_left.line_halfwidth
    assert hw is not None and np.allclose(hw, 0.012, atol=0.002)


def test_line_ends_do_not_hook():
    """비스듬한 곧은 선은 양 끝까지 곧게 그린다 (끝 행이 안쪽으로 휘지 않는다)."""
    tracker = EgoLaneTracker(FPS)
    ys = np.linspace(0.82, 0.50, 9)
    cx = 0.20 + 0.5 * (ys - 0.50)              # 아래 0.36 -> 위 0.20
    ribbon = np.vstack([np.stack([cx - 0.008, ys], 1), np.stack([cx + 0.008, ys], 1)[::-1]]).astype(np.float32)
    out = None
    for _ in range(10):
        out = tracker.update([ribbon, strip(0.65)])
    line = out.ego_left.line
    expected = 0.20 + 0.5 * (line[:, 1] - 0.50)
    assert np.abs(line[:, 0] - expected).max() < 0.003     # 첫 행과 마지막 행까지
