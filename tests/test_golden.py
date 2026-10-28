"""실제 모델 출력으로 고정 정답 비교 — 추적기가 평가에 쓴 결과를 그대로 다시 내는지.

tests/data/golden.pkl.xz 는 평가 수치를 낸 추적기가 실제 모델 출력에 낸 프레임별 결과다.
이 코드가 같은 결과를 내면 평가 수치가 이 코드에도 그대로 해당한다.
다른 기기(젯슨 ARM)의 수치 계산 차이를 고려해 실수는 1e-6 까지 허용한다.

구간: 급커브 · 헤어핀 시뮬 녹화, 이중선 · 차로변경이 있는 긴 주행, 한국 야간 우천 도로 영상.
"""

from __future__ import annotations

import lzma
import pickle
from pathlib import Path

import numpy as np
import pytest

from postprocess import EgoLaneTracker

DATA = Path(__file__).resolve().parent / "data" / "golden.pkl.xz"
ATOL = 1e-6


def _cases():
    with lzma.open(DATA, "rb") as f:
        return pickle.load(f)["cases"]


CASES = _cases()


def _close(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return a.shape == b.shape and np.allclose(a, b, rtol=1e-6, atol=ATOL)


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_tracker_reproduces_the_evaluated_output(case):
    tr = EgoLaneTracker(case["fps"])
    if case["seed"] is not None:                 # 평가 때와 같은 출발 상태 (소실점 · 차로폭)
        tr.vp, tr._vp_n = case["seed"][0], 10_000
        tr._lane_w_est = case["seed"][1]
        tr._lane_w_hist.extend([case["seed"][1]] * 15)
    times = case["times"]
    for i, (polys, exp) in enumerate(zip(case["polys"], case["frames"])):
        out = tr.update(polys, None if times is None else times[i])
        where = f"{case['name']} 프레임 {i}"
        assert out.state == exp["state"], where
        assert (out.event, out.intent) == (exp["event"], exp["intent"]), where
        assert abs(out.center_x - exp["center_x"]) <= ATOL and abs(out.ref_row - exp["ref_row"]) <= ATOL, where
        for role in ("ego_left", "ego_right", "extra"):
            b, e = getattr(out, role), exp[role]
            assert (b is None) == (e is None), f"{where} {role}"
            if b is None:
                continue
            assert b.track_id == e["id"], f"{where} {role} 번호"
            assert abs(b.x - e["x"]) <= ATOL and abs(b.age_s - e["age_s"]) <= ATOL, f"{where} {role} 위치"
            assert _close(out.display_line(b), e["line"]), f"{where} {role} 그린 선"
