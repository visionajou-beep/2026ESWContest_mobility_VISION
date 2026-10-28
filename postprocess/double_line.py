"""이중선 처리.

이중선(안쪽 실선 + 바깥 선)은 모델이 프레임마다 한쪽 획만 번갈아 내서, 그대로 따라가면 선이 두 위치를 오간다.
짧은 시간 안에 같은 크기로 여러 번 오가면 이중선으로 보고, 바깥 획에서 읽은 값은 그 간격만큼 안쪽으로 옮긴다.
차에 가까운 안쪽 선을 자차 선으로 쓴다.
"""

from __future__ import annotations

import numpy as np

from .config import (
    STROKE_GAP,
    STROKE_INNER_SEC,
    STROKE_MEMORY_SEC,
    STROKE_MIN_SWITCHES,
    STROKE_SAME_ROWS,
    STROKE_WINDOW_SEC,
)
from .geometry import out_sign
from .lane import Track


def two_levels(values: list[tuple[float, float]]) -> float | None:
    """두 획을 번갈아 읽은 이중선이면 두 획 사이 간격, 아니면 None.

    `values` 는 시간 순 (x, 마스크 아래 끝 행). 연속한 두 값 사이에 STROKE_GAP 크기의 점프가
    STROKE_MIN_SWITCHES 번 이상, 양쪽 방향으로 고르게, 비슷한 크기로(1.5 배 안) 있고,
    두 마스크의 아래 끝 높이가 같을 때만 이중선으로 본다. 한 번 튄 것(빗물 줄기)은 이중선이 아니다.
    """
    if len(values) < 3:
        return None
    big = [b[0] - a[0] for a, b in zip(values, values[1:])
           if STROKE_GAP[0] <= abs(b[0] - a[0]) <= STROKE_GAP[1] and abs(b[1] - a[1]) <= STROKE_SAME_ROWS]
    if (len(big) < STROKE_MIN_SWITCHES or sum(d > 0 for d in big) < STROKE_MIN_SWITCHES // 2
            or sum(d < 0 for d in big) < STROKE_MIN_SWITCHES // 2):
        return None
    mags = [abs(d) for d in big]
    if max(mags) > 1.5 * min(mags):
        return None
    return float(np.median(mags))


class DoubleLineMixin:
    """EgoLaneTracker 의 이중선 처리 부분."""

    def _stroke_gap(self, tr: Track) -> float | None:
        """배운 이중선 간격 (기억 시간이 지나면 None)."""
        if tr.stroke is None or (self._u - tr.stroke[1]) * self.dt > STROKE_MEMORY_SEC:
            return None
        return tr.stroke[0]

    def _stroke_observe(self, tr: Track, prev: float, center_x: float) -> None:
        """읽은 위치를 기록하고, 이중선이면 간격을 배우고, 바깥 획에서 읽은 값은 안쪽으로 옮긴다."""
        raw = tr.x
        tr.hist.append((self._u, raw, float(tr.poly[:, 1].max())))
        tr.hist = [h for h in tr.hist if (self._u - h[0]) * self.dt <= STROKE_WINDOW_SEC]
        gap = two_levels([(h[1], h[2]) for h in tr.hist])
        if gap is not None:
            tr.stroke = (gap, self._u)
        gap = self._stroke_gap(tr)
        if gap is None or self.vp is None:
            return
        s = out_sign(prev, center_x)
        row = float(tr.poly[:, 1].max())
        if s * (raw - prev) < 0.5 * gap:      # 안쪽 획 (또는 안쪽으로 오는 중) — 읽은 그대로 쓴다
            tr.inner_u, tr.inner_row = self._u, row
            return
        if (self._u - tr.inner_u) * self.dt > STROKE_INNER_SEC:
            return                            # 한동안 바깥 획만 보였다면 그것이 선이다
        if abs(row - tr.inner_row) > STROKE_SAME_ROWS:
            return                            # 아래 끝 높이가 다른 마스크는 바깥 획이 아니다
        vy = float(self.vp[1])
        q = tr.poly.copy()
        q[:, 0] -= s * gap * np.clip(q[:, 1] - vy, 0.0, None) / max(self.ref_row - vy, 1e-3)
        tr.x, tr.poly, tr.from_outer = raw - s * gap, q, True
