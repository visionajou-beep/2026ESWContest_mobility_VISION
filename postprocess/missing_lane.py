"""한쪽 선이 안 보일 때의 처리 — 한 줄 유지와 반대쪽 역산.

한 줄 유지: 한쪽을 잃어도, 남은 선이 방금 전까지 한 차로폭 짝의 한쪽이었고 그 뒤 옆으로 크게 움직이지 않았으면
그 선은 계속 그린다 (State 1).

역산: 한 줄만 그릴 때, 마지막으로 두 선이 함께 보인 쌍의 반대쪽 선을 지금 보이는 선이 그 뒤 움직인 만큼 옮겨 그린다.
차가 옆으로 움직이거나 방향을 틀 때 화면의 모든 선은 행(y)에 대한 직선(c0 + s·y)만큼 함께 움직이므로,
반대쪽 선은 제 모양(곧은 길은 곧게, 커브는 커브로)을 유지한다. 보이는 선이 그동안 혼자 모양을 바꿨다면
(우회전 · 출구처럼 차로를 떠나는 선) 역산하지 않는다. 역산한 선은 표시(inferred)를 달아 내보낸다.
"""

from __future__ import annotations

import numpy as np

from .config import (
    INFER_MIN_BELOW_VP,
    INFER_MIN_POINTS,
    INFER_SEC,
    INFER_SHAPE_TOL,
    LANE_W_PRIOR,
    ONE_SIDE_DX,
    ONE_SIDE_PAIR,
    ONE_SIDE_SEC,
)
from .lane import EGO_ROLES, Boundary


class MissingLaneMixin:
    """EgoLaneTracker 의 '한쪽 선이 안 보일 때' 부분."""

    def _note_pair(self, left: Boundary | None, right: Boundary | None) -> None:
        """두 선을 이번 프레임에 모두 보고 그렸고 간격이 한 차로폭이면, 그 쌍을 기억한다."""
        if left is None or right is None or left.age_s != 0.0 or right.age_s != 0.0:
            return
        if left.line is None or right.line is None:
            return
        lw = self._lane_w_est or LANE_W_PRIOR
        if not (ONE_SIDE_PAIR[0] * lw <= right.x - left.x <= ONE_SIDE_PAIR[1] * lw):
            return
        alive = set(self._by_id)
        self._paired = {k: v for k, v in self._paired.items() if k in alive}
        for b in (left, right):
            self._paired[b.track_id] = (self._u, b.x)
        # 역산용: 쌍을 본 시각과 두 선의 번호 · 모양
        self._pair_lines = {"u": self._u, "ego_left": (left.track_id, left.line[np.argsort(left.line[:, 1])].copy()),
                            "ego_right": (right.track_id, right.line[np.argsort(right.line[:, 1])].copy())}

    def _one_side_ok(self) -> bool:
        """한쪽만 잃었고, 남은 쪽이 이번 프레임에 본 '짝으로 확인된 선'인지."""
        lost = [r for r in EGO_ROLES if self.miss[r] > self.miss_s]
        if len(lost) != 1:
            return False
        alive = "ego_right" if lost[0] == "ego_left" else "ego_left"
        if self.miss[alive] > 0.0:                 # 이번 프레임에 보고 그린 선이어야 한다
            return False
        tr = self._by_id.get(self.committed[alive] or -1)
        rec = self._paired.get(tr.tid) if tr is not None else None
        if rec is None:
            return False
        u, x = rec
        if (self._u - u) * self.dt > ONE_SIDE_SEC:
            return False
        return abs(tr.x - x) <= ONE_SIDE_DX * (self._lane_w_est or LANE_W_PRIOR)

    def _infer_other_side(self, left: Boundary | None, right: Boundary | None
                          ) -> tuple[Boundary | None, Boundary | None, str]:
        """한 줄만 그릴 때 반대쪽 선을 마지막 쌍에서 역산한다. 역산한 쪽을 "left" / "right" 로 돌려준다."""
        def shown(b: Boundary | None) -> bool:
            return b is not None and b.drawable(self.hold_s) and b.line is not None
        sl, sr = shown(left), shown(right)
        if sl == sr:
            return left, right, ""
        alive = "ego_left" if sl else "ego_right"
        b = left if sl else right
        if b.age_s != 0.0 or b.line is None or self.vp is None or self._lane_w_est is None:
            return left, right, ""
        tr = self._by_id.get(self.committed[alive] or -1)
        rec = self._paired.get(tr.tid) if tr is not None else None
        lw = float(self._lane_w_est)
        if rec is None or (self._u - rec[0]) * self.dt > INFER_SEC or abs(tr.x - rec[1]) > ONE_SIDE_DX * lw:
            return left, right, ""
        vy = float(self.vp[1])
        if self.ref_row - vy <= INFER_MIN_BELOW_VP:
            return left, right, ""
        sign = 1.0 if alive == "ego_left" else -1.0
        pair = self._pair_lines
        if pair is None or (self._u - pair["u"]) * self.dt > INFER_SEC:
            return left, right, ""
        other = "ego_right" if alive == "ego_left" else "ego_left"
        tid_then, seen_then = pair[alive]
        if tid_then != tr.tid:
            return left, right, ""
        # 보이는 선이 쌍을 본 뒤 움직인 양 d(y) = 지금 - 그때 를 직선 c0 + s·y 로 맞춘다
        now = np.asarray(b.line, dtype=np.float64)
        now = now[np.argsort(now[:, 1])]
        lo = max(float(now[0, 1]), float(seen_then[0, 1]), vy + INFER_MIN_BELOW_VP)
        hi = min(float(now[-1, 1]), float(seen_then[-1, 1]))
        if hi - lo < 0.05:
            return left, right, ""
        ys = np.linspace(lo, hi, 16)
        d = np.interp(ys, now[:, 1], now[:, 0]) - np.interp(ys, seen_then[:, 1], seen_then[:, 0])
        s, c0 = np.polyfit(ys, d, 1)
        resid = float(np.max(np.abs(d - (c0 + s * ys))))
        self._infer_info = {**self._infer_info, "infer_shape_resid": round(resid, 4)}
        if resid > INFER_SHAPE_TOL:
            return left, right, ""                 # 혼자 모양을 바꾼 선 — 같은 차로가 아니다
        # 반대쪽 선을 같은 만큼 옮긴다
        _, o = pair[other]
        y = o[:, 1].astype(np.float64)
        x = o[:, 0] + c0 + s * y
        keep = ((y - vy) > INFER_MIN_BELOW_VP) & (x >= 0.0) & (x <= 1.0)
        if int(keep.sum()) < INFER_MIN_POINTS:
            return left, right, ""
        derived = np.stack([x[keep], y[keep]], axis=1).astype(np.float32)
        nb = Boundary(track_id=-2, poly=derived, x=float(b.x + sign * lw), age_s=0.0, held=False,
                      line=derived, line_halfwidth=None, inferred=True)
        return (b, nb, "right") if alive == "ego_left" else (nb, b, "left")
