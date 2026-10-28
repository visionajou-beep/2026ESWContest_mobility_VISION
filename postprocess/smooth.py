"""선 다듬기 — 마스크 조각들에서 자차 선마다 매끄러운 중심선 하나를 만든다.

점선은 프레임마다 조각 수가 달라 마스크를 그대로 그리면 선의 길이와 먼 끝이 튄다.
같은 선 위의 조각을 모두 모아 고정된 행마다 중심을 읽고, 행마다 EMA 로 다듬는다.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .config import (
    FAR_KINK_MAX,
    GUARD_MIN_OBS_ROWS,
    LINE_ALPHA_EXT,
    LINE_ALPHA_WIDTH,
    LINE_ALPHA_X,
    LINE_CONTINUE_FIT_ROWS,
    LINE_EDGE_EPS,
    LINE_MAX_EXT_STEP,
    LINE_PIECE_TOL,
    LINE_ROWS,
    LINE_SMOOTH_ROWS,
    WIDE_ROW_FACTOR,
    WIDE_ROW_FALLBACK,
    WIDE_ROW_MIN_W,
)

if TYPE_CHECKING:
    from .lane import Boundary


def ema_alpha(alpha: float, step: float) -> float:
    """한 프레임이 기준 프레임 `step` 개 길이일 때의 EMA 가중치 — 처리 속도와 상관없이 같은 시간 상수를 지킨다."""
    if step == 1.0:
        return alpha
    if step <= 0.0:
        return 0.0
    return 1.0 - (1.0 - alpha) ** step


def _smooth_along_rows(x: np.ndarray, half: int) -> np.ndarray:
    """연속한 2*half+1 행마다 직선 맞춤 (Savitzky-Golay 1 차). 이동평균과 달리 양 끝이 안쪽으로 휘지 않는다."""
    if half <= 0 or len(x) < 3:
        return x
    n = len(x)
    t = np.arange(n, dtype=float)
    # 모든 창의 최소제곱을 누적합으로 한 번에 계산한다
    lo = np.clip(np.arange(n) - half, 0, n)
    hi = np.clip(np.arange(n) + half + 1, 0, n)
    c1, ct, cx, ctt, ctx = (np.concatenate([[0.0], np.cumsum(v)]) for v in (np.ones(n), t, x, t * t, t * x))
    m = hi - lo
    st, sx, stt, stx = ct[hi] - ct[lo], cx[hi] - cx[lo], ctt[hi] - ctt[lo], ctx[hi] - ctx[lo]
    tm, xm = st / m, sx / m
    var = stt - m * tm * tm
    cov = stx - m * tm * xm
    a = np.where(var > 0, cov / np.where(var > 0, var, 1.0), 0.0)
    out = xm + a * (t - tm)
    return np.where(m < 3, xm, out)


def _row_crossings_all(polys: list[np.ndarray], rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """행마다 모든 폴리곤 변과의 가장 왼쪽 · 가장 오른쪽 교점 (교점이 둘 미만이면 NaN)."""
    lo = np.full(rows.shape, np.inf)
    hi = np.full(rows.shape, -np.inf)
    cnt = np.zeros(rows.shape, dtype=int)
    for q in polys:
        x0, y0 = q[:, 0], q[:, 1]
        x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
        dy = y1 - y0
        valid = dy != 0
        r = rows[:, None]
        cross = ((y0[None, :] - r) * (y1[None, :] - r) <= 0) & valid[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (r - y0[None, :]) / np.where(valid, dy, 1.0)[None, :]
        xs = x0[None, :] + t * (x1 - x0)[None, :]
        xs = np.where(cross, xs, np.nan)
        cnt += cross.sum(1)
        lo = np.fmin(lo, np.nanmin(np.where(cross, xs, np.inf), axis=1))
        hi = np.fmax(hi, np.nanmax(np.where(cross, xs, -np.inf), axis=1))
    two = cnt >= 2
    return np.where(two, lo, np.nan), np.where(two, hi, np.nan)


def _continue_line(x: np.ndarray, hw: np.ndarray, i0: int, i1: int,
                   n: int = LINE_CONTINUE_FIT_ROWS) -> None:
    """[i0, i1] 밖의 행을 양 끝 근처 행으로 맞춘 직선으로 채운다."""
    if i1 - i0 < 1:
        return
    rows = LINE_ROWS
    if i0 > 0:
        s = slice(i0, min(i0 + n, i1 + 1))
        x[:i0] = np.polyval(np.polyfit(rows[s], x[s], 1), rows[:i0])
        hw[:i0] = hw[i0]
    if i1 < len(rows) - 1:
        s = slice(max(i1 - n + 1, i0), i1 + 1)
        x[i1 + 1:] = np.polyval(np.polyfit(rows[s], x[s], 1), rows[i1 + 1:])
        hw[i1 + 1:] = hw[i1]


class ShapeStabilizer:
    """자차 좌 · 우(와 셋째 선)마다 다듬은 중심선을 하나씩 유지한다.

    이번 프레임에 본 선이면 그 선 근처의 조각을 모아 행마다 중심을 읽고, 이전 선과 EMA 로 섞는다.
    유지 중인 선은 마지막 모양을 그대로 쓰고, 선 번호가 바뀌면 새로 시작한다 (다른 선을 섞지 않는다).
    """

    def __init__(self) -> None:
        self.state: dict[str, dict] = {}
        self.lane_w: float | None = None   # 기준행 차로폭 (추적기가 매 프레임 넣는다)
        self.ref_row: float | None = None  # 기준행
        self.vp_y: float | None = None     # 소실점 행 — 원근에 따른 도색 폭 계산용
        self.k_hw: float | None = None     # 소실점 아래 깊이당 도색 폭의 절반 (카메라 상수라 천천히 배운다)

    def _trim_far_kink(self, st: dict, sel: np.ndarray) -> np.ndarray:
        """선의 먼 쪽 절반에서 처음 나오는 급한 꺾임 너머를 버린다.

        출력만 자르면 다음 프레임에 다시 길어져 깜빡이므로 선의 끝(y_top) 자체를 옮긴다.
        끝은 이후 정해진 속도로 천천히 다시 자란다.
        """
        x = st["x"]
        idx = np.flatnonzero(sel)
        vy, rr, lw = self.vp_y, self.ref_row, self.lane_w
        if len(idx) < 5 or vy is None or rr is None or not lw or rr - vy <= 1e-3:
            return sel
        xs, ys = x[idx], LINE_ROWS[idx]
        lw_row = lw * np.clip(ys - vy, 1e-3, None) / (rr - vy)
        mid = vy + 0.5 * (float(ys.max()) - vy)          # 먼 쪽 절반의 경계
        cut = None
        for k in range(len(idx) - 2, 0, -1):             # 가까운 쪽에서 먼 쪽으로
            if ys[k] > mid:
                continue
            if abs(xs[k + 1] - 2.0 * xs[k] + xs[k - 1]) / lw_row[k] > FAR_KINK_MAX:
                cut = k
                break
        if cut is None:
            return sel
        out = sel.copy()
        out[idx[:cut]] = False                           # 꺾임보다 먼 행
        st["y_top"] = max(st["y_top"], float(LINE_ROWS[idx[cut]]))
        return out

    def update(self, role: str, b: "Boundary | None", polys: list[np.ndarray],
               hold_s: float, step: float = 1.0) -> np.ndarray | None:
        """이번 프레임의 중심선을 계산해 `b.line` · `b.line_halfwidth` 에 넣는다. `step` 은 이번 프레임의 길이(기준 프레임 수)."""
        if b is None or not b.drawable(hold_s):
            self.state.pop(role, None)
            return None
        a_x = ema_alpha(LINE_ALPHA_X, step)
        a_ext = ema_alpha(LINE_ALPHA_EXT, step)
        max_ext = LINE_MAX_EXT_STEP * step
        st = self.state.get(role)
        if st is not None and st["tid"] != b.track_id:
            st = None
        if b.age_s == 0.0:
            # 같은 선의 조각 모으기: 추적 중인 폴리곤에 맞춘 곡선(길면 2 차)에서 가까운 조각
            span_p = float(b.poly[:, 1].max() - b.poly[:, 1].min())
            deg = 2 if span_p > 0.10 and len(b.poly) >= 6 else 1
            coef = np.polyfit(b.poly[:, 1], b.poly[:, 0], deg)
            pieces = [q for q in polys
                      if float(np.mean(np.abs(q[:, 0] - np.polyval(coef, q[:, 1])))) < LINE_PIECE_TOL]
            if not pieces or b.from_outer:
                pieces = [b.poly]                        # 이중선 바깥 획을 옮긴 경우 옆 마스크를 섞지 않는다
            pts = np.concatenate(pieces)
            y_top, y_bot = float(pts[:, 1].min()), float(pts[:, 1].max())
            inside = (LINE_ROWS >= y_top) & (LINE_ROWS <= y_bot)
            # 행마다 가장 바깥 두 교점의 가운데를 중심으로 읽는다 (한 곡선으로 맞추면 커브가 곧게 그려진다)
            x_obs = np.full(LINE_ROWS.shape, np.nan)
            w_obs = np.full(LINE_ROWS.shape, np.nan)
            lo_x, hi_x = _row_crossings_all(pieces, LINE_ROWS)
            ok = inside & ~np.isnan(lo_x) & ((hi_x - lo_x) > 1e-4)
            with np.errstate(invalid="ignore"):
                ok &= ~((lo_x <= LINE_EDGE_EPS) | (hi_x >= 1.0 - LINE_EDGE_EPS))
                vy = self.vp_y
                if self.k_hw is not None and vy is not None:
                    expect = np.maximum(2.0 * self.k_hw * np.clip(LINE_ROWS - vy, 0.0, None), WIDE_ROW_MIN_W)
                    ok &= ~((hi_x - lo_x) > WIDE_ROW_FACTOR * expect)
                else:
                    ok &= ~((hi_x - lo_x) > WIDE_ROW_FALLBACK)
            x_obs[ok] = 0.5 * (hi_x[ok] + lo_x[ok])
            w_obs[ok] = 0.5 * (hi_x[ok] - lo_x[ok])
            have = inside & ~np.isnan(x_obs)
            if have.sum() >= 2:
                # 선의 양 끝 = 도색이 실제로 읽힌 첫 행 · 마지막 행
                idx = np.flatnonzero(have)
                y_top, y_bot = float(LINE_ROWS[idx[0]]), float(LINE_ROWS[idx[-1]])
                inside = (LINE_ROWS >= y_top) & (LINE_ROWS <= y_bot)
                if idx[-1] - idx[0] + 1 < GUARD_MIN_OBS_ROWS:
                    return None
            gap = inside & np.isnan(x_obs)
            if have.sum() >= 2 and gap.any():
                x_obs[gap] = np.interp(LINE_ROWS[gap], LINE_ROWS[have], x_obs[have])
                w_obs[gap] = np.interp(LINE_ROWS[gap], LINE_ROWS[have], np.nan_to_num(w_obs[have], nan=0.004))
            elif have.sum() < 2:
                return None
            x_obs[inside] = _smooth_along_rows(x_obs[inside], LINE_SMOOTH_ROWS)
            if st is None:
                x = np.full(LINE_ROWS.shape, np.nan)
                x[inside] = x_obs[inside]
                hw = np.full(LINE_ROWS.shape, np.nan)
                hw[inside] = w_obs[inside]
                st = {"tid": b.track_id, "x": x, "hw": hw, "y_top": y_top, "y_bot": y_bot}
            else:
                x, hw = st["x"], st["hw"]
                known = inside & ~np.isnan(x)
                new = inside & np.isnan(x)
                x[known] += a_x * (x_obs[known] - x[known])
                x[new] = x_obs[new]
                wk = known & ~np.isnan(w_obs) & ~np.isnan(hw)
                hw[wk] += a_x * (w_obs[wk] - hw[wk])
                hw[new] = w_obs[new]
                for k, v in (("y_top", y_top), ("y_bot", y_bot)):
                    d = float(np.clip(a_ext * (v - st[k]), -max_ext, max_ext))
                    st[k] += d
            # 관측이 없는 행은 이전 값을 남기지 않고 이번 선을 같은 방향으로 잇는다 (화면 아래 모서리의 갈고리 방지)
            idx = np.flatnonzero(inside)
            _continue_line(st["x"], st["hw"], int(idx[0]), int(idx[-1]))
        if st is None:
            return None
        self.state[role] = st
        sel = (LINE_ROWS >= st["y_top"]) & (LINE_ROWS <= st["y_bot"]) & ~np.isnan(st["x"])
        if sel.sum() < 2:
            return None
        sel = self._trim_far_kink(st, sel)
        if sel.sum() < 2:
            return None
        line = np.stack([st["x"][sel], LINE_ROWS[sel]], 1).astype(np.float32)
        b.line = line
        # 도색 폭은 소실점 아래 깊이에 비례한다. 마스크 폭은 차 가까이에서만 믿을 만해서
        # 가까운 절반에서 폭 = k * 깊이 의 k 를 구하고 그 k 로 전체 폭을 정한다.
        hw = st["hw"][sel]
        rows = LINE_ROWS[sel]
        vp_y = self.vp_y
        ok = ~np.isnan(hw)
        if vp_y is not None:
            depth = rows - vp_y
            near = ok & (rows >= np.median(rows[ok])) & (depth > 1e-3) if ok.sum() >= 3 else np.zeros_like(ok)
            if near.sum() >= 2 and b.age_s == 0.0:
                k = float(np.median(hw[near] / depth[near]))
                self.k_hw = k if self.k_hw is None else self.k_hw + ema_alpha(LINE_ALPHA_WIDTH, step) * (k - self.k_hw)
            if self.k_hw is not None:
                hw = np.clip(self.k_hw * depth, 0.0, None)
        b.line_halfwidth = np.nan_to_num(hw, nan=0.004).astype(np.float32)
        return line
