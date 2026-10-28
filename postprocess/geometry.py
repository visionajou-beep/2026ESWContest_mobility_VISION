"""차선 마스크(폴리곤)를 읽는 기하 함수.

좌표는 모두 화면 폭 · 높이로 나눈 0 ~ 1 값이다 (YOLO `result.masks.xyn` 그대로).
"""

from __future__ import annotations

import numpy as np

from .config import CROSS_MARGIN, LINE_PIECE_TOL, VP_TOGETHER_STRAIGHT

# 자차 좌우를 판단하는 기준행(화면 높이 비율)의 기본값. 실제 기준행은 추적기가 소실점에서 정한다.
REF_ROW = 0.65
# 세로 길이 / 가로 길이가 이보다 작으면 도로를 가로지르는 표시(정지선 · 횡단보도)다.
MIN_SLOPE = 0.20
# 기준행에서 이보다 가까운 두 선은 다른 차로가 아니라 한 도색의 조각이다.
MIN_LANE_GAP = 0.06
# 소실점 추정에 쓰려면 적어도 이만큼 세로로 길어야 한다 (짧으면 기울기를 믿을 수 없다).
VP_MIN_SPAN = 0.03


def x_at_row(poly: np.ndarray, row: float, tol: float = 0.02,
             extrapolate: bool = True, max_reach_ratio: float = 1.0) -> float | None:
    """폴리곤이 `row` 행에서 놓인 가로 위치.

    1. 폴리곤의 변이 행과 만나는 교점들의 평균 — 띠의 양쪽 변에서 교점이 나오므로 그 평균이 선의 중심이다.
    2. 교점이 없으면 (행이 꼭짓점을 정확히 지나는 경우) 행 ±tol 안 꼭짓점의 평균.
    3. 행까지 오지 않는 짧은 선은 x = a*y + b 로 늘여 읽는다. 단 폴리곤 길이의 `max_reach_ratio` 배까지만
       (짧은 조각을 멀리 늘이면 엉뚱한 위치가 나온다).
    """
    ys, xs = poly[:, 1], poly[:, 0]
    y1, x1 = np.roll(ys, -1), np.roll(xs, -1)      # 닫힌 폴리곤이므로 마지막 변도 본다
    dy = y1 - ys
    cross = ((ys - row) * (y1 - row) <= 0) & (dy != 0)
    if cross.any():
        t = (row - ys[cross]) / dy[cross]
        return float(np.mean(xs[cross] + t * (x1[cross] - xs[cross])))

    band = poly[np.abs(poly[:, 1] - row) <= tol]
    if band.size:
        return float(band[:, 0].mean())

    if not extrapolate:
        return None
    lo, hi = float(ys.min()), float(ys.max())
    gap = (lo - row) if row < lo else (row - hi)
    if gap > max_reach_ratio * max(hi - lo, 1e-6):
        return None                                 # 너무 멀리 늘여야 한다
    if hi - lo < 1e-4:
        return float(xs.mean())
    a, b = np.polyfit(ys, xs, 1)
    return float(a * row + b)


def is_longitudinal(poly: np.ndarray, center_x: float, row: float, center_test: bool = True) -> bool:
    """자차 차선의 경계가 될 수 있는 모양인지.

    - 도로를 가로지르는 모양(정지선 · 횡단보도 · 교차로의 다른 길)은 아니다.
    - `center_test` 이면, `row` 아래(차 가까운 쪽)에서 진행 경로(center_x)에 걸친 선도 아니다.
      먼 쪽은 보지 않는다 — 모든 차선은 소실점에서 경로와 만난다.
    """
    dx = float(poly[:, 0].max() - poly[:, 0].min())
    dy = float(poly[:, 1].max() - poly[:, 1].min())
    if dy / (dx + 1e-6) < MIN_SLOPE:
        return False
    if not center_test:
        return True
    near = poly[poly[:, 1] >= row]
    if near.size == 0:
        near = poly                                 # 먼 쪽에만 있으면 전체로 판단
    if near[:, 0].min() < center_x < near[:, 0].max():
        return False
    return True


def assign_roles(polys: list[np.ndarray], xs: list[float], sides: list[int], center_x: float,
                 row: float, max_dist: float, pair_w: float) -> dict[str, int]:
    """선들에 역할을 정한다: ego_left / ego_right / adj_left / adj_right -> 선 번호.

    - `xs`: 선마다 기준행 위치.
    - `sides`: 선마다 -1 왼쪽 / +1 오른쪽 / 0 모름. 0 이 아니면 그 쪽으로 보고 경로 걸침 검사를 건너뛴다
      (차로를 바꾸며 밟고 있는 선은 이번 프레임만 보면 쪽이 없지만, 추적기는 어느 쪽에서 왔는지 안다).
    - 각 쪽에서 경로에 가장 가까운 선이 자차 선, 그다음(MIN_LANE_GAP 이상 떨어진) 선이 옆 차로 선이다.
    - `max_dist`: 경로에서 이보다 먼 선은 자차 선이 될 수 없다 (진짜 자차 선을 놓쳤을 때 옆 선을 대신 그리지 않게).
      예외 — 반대쪽 자차 선이 안쪽에 있고 그 선과 0.7 ~ 1.3 차로폭(`pair_w`) 떨어져 있으면 받아들인다
      (차로를 바꾸는 중에는 한쪽 선이 경로에서 한 차로폭 떨어져 있다).
    - 옆 차로 선은 거리 제한 없이 찾는다.
    """
    left: list[tuple[float, int]] = []
    right: list[tuple[float, int]] = []
    for i, p in enumerate(polys):
        known = sides[i]
        if not is_longitudinal(p, center_x, row, center_test=(known == 0)):
            continue
        x = xs[i]
        on_left = (known < 0) if known else (x < center_x)
        (left if on_left else right).append((abs(x - center_x), i))
    left.sort()
    right.sort()
    left_all, right_all = list(left), list(right)

    def gated(side: list[tuple[float, int]], other: list[tuple[float, int]]) -> list[tuple[float, int]]:
        inside = [c for c in side if c[0] <= max_dist]
        if inside:
            return inside
        partner = [c for c in other if c[0] <= max_dist]
        if not partner:
            return []
        xo = xs[partner[0][1]]
        for d, i in side:                           # 가까운 것부터
            w = abs(xs[i] - xo)
            if 0.7 * pair_w <= w <= 1.3 * pair_w:
                return [(d, i)]
        return []

    left, right = gated(left, right), gated(right, left)

    def distinct(side: list[tuple[float, int]]) -> list[int]:
        """같은 도색의 두 조각을 두 차로로 세지 않는다 (MIN_LANE_GAP 안이면 하나)."""
        keep: list[int] = []
        for d, i in side:
            if all(abs(d - side[[k[1] for k in side].index(j)][0]) >= MIN_LANE_GAP
                   for j in keep):
                keep.append(i)
            if len(keep) == 2:
                break
        return keep

    roles: dict[str, int] = {}
    lk, rk = distinct(left), distinct(right)
    if len(lk) >= 1:
        roles["ego_left"] = lk[0]
    if len(rk) >= 1:
        roles["ego_right"] = rk[0]
    for role, ego, side_all in (("adj_left", "ego_left", left_all), ("adj_right", "ego_right", right_all)):
        if ego not in roles:
            continue
        d_ego = next(d for d, i in side_all if i == roles[ego])
        nxt = [i for d, i in side_all if d >= d_ego + MIN_LANE_GAP]
        if nxt:
            roles[role] = nxt[0]
    return roles


def estimate_vanishing_point(polys: list[np.ndarray]) -> tuple[float, float] | None:
    """차선들이 모이는 점(소실점). 선마다 x = a*y + b 로 맞추고, 두 선씩 만나는 점들의 중앙값을 쓴다.

    화면 가운데(0.5)가 정면이라는 보장은 없다 — 카메라마다 소실점이 가운데에서 벗어나 있다.
    """
    lines: list[tuple[float, float]] = []
    for q in polys:
        ys, xs = q[:, 1], q[:, 0]
        if float(ys.max() - ys.min()) < VP_MIN_SPAN:
            continue
        a, b = np.polyfit(ys, xs, 1)
        lines.append((float(a), float(b)))
    if len(lines) < 2:
        return None
    xs_, ys_ = [], []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            a1, b1 = lines[i]
            a2, b2 = lines[j]
            if abs(a1 - a2) < 1e-3:
                continue                            # 화면에서 평행 — 만나지 않는다
            y = (b2 - b1) / (a1 - a2)
            x = a1 * y + b1
            if -0.5 < x < 1.5 and -0.5 < y < 1.2:
                xs_.append(x)
                ys_.append(y)
    if not xs_:
        return None
    return float(np.median(xs_)), float(np.median(ys_))


def path_x_at_row(vp: tuple[float, float] | None, row: float) -> float:
    """차가 나아가는 경로(화면 아래 가운데 -> 소실점)가 `row` 행을 지나는 가로 위치."""
    if vp is None:
        return 0.5
    vx, vy = vp
    if not (0.0 <= vy < 1.0):
        return 0.5
    t = (1.0 - row) / (1.0 - vy)                    # 화면 아래 0, 소실점 1
    t = min(max(t, 0.0), 1.0)
    return 0.5 + t * (vx - 0.5)


def vp_miss(poly: np.ndarray, vp: tuple[float, float]) -> float | None:
    """마스크 양 끝을 잇는 직선이 소실점 높이에서 소실점을 얼마나 비켜 가는지 (화면 폭 비율).

    차선은 0.02 ~ 0.05 안이고, 사선 연석 · 화살표 같은 표시는 0.1 을 넘는다.
    """
    ys = poly[:, 1]
    lo, hi = float(ys.min()), float(ys.max())
    if hi - lo < 0.04:
        return None
    x_lo = x_at_row(poly, lo + 0.01, extrapolate=False)
    x_hi = x_at_row(poly, hi - 0.01, extrapolate=False)
    if x_lo is None or x_hi is None:
        return None
    a = (x_hi - x_lo) / ((hi - 0.01) - (lo + 0.01))
    return float(x_lo + a * (vp[1] - (lo + 0.01)) - vp[0])


def vp_together(x: float, miss: float, others, tol: float, center_x: float) -> bool:
    """경로 반대쪽에 떨어져 있는 다른 후보가 비슷한 만큼(tol 안) 같이 비켜 가는지 — 도로가 휘는 경우.

    어느 후보든 소실점을 지나면 곧은 도로이므로 아니다.
    """
    if any(m is not None and abs(m) <= VP_TOGETHER_STRAIGHT for _, m in others):
        return False
    return any(m is not None and abs(xo - x) >= MIN_LANE_GAP and abs(m - miss) <= tol
               and (xo - center_x) * (x - center_x) < 0 for xo, m in others)


def out_sign(x: float, center_x: float) -> int:
    """경로에서 바깥쪽 방향: 왼쪽 선 -1, 오른쪽 선 +1."""
    return -1 if x < center_x else 1


def side_of(x: float, center_x: float) -> int:
    """선이 경로의 어느 쪽인지: -1 왼쪽, +1 오른쪽, 경로에 걸치면 0."""
    d = x - center_x
    if abs(d) < CROSS_MARGIN:
        return 0
    return -1 if d < 0 else 1


def same_line(pa: np.ndarray, pb: np.ndarray) -> bool:
    """두 마스크가 한 도색인지 — 짧은 쪽이 긴 쪽에 맞춘 곡선에서 LINE_PIECE_TOL 안에 있으면 같은 선."""
    lng, sht = (pa, pb) if np.ptp(pa[:, 1]) >= np.ptp(pb[:, 1]) else (pb, pa)
    deg = 2 if np.ptp(lng[:, 1]) > 0.10 and len(lng) >= 6 else 1
    coef = np.polyfit(lng[:, 1], lng[:, 0], deg)
    return float(np.mean(np.abs(sht[:, 0] - np.polyval(coef, sht[:, 1])))) < LINE_PIECE_TOL
