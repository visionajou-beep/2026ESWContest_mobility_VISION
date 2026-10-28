"""정답 라벨 읽기와 정답 쪽 자차 선 고르기.

정답 데이터 폴더 하나 = 영상 하나:

    split_manifest.csv           정답 프레임 목록 (stem, frame, source, output_split …)
    labels/<split>/<stem>.txt    차선 폴리곤 (YOLO-seg 형식: 클래스 + 0 ~ 1 정규화 좌표)
    camera.json                  카메라(source)별 소실점 · 기준행 · 차로폭 — 정답 라벨에서 미리 계산한 값

정답의 '자차 선'은 사람이 표시한 것이 아니라 규칙으로 고른다: 기준행(소실점에서 화면 아래까지의 35.4 %)에서
진행 경로 양쪽으로 가장 가까운 정답 선. 이 규칙이 맞지 않는 장면은 채점에서 뺀다 (excluded_sides).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from postprocess.geometry import is_longitudinal, path_x_at_row, x_at_row

EGO = ("ego_left", "ego_right")
GT_MAX_DIST = 1.0      # 정답 자차 선이 경로에서 이 차로폭보다 멀면 진짜 자차 선이 라벨에 없는 것이다
STRADDLE = 0.03        # 차 바로 앞에서 정답 선이 경로에서 이만큼 안에 있으면 차가 선을 밟고 있다 (좌우가 애매)
NEAR_ROW_FRAC = 0.75   # 근거리 검사 행 = 기준행 ~ 화면 아래의 75 % 지점
NEAR_PAIR_MAX = 1.5    # 근거리 정답 좌 · 우 간격이 예상 차로폭의 이 배를 넘으면 한쪽은 한 차로 건너 선이다
NEAR_SAME_LINE = 0.3   # 근거리 선이 기준행에서 고른 선보다 이 차로폭 넘게 안쪽이면 다른 선이다


def read_labels(path: Path) -> list[np.ndarray]:
    """YOLO-seg 라벨 파일 -> (N, 2) 폴리곤 목록."""
    out: list[np.ndarray] = []
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return out
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 7:                          # 클래스 + 점 3 개 이상
            continue
        coords = np.asarray(parts[1:], dtype=np.float32)
        if coords.size % 2:
            coords = coords[:-1]
        out.append(coords.reshape(-1, 2))
    return out


def load_cameras(path: Path) -> dict[str, dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data:
        raise ValueError(f"camera.json 이 비어 있습니다: {path}")
    return data


def camera_for(cameras: dict[str, dict], row: dict) -> dict | None:
    """정답 프레임 한 장의 카메라 — 목록의 source 열로 찾고, 카메라가 하나뿐이면 그것."""
    src = row.get("source")
    if src and src in cameras:
        return cameras[src]
    if "default" in cameras:
        return cameras["default"]
    if len(cameras) == 1:
        return next(iter(cameras.values()))
    return None


def select_ego(gt: list[np.ndarray], cam: dict):
    """정답 쪽 자차 좌 · 우 선을 고른다.

    돌려주는 값: 소실점, 기준행, 진행 경로가 기준행을 지나는 위치, {역할: 정답 선 번호}, 정답에서 잰 기준행 차로폭(없으면 None).
    좌우는 기준행 아래(차 가까이)에서 판단하고, 기준행 위치로 가장 가까운 선을 고른다.
    """
    vp = (float(cam["vp_x"]), float(cam["vp_y"]))
    ref_row = float(cam["ref_row"])
    center_x = path_x_at_row(vp, ref_row)
    left: list[tuple[float, int]] = []
    right: list[tuple[float, int]] = []
    for i, p in enumerate(gt):
        if not is_longitudinal(p, center_x, ref_row):
            continue
        x = x_at_row(p, ref_row)
        if x is None:
            continue
        (left if x < center_x else right).append((abs(x - center_x), i))
    left.sort()
    right.sort()
    roles = {}
    if left:
        roles["ego_left"] = left[0][1]
    if right:
        roles["ego_right"] = right[0][1]
    lane_w = None
    if "ego_left" in roles and "ego_right" in roles:
        xl, xr = x_at_row(gt[roles["ego_left"]], ref_row), x_at_row(gt[roles["ego_right"]], ref_row)
        if xl is not None and xr is not None and abs(xr - xl) > 0.05:
            lane_w = abs(xr - xl)
    return vp, ref_row, center_x, roles, lane_w


def _x_extended(q, y: float) -> float | None:
    """정답 선 하나의 y 행 위치 — 범위 안이면 보간, 벗어나면 선의 아래쪽 절반(짧으면 전체)을 직선으로 맞춰 늘인다."""
    q = np.asarray(q, dtype=float)
    q = q[np.argsort(q[:, 1])]
    if len(q) < 2:
        return None
    if q[0, 1] <= y <= q[-1, 1]:
        return float(np.interp(y, q[:, 1], q[:, 0]))
    lo = q[-1, 1] - 0.5 * (q[-1, 1] - q[0, 1])
    s = q[q[:, 1] >= lo]
    if len(s) < 3 or s[-1, 1] - s[0, 1] < 0.03:
        s = q                      # 짧은 선은 아래쪽 몇 점이 같은 높이에 몰려 기울기가 무너지므로 선 전체로 맞춘다
    a, b = np.polyfit(s[:, 1], s[:, 0], 1)
    return float(a * y + b)


def _near_check(gt: list, roles: dict, vp, ref_row: float, cam_lw: float) -> set:
    """근거리 검사 — 정답 선을 차 바로 앞에만 짧게 그려 기준행까지 오지 않으면, 기준행에서는 한 차로 건너 선이
    자차 선으로 뽑힌다. 근거리 행에서 늘이지 않고 다시 골라 보고, 맞지 않는 쪽을 돌려준다.

    1. 근거리에서 고른 선이 기준행에서 고른 선보다 안쪽(경로에 가까움)으로 차로폭의 NEAR_SAME_LINE 배 넘게 있으면 그 쪽.
       같은 도색을 조각으로 나눠 그은 라벨은 같은 선으로 본다 (기준행에서 고른 선을 근거리 행까지 늘여 비교).
    2. 근거리 좌 · 우 정답 선 간격이 예상 차로폭의 NEAR_PAIR_MAX 배를 넘으면 경로에서 먼 쪽 (그 쪽 자차 선이 라벨에 없다).
    """
    y = ref_row + NEAR_ROW_FRAC * (1.0 - ref_row)
    path = path_x_at_row(vp, y)
    xs = [(i, x_at_row(q, y, extrapolate=False)) for i, q in enumerate(gt)]
    xs = [(i, float(x)) for i, x in xs if x is not None]
    near = {}
    left = [t for t in xs if t[1] < path]
    right = [t for t in xs if t[1] > path]
    if left:
        near["ego_left"] = max(left, key=lambda t: t[1])
    if right:
        near["ego_right"] = min(right, key=lambda t: t[1])
    bad = set()
    expect = cam_lw * (y - vp[1]) / (ref_row - vp[1]) if vp is not None and ref_row - vp[1] > 1e-3 else None
    for r in EGO:
        if r in roles and r in near and near[r][0] != roles[r] and expect is not None:
            xc = _x_extended(gt[roles[r]], y)
            if xc is None:
                continue
            inner_by = (near[r][1] - xc) if r == "ego_left" else (xc - near[r][1])
            if inner_by > NEAR_SAME_LINE * expect:
                bad.add(r)
    if "ego_left" in near and "ego_right" in near and expect is not None:
        width = near["ego_right"][1] - near["ego_left"][1]
        if expect > 0 and width > NEAR_PAIR_MAX * expect:
            bad.add("ego_left" if (path - near["ego_left"][1]) > (near["ego_right"][1] - path) else "ego_right")
    return bad


def excluded_sides(gt: list, roles: dict, vp, ref_row: float, center_x: float, cam_lw: float) -> tuple[bool, set]:
    """채점에서 뺄 경우: (차가 선을 밟고 있는지 — 그러면 좌 · 우 모두, 뺄 쪽 집합)."""
    straddle = False
    for q in gt:
        y_near = float(q[:, 1].max()) - 0.005
        if y_near < ref_row + 0.4 * (1.0 - ref_row):
            continue                                # 차 바로 앞까지 내려오지 않은 선
        xn = x_at_row(q, y_near, extrapolate=False)
        if xn is not None and abs(xn - path_x_at_row(vp, y_near)) < STRADDLE:
            straddle = True
    bad = set()
    for r in EGO:
        if r in roles:
            gx = x_at_row(gt[roles[r]], ref_row)
            if gx is not None and abs(gx - center_x) > GT_MAX_DIST * cam_lw:
                bad.add(r)                          # 정답 자차 선이 한 차로폭보다 멀다
    bad |= _near_check(gt, roles, vp, ref_row, cam_lw)
    return straddle, bad
