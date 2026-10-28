"""정답 영상 채점 준비 — 정답 프레임까지 후처리를 돌리고, 화면에 그린 자차 좌 · 우 선을 정답 선과 짝짓는다.

정답 프레임마다 그 앞 6 초의 모델 출력(predict.py 가 저장)을 처음부터 추적기에 넣고, 정답 프레임에서 화면에 그리는 선
(`TrackerOutput.display_line`)을 꺼낸다. 추적기는 카메라의 소실점 · 차로폭(camera.json)을 알고 시작한다
(실제로는 계속 돌고 있어 이미 자리 잡은 상태이기 때문이다).

영상마다 정답 프레임을 시간 순으로 나눠 앞 절반은 개발용(dev, 설정 고르기), 뒤 절반은 보고용(report)으로 쓴다.
"""

from __future__ import annotations

import csv
import gzip
import pickle
from pathlib import Path

import numpy as np

from ground_truth import EGO, camera_for, excluded_sides, load_cameras, read_labels, select_ego
from postprocess import EgoLaneTracker

MATCH_MAX = 6.0        # 저장된 클립의 채점 프레임이 정답 사진과 닮은 정도(64x36 회색 평균 차)의 한계 — 넘으면 다른 장면


def load_clips(data_dir: Path, cache_dir: Path) -> list[dict]:
    """정답 데이터 폴더와 모델 출력 캐시에서 클립 목록을 만든다."""
    data_dir, cache_dir = Path(data_dir), Path(cache_dir)
    cams = load_cameras(data_dir / "camera.json")
    rows = list(csv.DictReader(open(data_dir / "split_manifest.csv", encoding="utf-8")))
    rows.sort(key=lambda r: int(r["frame"]))
    half = len(rows) // 2
    items = []
    for k, r in enumerate(rows):
        f = cache_dir / f"{r['stem']}.pkl.gz"
        if not f.exists():
            continue
        with gzip.open(f, "rb") as fh:
            clip = pickle.load(fh)
        gt = read_labels(data_dir / "labels" / r["output_split"] / f"{r['stem']}.txt")
        items.append({"stem": r["stem"], "frame": int(r["frame"]), "half": "dev" if k < half else "report",
                      "cam": camera_for(cams, r), "gt": gt, "polys": clip["polys"], "match": clip["match"]})
    return items


def run_tracker(item: dict):
    """클립 하나를 처음부터 추적기에 넣고 정답 프레임(마지막 프레임)의 결과를 돌려준다."""
    cam = item["cam"]
    tr = EgoLaneTracker(30.0, hold_s=0.15, miss_s=0.30, hysteresis_s=0.25)
    tr.vp, tr._vp_n = (float(cam["vp_x"]), float(cam["vp_y"])), 10_000
    lw = float(cam["lane_width_norm"])
    tr._lane_w_est = lw
    tr._lane_w_hist.extend([lw] * 15)
    out = None
    for ps in item["polys"]:
        out = tr.update(ps)
    return out


def score_clip(item: dict) -> list[dict]:
    """클립 하나의 채점 대상 쪽(좌 · 우)마다 정답 선과 그린 선을 짝지어 돌려준다.

    채점 대상: 정답 자차 선이 있고 빼는 경우(ground_truth.excluded_sides)가 아닌 쪽.
      - 그 쪽을 안 그렸으면 status = "missed" (미표시)
      - 그렸으면 status = "drawn" — 단, 그린 선이 기준행에 닿지 않으면 정답과 같은 자리에서 비교할 수 없어 뺀다
    """
    out = run_tracker(item)
    gt, cam = item["gt"], item["cam"]
    vp, ref_row, center_x, roles, lane_w_meas = select_ego(gt, cam)
    cam_lw = float(cam.get("lane_width_norm") or 0.32)
    straddle, bad = excluded_sides(gt, roles, vp, ref_row, center_x, cam_lw)
    if straddle:
        return []
    usable = {r: (r in roles and r not in bad) for r in EGO}
    pair = {r: (gt[roles[r]] if usable[r] else None) for r in EGO}   # 선 오차의 차로폭은 이 정답 쌍에서 잰다
    lane_w_fb = float(cam.get("lane_width_norm") or lane_w_meas or 0.32)
    sides = []
    for r in EGO:
        if not usable[r]:
            continue
        b = getattr(out, r) if out is not None else None
        line = out.display_line(b) if out is not None else None
        if line is None:
            status = "missed"
        else:
            ys = line[:, 1]
            if len(line) < 2 or not (ys.min() - 0.005 <= ref_row <= ys.max() + 0.005):
                continue                            # 기준행에 닿지 않는 선 — 비교할 수 없다
            status = "drawn"
        sides.append({"stem": item["stem"], "half": item["half"], "role": r, "status": status,
                      "gt": gt[roles[r]], "line": line, "inferred": bool(b is not None and b.inferred and line is not None),
                      "pair": pair, "vy": float(vp[1]), "ref_row": ref_row, "lane_w_fb": lane_w_fb})
    return sides
