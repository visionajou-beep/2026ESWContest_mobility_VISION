"""선 오차 — 화면에 그린 자차 선이 정답 도색에서 옆으로 얼마나 벗어났는지 (주 평가 지표).

좌 · 우 선마다, 정답 중심선과 그린 선이 둘 다 지나는 높이 전부(화면 높이 0.005 간격)에서
'가로 거리 ÷ 그 높이의 정답 차로폭'을 평균한다. 차로폭은 정답 좌 · 우 선의 간격으로만 잰다
(실제 차로폭 3.5 m 같은 가정을 쓰지 않는다).

    정확      평균 5 % 안
    어긋남    5 ~ 10 %
    틀린 선   10 % 초과 — 방향(바깥 = 옆 차로 쪽, 안쪽 = 내 차로 안쪽)도 나눈다
    미표시    정답 자차 선이 있는데 그리지 않음 (안전한 실패)

실행 (정답 데이터 폴더와 predict.py 가 만든 모델 출력 캐시가 필요하다):

    python eval/line_error.py --set day_clear=data/kr_day_clear --set rain_highway=data/kr_rain_highway \\
        --cache cache --out results
"""

from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from postprocess.geometry import x_at_row  # noqa: E402

STEP = 0.005                    # 비교하는 높이 간격 (화면 높이 비율)
OK_FRAC, WRONG_FRAC = 0.05, 0.10
LANE_M = 3.5                    # 표에 참고로 붙이는 cm 환산용 (판정에는 쓰지 않는다)


def x_on_line(line: np.ndarray | None, y: float) -> float | None:
    """그린 선의 높이 y 에서의 x. 선이 그 높이에 닿지 않으면 None (늘이지 않는다)."""
    if line is None or len(line) < 2:
        return None
    ys = line[:, 1]
    if y < ys.min() - 0.005 or y > ys.max() + 0.005:
        return None
    o = np.argsort(ys)
    return float(np.interp(y, ys[o], line[:, 0][o]))


def line_error(g: np.ndarray, line: np.ndarray, gt_pair: dict, vy: float, ref_row: float, lane_w_fb: float,
               out_sign: float):
    """정답 선 g 와 그린 선 하나의 선 오차.

    돌려주는 값: (평균 비율, 최대 비율, 잰 높이 수, 방향 있는 평균 — 차로 바깥쪽이 +). 잰 높이가 3 개 미만이면 None.
    차로폭은 그 높이의 정답 좌 · 우 간격이고, 한쪽 정답이 그 높이에 없으면 같은 장면의 정답 쌍에서 구한
    원근 비례식(간격 = k * (y - 소실점 y))으로, 정답 쌍이 아예 없으면 카메라 차로폭으로 대신한다.
    out_sign: 오른쪽 선 +1, 왼쪽 선 -1 (왼쪽은 x 가 작아지는 쪽이 바깥).
    """
    y_lo, y_hi = float(line[:, 1].min()), float(line[:, 1].max())
    gy = g[:, 1]
    top = max(float(gy.min()), y_lo, vy + 0.02)
    bot = min(float(gy.max()), y_hi)
    if bot - top < STEP:
        return None
    ys = np.arange(top, bot + 1e-9, STEP)
    gl, gr = gt_pair.get("ego_left"), gt_pair.get("ego_right")
    ks = []
    pair_w = {}
    if gl is not None and gr is not None:
        for y in np.arange(vy + 0.02, 1.0, STEP):
            l, r = x_at_row(gl, y, extrapolate=False), x_at_row(gr, y, extrapolate=False)
            if l is not None and r is not None and r - l > 0.02:
                pair_w[round(y, 4)] = r - l
                ks.append((r - l) / (y - vy))
    k = float(np.median(ks)) if ks else lane_w_fb / max(ref_row - vy, 1e-3)
    errs, signed = [], []
    for y in ys:
        gx, px = x_at_row(g, y, extrapolate=False), x_on_line(line, y)
        if gx is None or px is None:
            continue
        w = pair_w.get(round(y, 4)) or k * (y - vy)
        if w > 1e-4:
            errs.append(abs(px - gx) / w)
            signed.append(out_sign * (px - gx) / w)
    if len(errs) < 3:
        return None
    return float(np.mean(errs)), float(np.max(errs)), len(errs), float(np.mean(signed))


def judge(rec: dict) -> str:
    """ok · off · wrong · missed · nooverlap(그렸지만 정답과 같은 높이가 거의 없어 못 잰 쪽)."""
    if rec["status"] == "missed":
        return "missed"
    if rec["mean"] is None:
        return "nooverlap"
    f = rec["mean"]
    return "ok" if f <= OK_FRAC else ("off" if f <= WRONG_FRAC else "wrong")


def evaluate_set(args) -> list[dict]:
    """정답 데이터 하나(영상 하나)의 모든 채점 대상 쪽."""
    name, data_dir, cache_dir = args
    from score import MATCH_MAX, load_clips, score_clip
    recs = []
    for item in load_clips(Path(data_dir), Path(cache_dir)):
        if item["match"] > MATCH_MAX:
            continue
        for s in score_clip(item):
            rec = {"set": name, "stem": s["stem"], "half": s["half"], "role": s["role"], "status": s["status"],
                   "inferred": s["inferred"], "mean": None, "max": None, "rows": 0, "signed": None}
            if s["status"] == "drawn":
                res = line_error(s["gt"], s["line"], s["pair"], s["vy"], s["ref_row"], s["lane_w_fb"],
                                 out_sign=-1.0 if s["role"] == "ego_left" else 1.0)
                if res is not None:
                    rec["mean"], rec["max"], rec["rows"], rec["signed"] = res
            recs.append(rec)
    return recs


def summary_rows(recs: list[dict], groups: list[tuple[str, set]], half: str) -> list[str]:
    head = ["| 묶음 | 정답 있는 쪽 | 정확 (5 % 안) | 어긋남 (5 ~ 10 %) | 틀린 선 (10 % 초과) | 미표시 | 못 잰 쪽 | 오차 중앙 | 90 % |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    lines = list(head)
    for label, sets in groups:
        rr = [r for r in recs if r["half"] == half and r["set"] in sets]
        n = len(rr)
        if not n:
            continue
        c = {k: sum(judge(r) == k for r in rr) for k in ("ok", "off", "wrong", "missed", "nooverlap")}
        m = [r["mean"] for r in rr if r["mean"] is not None]
        pct = lambda k: f"{c[k] / n * 100:.1f} % ({c[k]})"
        fmt = lambda f: f"{f * 100:.1f} % (≈ {f * LANE_M * 100:.1f} cm)"
        lines.append(f"| {label} | {n} | **{pct('ok')}** | {pct('off')} | {pct('wrong')} | {pct('missed')} | {c['nooverlap']} | "
                     f"{fmt(float(np.median(m))) if m else '-'} | {fmt(float(np.percentile(m, 90))) if m else '-'} |")
    return lines


def side_rows(recs: list[dict], groups: list[tuple[str, set]], half: str) -> list[str]:
    def kind(r):
        k = judge(r)
        if k in ("off", "wrong"):
            return f"{k}_{'out' if (r['signed'] or 0.0) > 0 else 'in'}"
        return k
    lines = ["| 묶음 | 선 | 정답 있는 쪽 | 정확 | 어긋남 바깥 · 안쪽 | 틀린 선 바깥 · 안쪽 | 미표시 | 치우침 중앙 (+ 바깥) |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for label, sets in groups:
        for role, name in (("ego_left", "왼쪽"), ("ego_right", "오른쪽")):
            rr = [r for r in recs if r["half"] == half and r["set"] in sets and r["role"] == role]
            n = len(rr)
            if not n:
                continue
            c = {k: sum(kind(r) == k for r in rr) for k in ("ok", "off_out", "off_in", "wrong_out", "wrong_in", "missed")}
            s = [r["signed"] for r in rr if r["signed"] is not None]
            pct = lambda k: f"{c[k] / n * 100:.1f} %"
            lines.append(f"| {label} | {name} | {n} | **{pct('ok')}** | {c['off_out']} · {c['off_in']} | "
                         f"**{c['wrong_out']}** · {c['wrong_in']} | {pct('missed')} | "
                         f"{f'{np.median(s) * 100:+.2f} %' if s else '-'} |")
    return lines


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--set", action="append", required=True, metavar="NAME=DATA_DIR",
                    help="정답 데이터 폴더 (여러 번). 모델 출력은 <cache>/<NAME>/ 에서 읽는다")
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--half", default="report", choices=["dev", "report"])
    a = ap.parse_args(argv)
    sets = [s.split("=", 1) for s in a.set]
    jobs = [(name, d, a.cache / name) for name, d in sets]
    recs: list[dict] = []
    with ProcessPoolExecutor(max_workers=len(jobs)) as ex:
        for rr in ex.map(evaluate_set, jobs):
            recs += rr
    a.out.mkdir(parents=True, exist_ok=True)
    keys = ("set", "stem", "half", "role", "status", "inferred", "mean", "max", "rows", "signed")
    with open(a.out / "line_error.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in recs:
            w.writerow({k: ("" if r[k] is None else (round(r[k], 6) if isinstance(r[k], float) else r[k])) for k in keys})
    names = [name for name, _ in sets]
    groups = [(n, {n}) for n in names] + ([("합계", set(names))] if len(names) > 1 else [])
    lines = [f"# 선 오차 ({a.half})", "", "선마다 겹치는 높이 전부의 '가로 거리 ÷ 그 높이의 정답 차로폭' 평균. "
             "괄호의 cm 는 3.5 m 차로라면의 참고 환산.", ""]
    lines += summary_rows(recs, groups, a.half) + ["", "## 좌 · 우 선, 바깥 · 안쪽", ""] + side_rows(recs, groups, a.half)
    (a.out / f"line_error_{a.half}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
