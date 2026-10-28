"""정답 프레임마다 그 앞 6 초를 모델로 추론해 저장한다 (정답 영상 채점의 입력).

정답 프레임은 원본 영상에서 띄엄띄엄 뽑은 것이라 그 자체로는 추적기를 돌릴 수 없다. 그래서 정답 프레임마다
앞 `--seconds` 초 ~ 뒤 `--after` 초를 원본 영상에서 읽고, 그중 정답 사진과 가장 닮은 프레임을 채점 시점으로 삼아
그 프레임까지 모델 출력(차선 폴리곤)을 저장한다. 영상 탐색이 몇 프레임 어긋날 수 있어서 시점을 사진으로 맞춘다.

    python eval/predict.py --data data/kr_rain_highway --images data/kr_rain_highway/frames \\
        --video videos/rain_highway.mp4 --weights best.pt --out cache/rain_highway

결과: <out>/<stem>.pkl.gz (프레임별 폴리곤 목록 · 채점 시점 · 사진과 닮은 정도)
이미 만든 클립은 건너뛰므로 중간에 멈춰도 다시 돌리면 이어서 한다.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import pickle
import sys
import time
from pathlib import Path

import numpy as np

MATCH_MAX = 6.0            # 64x36 회색 평균 차 — 같은 장면이면 1 ~ 3, 다른 장면이면 20 이상
IMGSZ = (544, 960)         # 모델 입력 (높이, 폭)


def grey_small(img):
    import cv2
    return cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (64, 36)).astype(np.float32)


def build(data_dir: Path, images: Path, video: Path, out_dir: Path, model, seconds: float, after: float,
          conf: float) -> None:
    import cv2
    rows = list(csv.DictReader(open(data_dir / "split_manifest.csv", encoding="utf-8")))
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [r for r in rows if not (out_dir / f"{r['stem']}.pkl.gz").exists()]
    print(f"정답 {len(rows)} 장, 남은 것 {len(todo)}", flush=True)
    if not todo:
        return
    todo.sort(key=lambda r: int(r["frame"]))           # 영상 순서대로
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"영상을 열 수 없습니다: {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    t0 = time.perf_counter()
    for n, r in enumerate(todo):
        stem, idx = r["stem"], int(r["frame"])
        ref = cv2.imread(str(images / f"{stem}.jpg"))
        if ref is None:
            print(f"   {stem}: 정답 사진 없음 — 건너뜀", flush=True)
            continue
        g_ref = grey_small(ref)
        start = max(0, idx - int(round(seconds * fps)))
        stop = idx + int(round(after * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        frames, diffs = [], []
        for _ in range(start, stop + 1):
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(fr)
            diffs.append(float(np.abs(grey_small(fr) - g_ref).mean()))
        if not frames or min(diffs) > MATCH_MAX:
            print(f"   {stem}: 정답 장면을 찾지 못함 — 건너뜀", flush=True)
            continue
        best = int(np.argmin(diffs))
        polys_all = []
        for fr in frames[: best + 1]:                    # 채점 시점까지만 추론
            res = model.predict(fr, imgsz=IMGSZ, conf=conf, verbose=False)[0]
            ps = []
            if res.masks is not None:
                for q in res.masks.xyn:
                    q = np.asarray(q, dtype=np.float32)
                    if len(q) >= 3:
                        ps.append(q)
            polys_all.append(ps)
        rec = {"stem": stem, "frame": idx, "fps": fps, "start_frame": start, "target": best,
               "match": diffs[best], "polys": polys_all}
        with gzip.open(out_dir / f"{stem}.pkl.gz", "wb") as f:
            pickle.dump(rec, f, protocol=pickle.HIGHEST_PROTOCOL)
        if (n + 1) % 10 == 0 or n == 0:
            print(f"   {n + 1}/{len(todo)} · {time.perf_counter() - t0:.0f}s · 닮음 {diffs[best]:.1f}", flush=True)
    cap.release()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, required=True, help="정답 데이터 폴더 (split_manifest.csv)")
    ap.add_argument("--images", type=Path, required=True, help="정답 프레임 사진 폴더 (<stem>.jpg)")
    ap.add_argument("--video", type=Path, required=True, help="원본 영상")
    ap.add_argument("--weights", type=Path, required=True, help="학습한 모델 (best.pt)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--after", type=float, default=1.0)
    ap.add_argument("--conf", type=float, default=0.25)
    a = ap.parse_args(argv)
    from ultralytics import YOLO
    model = YOLO(str(a.weights), task="segment")
    build(a.data, a.images, a.video, a.out, model, a.seconds, a.after, a.conf)
    return 0


if __name__ == "__main__":
    sys.exit(main())
