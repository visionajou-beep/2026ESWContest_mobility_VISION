"""차선 분할 모델 학습 — YOLO11s-seg, 입력 960, BDD100K 8 만 장 (1 클래스 lane_boundary).

    # 1. BDD100K 차선 라벨 -> YOLO 분할 데이터셋 (공식 100k 이미지 · 라벨 폴더가 필요)
    python train/make_dataset.py --images-root <bdd100k_images_100k/100k> --labels-root <bdd100k_labels/100k>

    # 2. 학습 (배포 모델을 만든 설정 그대로)
    python train/train.py --config train/config.yaml

    # 끊긴 학습 이어서 (optimizer · scheduler · epoch 상태까지)
    python train/train.py --resume train/runs/segment/desktop_full_960/weights/last.pt

    # 넘길 인자만 보기
    python train/train.py --config train/config.yaml --dry-run

설정 파일의 값을 그대로 `YOLO(model).train(**설정)` 에 넘긴다. `data` · `project` 의 상대경로는 설정 파일 폴더 기준이다.
배포 모델은 RTX 4070 SUPER 12 GB · PyTorch 2.13 + CUDA 13.0 · Ultralytics 8.4.112 에서 학습했다 (100 epoch, 가장 좋은 epoch 81).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


def load_train_args(config: Path) -> dict:
    """설정 파일 -> train() 인자. 상대경로(data · project)는 설정 파일 폴더 기준으로 바꾼다."""
    with config.open(encoding="utf-8") as f:
        args = yaml.safe_load(f)
    if not isinstance(args, dict):
        raise SystemExit(f"[에러] 설정은 키: 값 모음이어야 합니다: {config}")
    base = config.resolve().parent
    for key in ("data", "project"):
        if key in args and not Path(str(args[key])).is_absolute():
            args[key] = str((base / str(args[key])).resolve())
    return args


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="YOLO11s-seg 차선 분할 학습")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--config", type=Path, help="새 학습 설정 (예: train/config.yaml)")
    src.add_argument("--resume", type=Path, metavar="LAST.PT", help="끊긴 학습의 last.pt 에서 이어서")
    ap.add_argument("--dry-run", action="store_true", help="넘길 인자만 출력하고 학습은 안 한다")
    a = ap.parse_args(argv)

    if a.resume is not None:
        if a.resume.name.lower() != "last.pt" or not a.resume.is_file():
            raise SystemExit("[에러] 이어서 하려면 학습 폴더의 weights/last.pt 가 필요합니다")
        print(json.dumps({"resume": str(a.resume)}, ensure_ascii=False))
        if a.dry_run:
            return 0
        from ultralytics import YOLO
        YOLO(str(a.resume)).train(resume=True)
        return 0

    args = load_train_args(a.config)
    print(json.dumps(args, ensure_ascii=False, indent=2))
    if a.dry_run:
        return 0
    if not Path(args["data"]).is_file():
        raise SystemExit(f"[에러] 데이터셋이 없습니다: {args['data']} — 먼저 make_dataset.py 로 만든다")
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("[에러] CUDA 를 쓸 수 없습니다 — NVIDIA 드라이버와 CUDA 판 PyTorch 를 확인")
    from ultralytics import YOLO
    model = YOLO(str(args.pop("model")))
    model.train(**args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
