# 악천후 차선 인식 · AR-HUD 투영 — 팀 VISION

제24회 임베디드 SW 경진대회 자동차 / 모빌리티 부문 출품작 **「딥러닝 기반 악천후 차선 추론 및 AR-HUD 투영 주행 보조 장치」** 의 전체 코드다.
카메라 영상에서 차선을 찾고(딥러닝 분할 모델), 그중 내 차로의 좌 · 우 선을 골라 시간에 따라 안정되게 따라간 뒤(후처리),
좌표만 UDP 로 Raspberry Pi 에 보내면 Pi 가 운전자 시점으로 정렬해 하프미러 HUD 에 그린다.
비 · 야간에 한쪽 선이 안 보이면 반대쪽 선을 역산하고, 확신이 없으면 그리지 않는다.

```
카메라 영상 ─▶ 차선 분할 모델 (YOLO11s-seg, 960x544) ─▶ 후처리 (자차 좌 · 우 선 + State) ─▶ UDP/JSON ─▶ Raspberry Pi 5 ─▶ 하프미러 HUD
               Jetson Orin Nano (TensorRT FP16, 34 FPS)          postprocess/                    jetson_live.py   hud-pi/
```

## 구성

```
train/          차선 분할 모델 학습 (개발 PC)
  bdd_label.py      BDD100K 차선 라벨 읽기 — 도색 하나를 폴리곤 하나로
  make_dataset.py   YOLO 분할 학습용 데이터셋 만들기 (1 클래스 lane_boundary)
  train.py          학습 실행
  config.yaml       학습 설정 (배포 모델을 만든 값 그대로)
postprocess/    자차 차선 후처리 (numpy 만 필요, 젯슨에서 그대로 동작)
  tracker.py        추적기 본체 — 자차 좌 · 우 선 고르기 · 추적 · 차로변경 · State 판정
  smooth.py         선 다듬기 — 조각을 모아 중심선 하나로, 행마다 평활
  geometry.py       소실점 · 기준행 · 좌우 판정
  missing_lane.py   한쪽 선이 안 보일 때 — 한 줄 유지 · 반대쪽 역산
  double_line.py    이중선 — 안쪽 선 쓰기
  lane.py           자료형 · State 0/1/2
  config.py         상수
  hud_message.py    추적기 결과 → HUD 메시지 (좌 · 우 2 차식, 휨 공유, y_range) · UDP 송신
  protocol.py       Jetson ↔ Pi 메시지 규격 (UDP 5005, JSON)
jetson_live.py  Jetson 실행 — 캡처 → 추론 → 후처리 → HUD 송신 · 브라우저 확인 화면 · 녹화
hud-pi/         Raspberry Pi HUD (OpenCV · numpy)
  hud_system.py     UDP 수신 · 패킷 검증 · 젯슨 2 차식 → 폴리라인 변환 · 스무딩 · 송신 어댑터 · 모의 송신
  hud_ui.py         실행 — 수신 + 표시, 미리보기, 샘플 PNG, 진단 패턴, 렌더 속도
  hud_state.py      State 판정 — 무수신 · 페이드 · 디바운싱 · State 1 상한
  hud_theme.py      AR 오버레이 렌더러 — 경계선 2 개 · State 색 · 상태 바 (가산 합성, 검정 배경)
  hud_align.py      운전자 시점 정렬 — 카메라 좌표 → 패널 픽셀 3×3 행렬 · 보정 도구 (place · pick · aim · verify · eyebox · profile)
  hud_trim.py       실주행 차선 위에서 키보드로 미세 보정
  hud_config.json   설정 — display · network · state · alignment · theme
  docs/             사용법 (시점 정렬 · 화면 구성 · 시안 구현 · 핸드오프)
eval/           평가 (정답 영상 채점)
  predict.py        정답 프레임마다 앞 6 초를 모델로 추론해 저장
  ground_truth.py   정답 라벨 읽기 · 정답 쪽 자차 선 고르기
  score.py          후처리를 돌려 그린 선과 정답 선 짝짓기
  line_error.py     선 오차 계산 · 결과 표
tests/          후처리 테스트 81 (실제 모델 출력 고정 정답 포함)
```

## 처리 흐름

- **모델**은 화면의 진행 방향 차선 도색을 모두 픽셀 단위로 찾는다. 어느 선이 내 차로의 선인지는 모델이 아니라 후처리가 정한다.
- **후처리**는 소실점과 기준행(소실점에서 화면 아래까지의 35.4 % 지점)에서 진행 경로 양쪽의 가장 가까운 선을 자차 선으로 고르고,
  선마다 번호를 붙여 프레임을 넘어 따라간다. 시간은 프레임 수가 아니라 찍힌 시각(초)으로 센다.
- **Jetson** 은 영상이 아니라 좌표만 보낸다 — 프레임당 JSON 한 개(약 260 B), 좌 · 우 선은 960×540 화면 좌표의 2 차식 `x = a·y² + b·y + c` (휨 a 공유).
- **Pi** 는 쌓인 패킷 중 가장 최근 것만 쓰고, 0.35 초 동안 수신이 없으면 State 2 로 올린다. 차선은 정렬 행렬로 투영해 가산 합성으로 그린다.

| State | 뜻 | Jetson 이 보내는 것 | HUD |
| --- | --- | --- | --- |
| 0 | 정상 | 자차 좌 · 우 선 | 초록 두 선 |
| 1 | 주의 | 잠깐 안 보인 선은 0.15 초 유지. 한쪽을 잃으면 짝으로 확인된 한 줄 + 역산한 반대쪽 선 (최대 1.2 초) | 노란 두 선 |
| 2 | 확신 없음 | 선 없음 — "틀리게 그리느니 안 그린다" | 소거 (빨간 상태 바) |

## 설치

```bash
pip install -r requirements.txt
```

- 학습 · 평가: PyTorch 는 GPU 에 맞는 CUDA 판을 따로 설치한다.
- Jetson: 저장소 최상위에서 `jetson_live.py` 를 실행한다. JetPack 의 TensorRT · OpenCV 를 쓰고 `ultralytics` 만 추가한다. 엔진은 `best.pt` 를 ONNX(opset 17) → TensorRT FP16(입력 960×544)으로 변환해 만든다.
- Raspberry Pi: `sudo apt install python3-opencv python3-numpy` (pip 로 OpenCV 를 설치하지 않는다). 후처리와 HUD 는 numpy · OpenCV 만 있으면 된다.

## 사용법

**학습** (BDD100K 공식 100k 이미지 · 차선 라벨이 필요하다)

```bash
python train/make_dataset.py --images-root <bdd100k_images_100k/100k> --labels-root <bdd100k_labels/100k>
python train/train.py --config train/config.yaml
```

**Jetson 실행** (카메라 또는 캡처보드 → 추론 → 후처리 → Pi 로 송신, 브라우저 `http://<젯슨IP>:8080` 에서 확인 · 녹화)

```bash
python3 jetson_live.py --model lane_v1_960x544_fp16.engine --device 0 --fps 30 --web 8080 --hud-host 192.168.50.20
python3 jetson_live.py --model best.pt --device 영상.mp4 --save --out result.mp4      # 영상 파일로 돌려 보기 (PC)
```

카메라는 차량 중앙(룸미러 쪽)에 고정한 뒤에 실행한다 — 소실점을 주행 중 스스로 배우므로(10 초 창), 켠 뒤 카메라를 움직이면 한동안 선을 못 고른다.

**Raspberry Pi HUD** (`hud-pi/` 폴더에서)

```bash
python3 hud_ui.py receive --theme           # 수신 + AR 오버레이 표시 (실제 운용)
python3 hud_ui.py receive --theme --stats   # 2 초마다 fps · 지연 · 버린 패킷
python3 hud_ui.py preview --theme           # 네트워크 없이 화면 구성만
python3 hud_ui.py bench --theme             # 이 장치의 렌더 시간 (1920×1200 에서 약 22 ms)
python3 hud_system.py mock                  # 젯슨 없이 가짜 차선 송신
```

운전자 시점 정렬(한 번) — 모니터를 먼저 단단히 고정하고, 노면 위 같은 지점을 카메라 영상과 HUD 화면에서 찍어 행렬을 구한다.

```bash
python3 hud_align.py place     # HUD 가 찍은 자리에 꼬깔 놓기
python3 hud_align.py pick --frame road.png
python3 hud_align.py aim       # 운전석에서 십자 맞추기
python3 hud_align.py verify    # 확인 · 통째로 밀기 · 저장
python3 hud_align.py trim      # 실주행 차선 위에서 키보드로 미세 보정
```

운전자가 바뀌면 `eyebox`(같은 눈 위치에 앉히기) · `profile --name <이름>`(두 사람 보정을 저장해 사이를 섞기)으로 재보정 없이 맞춘다.

**후처리만 쓰기** — 모델 출력(정규화 폴리곤)을 프레임마다 넣는다.

```python
from ultralytics import YOLO
from postprocess import EgoLaneTracker

model = YOLO("best.pt")
tracker = EgoLaneTracker(30.0)
for frame, t in frames:                              # 영상 프레임과 찍힌 시각(초)
    res = model.predict(frame, imgsz=(544, 960), conf=0.25, verbose=False)[0]
    polys = [] if res.masks is None else res.masks.xyn
    out = tracker.update(polys, t=t)
    left = out.display_line(out.ego_left)            # 그릴 선 (N, 2) 정규화 좌표, 없으면 None
    right = out.display_line(out.ego_right)
    print(out.state, out.inferred)                   # State 0/1/2, 역산한 쪽 ("left" / "right" / "")
```

**테스트**

```bash
python -m pytest tests -q
```

**평가** — 정답 데이터 폴더(목록 · 라벨 · 카메라 보정값)와 원본 영상이 필요하다 (저장소에 포함하지 않음).

```bash
python eval/predict.py --data <정답 폴더> --images <정답 프레임 사진> --video <원본 영상> --weights best.pt --out cache/<이름>
python eval/line_error.py --set <이름>=<정답 폴더> --cache cache --out results
```

## 평가 방법과 결과

**선 오차**(주 지표): 좌 · 우 선마다, 정답 중심선과 화면에 그린 선이 둘 다 지나는 높이 전부에서
'가로 거리 ÷ 그 높이의 정답 차로폭'을 평균한다. 차로폭은 정답에서만 잰다(실제 차로폭을 가정하지 않음).
5 % 안이면 정확, 10 % 를 넘으면 틀린 선이고, 그리지 않은 것은 미표시로 따로 센다.

면적 겹침(IoU)은 폭이 좁은 차선에 지나치게 엄격하다 — 몇 픽셀만 어긋나도 크게 떨어지지만 HUD 에서는 같은 자리다.
그래서 HUD 가 실제로 보여 주는 '선의 위치'를 직접 잰다. mIoU 도 표준 지표로 함께 보고한다.

**한국 고속도로 정답 영상** (학습에 쓰지 않은 공개 주행 영상 300 장, 팀이 직접 라벨, 영상마다 뒤 절반으로 보고)

| 조건 | 정답 있는 쪽 | 정확 | 어긋남 | 틀린 선 | 미표시 |
| --- | --- | --- | --- | --- | --- |
| 주간 맑음 | 112 | 100 % | 0 % | 0 % | 0 % |
| 야간 폭우 | 153 | 89.5 % | 2.0 % | 0.7 % | 7.8 % |
| **합계** | 265 | **94.0 %** | 1.1 % | **0.4 %** | 4.5 % |

- 옆 차로 쪽으로 넘어간 틀린 선은 0 건이다. 선 오차 중앙은 차로폭의 0.6 % (3.5 m 차로라면 약 2 cm).
- 한쪽을 일부러 가린 실험에서 역산 선의 95 % 가 차로폭 5 % 안에 그려졌다.
- 정답 쪽 자차 선은 규칙으로 고르며, 차가 선을 밟고 있거나 정답 선이 짧아 판별이 애매한 쪽은 뺀다 (`eval/ground_truth.py`).

| 그 밖의 측정 | 값 |
| --- | --- |
| mIoU — BDD100K test / 한국 야간 폭우 고속도로 | 0.771 / 0.670 (목표 0.85 미달) |
| 추론 속도 — Jetson Orin Nano, TensorRT FP16, 960x544 | 34.32 FPS (29.1 ms) |
| 라이브 — 캡처 → 추론 → 후처리 → 송신 | 28.4 fps · 캡처 → HUD 전송 중앙 52.5 ms |
| Pi HUD 렌더 — 1920×1200, AR 오버레이 | 21.9 ms / 프레임 |

## 데이터

- **학습**: BDD100K (Berkeley DeepDrive) 의 차선 라벨. 진행 방향 흰색 · 노란색 차선만 1 클래스로 쓴다. 데이터는 BDD100K 이용 조건에 따라 따로 받아야 한다.
- **평가**: 한국 고속도로 블랙박스 공개 영상(사용 허락을 받은 영상)에서 뽑은 프레임에 팀이 직접 그린 정답. 영상 · 정답은 저장소에 넣지 않았다.
- 학습한 모델 가중치(`best.pt`)와 TensorRT 엔진은 용량 때문에 저장소에 넣지 않았다.

## 라이선스

이 저장소는 **GNU AGPL-3.0** 을 따른다 (`LICENSE`). 학습 · 추론에 쓰는 Ultralytics YOLO 가 AGPL-3.0 이기 때문이다.

| 사용한 오픈소스 · 데이터 | 라이선스 |
| --- | --- |
| Ultralytics YOLO (YOLO11) | AGPL-3.0 |
| PyTorch | BSD-3-Clause |
| NumPy | BSD-3-Clause |
| OpenCV | Apache-2.0 |
| Pillow | MIT-CMU (HPND) |
| PyYAML | MIT |
| pytest | MIT |
| BDD100K 데이터셋 | BDD100K 이용 약관 (연구 · 비상업 목적) |
