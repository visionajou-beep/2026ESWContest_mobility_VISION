"""
Jetson <-> Raspberry Pi 5 차선 메시지 프로토콜 정의.
양쪽 기기에 동일한 파일을 복사해서 사용한다.

메시지 구조 (JSON, UTF-8):
{
    "seq":    123,              # 프레임 순번 (패킷 순서 뒤바뀜 감지용)
    "ts":     1725000000.123,   # Jetson 송신 시각 (time.time(), 지연 측정용)
    "img_w":  960,              # 추론 입력 이미지 폭 (호모그래피 기준)
    "img_h":  540,              # 추론 입력 이미지 높이
    "state":  0,                # FSM 상태 0=정상 1=보간 2=경고
    "conf":   0.72,             # ego lane 신뢰도 (0~1)
    "left":   [a, b, c],        # 좌측 차선 2차 다항식 x = a*y^2 + b*y + c (픽셀 좌표)
    "right":  [a, b, c],        # 우측 차선 (검출 실패 시 null)
    "fps":    24.3              # Jetson 추론 FPS (모니터링용)
}
"""

import json
import time

PROTOCOL_VERSION = 1

# 네트워크 설정 --------------------------------------------------------
# 랜선 직결 대역. 핫스팟이 192.168.1.x / 192.168.43.x 를 쓰는 경우가 많아 피한다.
JETSON_IP = "192.168.50.10"
PI_IP = "192.168.50.20"
UDP_PORT = 5005

# Pi 쪽에서 이 시간(초) 동안 패킷이 안 오면 Jetson 끊김으로 판단
RECV_TIMEOUT_SEC = 0.5

# 이보다 오래된 패킷은 버림 (초). 네트워크 지연이 이 값을 넘으면 HUD가 과거를 그리게 됨
MAX_PACKET_AGE_SEC = 0.3


def build_message(seq, state, conf, left, right, img_w, img_h, fps=0.0):
    """송신용 dict 생성. left/right 는 [a, b, c] 리스트 또는 None."""
    return {
        "v": PROTOCOL_VERSION,
        "seq": int(seq),
        "ts": time.time(),
        "img_w": int(img_w),
        "img_h": int(img_h),
        "state": int(state),
        "conf": round(float(conf), 4),
        "left": [round(float(x), 6) for x in left] if left is not None else None,
        "right": [round(float(x), 6) for x in right] if right is not None else None,
        "fps": round(float(fps), 1),
    }


def encode(msg: dict) -> bytes:
    return json.dumps(msg, separators=(",", ":")).encode("utf-8")


def decode(data: bytes) -> dict:
    msg = json.loads(data.decode("utf-8"))
    if msg.get("v") != PROTOCOL_VERSION:
        raise ValueError(f"protocol version mismatch: {msg.get('v')}")
    return msg
