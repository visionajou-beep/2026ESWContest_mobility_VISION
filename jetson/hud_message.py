"""추적기 출력 -> 젯슨에서 Pi 로 보내는 HUD 메시지.

규격은 `jetson/protocol.py` 다(UDP 5005, 프레임당 JSON 하나). 선마다 추론 화면(960x540) 픽셀 좌표의
2 차식 x = a*y^2 + b*y + c 와 state · conf · seq · ts · fps. 연결 끊김 · 순서는 Pi 쪽(hud 패키지)이
처리하므로 여기서는 추적기가 아는 것을 그 메시지로 바꾸기만 한다.

    hud = HudSender("192.168.50.20")
    hud.send(tracker.update(polys), fps=cur_fps)

    state   out.state                   0 정상 / 1 유지 / 2 없음 (양쪽 같은 뜻)
    conf    1.0 두 선 모두 이번 프레임에 보고 보냄 · 0.7 한쪽이 유지 중이거나 한 줄만 보냄 · 한쪽이 역산 · 0.0 State 2 또는 선 없음
    left    out.ego_left 의 안정화 선을 맞춘 [a, b, c] — 화면이 그리지 않는 선이면 None(규격상 null 허용)
    right   out.ego_right 의 [a, b, c]
    fps     넘겨받은 값(추론 + 후처리 속도)

Pi 가 무시해도 되는 추가 키(규격 버전은 1 그대로):

    id_left / id_right   선 번호 — 같은 도색은 같은 번호, 바뀌면 '새 선'
    event                "" | "change_left" | "change_right"   이번 프레임에 차로변경(선을 넘음)
    intent               "" | "left" | "right"                 그쪽으로 가는 중
    extra                차로변경 의도일 때 셋째 선의 [a, b, c], 아니면 None
    y_range              [위, 아래] 픽셀 — 2 차식이 맞는 범위(밖으로 늘이지 말 것)
    inferred             "left" | "right" — 그 쪽 선은 본 것이 아니라 반대쪽에서 역산. 역산했을 때만 붙는다

한 메시지의 선들은 2 차항(휨)을 공유한다(fit_shared_curvature) — 선마다 따로 맞추면 짧은 선과 긴 선이 다르게 휘어
소실점 근처에서 X 자로 교차한다. 그리고 y_range 의 위끝은 두 선이 만나는 행 아래로 내린다.
화면과 메시지는 같은 규칙(TrackerOutput.display_line)으로 그릴 선을 고른다.
"""

from __future__ import annotations

import socket

import numpy as np

from .protocol import PI_IP, UDP_PORT, build_message, encode

DEFAULT_IMG_W, DEFAULT_IMG_H = 960, 540      # 배포 모델 입력(960x544 의 실제 화면 부분)
STATE_LOST = 2


def fit_quadratic(line: np.ndarray, img_w: int, img_h: int) -> list[float] | None:
    """정규화 중심선 (N,2) -> 픽셀 [a, b, c] (x = a*y^2 + b*y + c)."""
    if line is None or len(line) < 2:
        return None
    xs = np.asarray(line[:, 0], dtype=float) * img_w
    ys = np.asarray(line[:, 1], dtype=float) * img_h
    if float(ys.max() - ys.min()) < 1e-6:
        return [0.0, 0.0, float(xs.mean())]
    if len(line) < 3:
        b, c = np.polyfit(ys, xs, 1)
        return [0.0, float(b), float(c)]
    a, b, c = np.polyfit(ys, xs, 2)
    return [float(a), float(b), float(c)]


def fit_shared_curvature(lines: list[np.ndarray], img_w: int, img_h: int) -> list[list[float]]:
    """선마다 2 차식 하나, 모두 같은 a: x_i = a*y^2 + b_i*y + c_i (픽셀).

    평평한 도로에서 커브가 화면에 만드는 휨은 모든 선에 같다(옆 위치에 따라 바뀌는 것은 위치와 기울기). 계산은
    u = y / img_h 로 한다(조건수). 시뮬 녹화 4 개 기준 휨 불일치 27 % -> 0 %, 맞춤 오차 중앙 1.4 -> 1.9 px.
    """
    k = len(lines)
    rows, rhs = [], []
    for j, ln in enumerate(lines):
        u = np.asarray(ln[:, 1], dtype=float)
        x = np.asarray(ln[:, 0], dtype=float) * img_w
        m = np.zeros((len(u), 1 + 2 * k))
        m[:, 0] = u * u
        m[:, 1 + 2 * j] = u
        m[:, 2 + 2 * j] = 1.0
        rows.append(m)
        rhs.append(x)
    sol, *_ = np.linalg.lstsq(np.vstack(rows), np.concatenate(rhs), rcond=None)
    a = float(sol[0]) / img_h ** 2
    return [[a, float(sol[1 + 2 * j]) / img_h, float(sol[2 + 2 * j])] for j in range(k)]


def clip_top_above_crossings(fits: list[list[float]], y_top: float, y_bottom: float,
                             margin: float = 6.0) -> float:
    """a 를 공유하면 이웃한 두 선의 차이는 직선이라 많아야 한 번(소실점 근처) 만난다. 그 행이 범위 안이면 위끝을 그
    바로 아래로 내린다 — HUD 가 X 를 그리지 않는다."""
    order = sorted(fits, key=lambda f: f[0] * y_bottom ** 2 + f[1] * y_bottom + f[2])
    top = y_top
    for l, r in zip(order, order[1:]):
        db, dc = r[1] - l[1], r[2] - l[2]
        if abs(r[0] - l[0]) > 1e-12 or abs(db) < 1e-9:
            continue
        yc = -dc / db
        if top - 1.0 <= yc <= y_bottom:
            top = max(top, min(y_bottom - 5.0, yc + margin))
    return top


def lane_message_fields(out, img_w: int = DEFAULT_IMG_W, img_h: int = DEFAULT_IMG_H,
                        conf: float | None = None) -> dict:
    """TrackerOutput 에서 build_message 에 넣을 값과 추가 키."""
    lost = out.state == STATE_LOST
    shown = {k: out.display_line(b) for k, b in
             (("left", out.ego_left), ("right", out.ego_right), ("extra", out.extra))}
    lines = {k: v for k, v in shown.items() if v is not None and len(v) >= 3}
    fits: dict[str, list[float] | None] = {"left": None, "right": None, "extra": None}
    if len(lines) >= 2:
        for k, f in zip(lines, fit_shared_curvature(list(lines.values()), img_w, img_h)):
            fits[k] = f
    else:
        for k, v in lines.items():
            fits[k] = fit_quadratic(v, img_w, img_h)
    left, right, extra = fits["left"], fits["right"], fits["extra"]
    if conf is None:
        if lost or (left is None and right is None):
            conf = 0.0
        elif (left is None or right is None
              or (out.ego_left is not None and out.ego_left.held)
              or (out.ego_right is not None and out.ego_right.held)
              or getattr(out, "inferred", "")):          # 역산한 선은 본 선이 아니다
            conf = 0.7
        else:
            conf = 1.0
    ego = [l for l in (shown["left"], shown["right"]) if l is not None]
    ys = [float(l[:, 1].min()) for l in ego]
    ye = [float(l[:, 1].max()) for l in ego]
    y_range = None if not ys else [round(min(ys) * img_h, 1), round(max(ye) * img_h, 1)]
    if y_range is not None:
        drawn = [f for f in (left, right, extra) if f is not None]
        if len(drawn) >= 2:
            y_range[0] = round(clip_top_above_crossings(drawn, y_range[0], y_range[1]), 1)
    out_fields = {
        "state": int(out.state), "conf": float(conf), "left": left, "right": right,
        "img_w": img_w, "img_h": img_h,
        "extras": {
            "id_left": None if out.ego_left is None or lost else int(out.ego_left.track_id),
            "id_right": None if out.ego_right is None or lost else int(out.ego_right.track_id),
            "event": out.event, "intent": out.intent, "extra": extra, "y_range": y_range,
        },
    }
    if getattr(out, "inferred", ""):
        # 역산한 쪽 — 그럴 때만 키를 단다. Pi 는 몰라도 된다:
        # 역산은 State 1 에서만 나오므로 HUD 는 이미 주의 색(노랑)으로 그린다. 점선으로 구분하려면 이 키를 읽는다
        out_fields["extras"]["inferred"] = out.inferred
    return out_fields


def build(out, seq: int, img_w: int = DEFAULT_IMG_W, img_h: int = DEFAULT_IMG_H,
          fps: float = 0.0, conf: float | None = None) -> dict:
    f = lane_message_fields(out, img_w, img_h, conf)
    msg = build_message(seq, f["state"], f["conf"], f["left"], f["right"], f["img_w"], f["img_h"], fps)
    ex = f["extras"]
    ex["extra"] = None if ex["extra"] is None else [round(float(v), 6) for v in ex["extra"]]
    msg.update(ex)
    return msg


class HudSender:
    """프레임마다 HUD 메시지를 UDP 로 보낸다."""

    def __init__(self, pi_ip: str = PI_IP, port: int = UDP_PORT,
                 img_w: int = DEFAULT_IMG_W, img_h: int = DEFAULT_IMG_H) -> None:
        self.addr = (pi_ip, port)
        self.img_w, self.img_h = img_w, img_h
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.seq = 0
        self.err_count = 0

    def send(self, out, fps: float = 0.0, conf: float | None = None) -> dict:
        msg = build(out, self.seq, self.img_w, self.img_h, fps, conf)
        try:
            self.sock.sendto(encode(msg), self.addr)
        except OSError as e:
            self.err_count += 1
            if self.err_count % 30 == 1:
                print(f"[hud] send failed ({self.err_count}): {e}")
        self.seq += 1
        return msg

    def close(self) -> None:
        self.sock.close()
