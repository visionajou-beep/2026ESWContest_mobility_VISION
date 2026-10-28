"""HUD(Raspberry Pi) — 패킷 해석 · State 판정 · 렌더 (실제 젯슨 패킷 3 개)."""
import json
import time

import numpy as np
import pytest

from hud import config as C
from hud import packet as P
from hud.render import ThemeRenderer
from hud.sender import _mock_lane
from hud.state import StateTracker

# 젯슨이 실제로 보낸 패킷 (시뮬 급커브 녹화) — 정상 두 선 · 한 줄 + 역산 없음 · 차로변경 의도(셋째 선)
PACKETS = {
    "s0": '{"v": 1, "seq": 201, "ts": 1793032509.902, "img_w": 960, "img_h": 540, "state": 0, "conf": 1.0, '
          '"left": [-0.000567, -0.441043, 594.907009], "right": [-0.000567, 1.913082, 8.493801], "fps": 26.4, '
          '"id_left": 6, "id_right": 2, "event": "", "intent": "", "extra": null, "y_range": [280.8, 534.6]}',
    "s1_one": '{"v": 1, "seq": 209, "ts": 1793032509.921, "img_w": 960, "img_h": 540, "state": 1, "conf": 0.7, '
              '"left": [-0.000997, -0.097339, 519.717413], "right": null, "fps": 26.1, "id_left": 6, "id_right": 2, '
              '"event": "", "intent": "", "extra": null, "y_range": [270.0, 534.6]}',
    "intent": '{"v": 1, "seq": 80, "ts": 1793032509.588, "img_w": 960, "img_h": 540, "state": 0, "conf": 1.0, '
              '"left": [-0.000397, 0.426684, 313.39978], "right": [-0.000397, 3.152533, -356.07989], "fps": 28.4, '
              '"id_left": 2, "id_right": 3, "event": "", "intent": "left", "extra": [-0.000397, -2.062088, 936.786514], '
              '"y_range": [264.6, 502.2]}',
}


@pytest.fixture
def cfg():
    c = C.load_config(C.DEFAULT_CONFIG_PATH)
    C.ensure_ui_config(c)
    return c


def test_decode_jetson_packet_two_lanes():
    pkt = P._decode_packet(PACKETS["s0"].encode())
    assert pkt["seq"] == 201 and pkt["status"] == "ok" and pkt["state"] == 0
    assert len(pkt["lanes"]) == 2
    for lane in pkt["lanes"]:                       # 2 차식이 정규화 폴리라인으로 펴진다
        assert len(lane) == P.JETSON_LANE_SAMPLES
        xs = [p[0] for p in lane]
        ys = [p[1] for p in lane]
        assert all(0.0 <= v <= 1.0 for v in xs + ys)
    assert pkt["lane_ids"] == [6, 2]


def test_decode_one_line_state1():
    pkt = P._decode_packet(PACKETS["s1_one"].encode())
    assert pkt["state"] == 1 and pkt["status"] == "degraded"
    assert len(pkt["lanes"]) == 1                   # right 가 null 이면 그리지 않는다


def test_decode_rejects_bad_packet():
    with pytest.raises(P.PacketError):
        P._decode_packet(b'{"v": 1}')
    with pytest.raises(P.PacketError):
        P._decode_packet(b"not json")


def test_state_no_packet_is_lost(cfg):
    t = StateTracker(cfg)
    s = t.update(10.0, raw_state=0, confidence=1.0, last_received=None)
    assert s.state == 2 and s.link_lost


def test_state_timeout_then_recover(cfg):
    t = StateTracker(cfg)
    now = 10.0
    for _ in range(8):                               # 정상 패킷이 계속 오면 0
        s = t.update(now, raw_state=0, confidence=1.0, last_received=now - 0.03)
        now += 0.033
    assert s.state == 0 and not s.link_lost
    s = t.update(now + 1.0, raw_state=0, confidence=1.0, last_received=now - 0.03)   # 1 초 무수신 → 즉시 2
    assert s.state == 2 and s.link_lost


def test_state_caution_has_timeout(cfg):
    t = StateTracker(cfg)
    now = 10.0
    for _ in range(8):
        s = t.update(now, raw_state=1, confidence=1.0, last_received=now)
        now += 0.033
    assert s.state == 1
    now += 2.0                                       # caution_timeout_seconds(1.2) 를 넘기면 젯슨이 1 을 보내도 2
    s = t.update(now, raw_state=1, confidence=1.0, last_received=now)
    assert s.state == 2


def test_render_black_background_and_lanes(cfg):
    r = ThemeRenderer(cfg)
    h, w = int(cfg["display"]["height"]), int(cfg["display"]["width"])
    empty = r.render(lanes=[], state=2, elapsed=5.0)
    assert empty.shape == (h, w, 3) and empty.dtype == np.uint8
    assert (empty == 0).mean() > 0.99                # 반사 광학계: 배경은 순수 검정
    lanes = [_mock_lane(0.18, 0.43, 0.0), _mock_lane(0.82, 0.57, 0.3)]
    drawn = r.render(lanes=lanes, state=0, elapsed=5.0)
    lit = drawn.max(axis=2) > 40
    assert lit.sum() > 2000                          # 선 두 개가 그려졌다
    g = drawn[..., 1][lit].mean()
    assert g > drawn[..., 2][lit].mean()             # state 0 은 초록 (BGR)
    caution = r.render(lanes=lanes, state=1, elapsed=5.0)
    lit_c = caution.max(axis=2) > 40
    assert caution[..., 2][lit_c].mean() > drawn[..., 2][lit].mean()   # state 1 은 노랑 — 빨강 성분이 더 크다


def test_render_is_deterministic(cfg):
    r = ThemeRenderer(cfg)
    lanes = [_mock_lane(0.18, 0.43, 0.0), _mock_lane(0.82, 0.57, 0.3)]
    a = r.render(lanes=lanes, state=0, elapsed=5.0)
    b = r.render(lanes=lanes, state=0, elapsed=5.0)
    assert np.array_equal(a, b)
