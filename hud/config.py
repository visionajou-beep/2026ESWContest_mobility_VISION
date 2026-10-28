# -*- coding: utf-8 -*-
"""HUD 설정 파일(hud_config.json) 읽기 · 쓰기와 기본값.

설정은 display(패널) · network(수신) · state(판정) · alignment(정렬) · ui · theme 항목으로 나뉜다.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "hud_config.json"


DEFAULT_CONFIG: dict[str, Any] = {
    "network": {
        "bind_host": "0.0.0.0",
        "port": 5005,
        "packet_timeout_seconds": 0.35,
        # 커널 수신 버퍼 요청값(바이트). 리눅스는 이 값의 두 배를 잡는다.
        "recv_buffer_bytes": 16_384,
    },
    "state": {
        "lane_fade_seconds": 0.4,
        "caution_timeout_seconds": 1.2,
        "rise_frames": 3,
        "fall_frames": 5,
    },
    "display": {
        "width": 1920,
        "height": 1200,
        "fullscreen": True,
        "window_name": "Rain Lane HUD",
        "line_thickness": 8,
        "line_color_bgr": [255, 255, 255],
        "flip_horizontal": False,
        "flip_vertical": False,
        "show_status": False,
        "smoothing_alpha": 0.55,
        "association_distance": 0.18,
        # 한 프레임에 허용할 차선 x 이동량(정규화). 실측에서 같은 트랙의
        # 프레임 간 이동은 p99 가 0.018 이라 0.05 면 정상 주행을 막지 않는다.
        "jump_gate_distance": 0.05,
        # 이만큼 연속으로 임계값을 넘으면 실제 이동으로 보고 받아들인다.
        "jump_gate_frames": 3,
    },
    "destination_quad_normalized": [
        [0.08, 0.08],
        [0.92, 0.08],
        [0.98, 0.98],
        [0.02, 0.98],
    ],
}


def write_default_config(path: Path, *, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        print(f"config already exists: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(DEFAULT_CONFIG, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"config created: {path}")


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        write_default_config(path)
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def save_config(path: Path, config: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


DEFAULT_UI: dict[str, Any] = {
    "safe_area": [0.06, 0.06, 0.94, 0.94],
    "lane": {
        "color_bgr": [255, 255, 255],
        "thickness": 8,
        "emphasis_thickness": 13,
    },
    "warning": {
        "enabled": True,
        "color_bgr": [70, 190, 255],
        "chevron_count": 3,
        "chevron_width": 0.045,
        "chevron_height": 0.070,
        "chevron_gap": 0.028,
        "thickness": 7,
        "side_margin": 0.085,
        "center_y": 0.52,
        "blink_hz": 2.4,
        "flow_hz": 1.6,
        "label": True,
        "label_scale": 0.85,
        "label_y": 0.22,
    },
    "debug": {
        "enabled": False,
        "color_bgr": [140, 140, 140],
        "scale": 0.5,
        "thickness": 1,
        "origin": [0.03, 0.06],
        "line_gap": 0.045,
    },
    "center_tick": {
        "enabled": False,
        "color_bgr": [120, 120, 120],
        "width": 0.05,
        "y": 0.95,
        "thickness": 4,
    },
}


def ensure_ui_config(config: dict[str, Any]) -> dict[str, Any]:
    """설정에 ui 항목이 없으면 기본값을 채워 넣는다."""
    ui = config.setdefault("ui", {})
    for section, defaults in DEFAULT_UI.items():
        if not isinstance(defaults, dict):
            ui.setdefault(section, defaults)
            continue
        target = ui.setdefault(section, {})
        for key, value in defaults.items():
            target.setdefault(key, value)
    return ui
