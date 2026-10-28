# -*- coding: utf-8 -*-
"""HUD 실행 — 수신 + 화면 표시, 미리보기, 샘플 PNG, 진단 패턴, 렌더 속도 측정.

    python -m hud.display receive --theme          # 젯슨 패킷 수신 + AR 오버레이 표시 (실제 운용)
    python -m hud.display receive --theme --stats  # 2 초마다 fps · 지연 · 버린 패킷
    python -m hud.display preview --theme          # 네트워크 없이 화면 구성만
    python -m hud.display sample --theme --out png # 상황별 PNG 저장
    python -m hud.display pattern                  # 패널 진단 패턴
    python -m hud.display bench --theme            # 이 장치의 렌더 시간

--theme 없이 실행하면 기본(흰 선) 표시, 설정 ui.style 이 ar_overlay 면 항상 AR 오버레이다.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import cv2

from .config import DEFAULT_CONFIG_PATH, ensure_ui_config, load_config, save_config
from .packet import WARNING_STATES
from .pattern import build_test_pattern
from .receiver import LaneFeed, _LatencyStats, _blank_debug
from .render import LANE_STATES, ThemeRenderer, normalize_state
from .render_basic import HudRenderer
from .sender import _mock_lane


def make_renderer(config: dict[str, Any], theme: bool):
    """시안 테마와 기본 선 표시 중 하나를 고른다."""
    if theme or str(config.get("ui", {}).get("style", "")) == "ar_overlay":
        return ThemeRenderer(config)
    return HudRenderer(config)


def _open_window(name: str, renderer: HudRenderer, windowed: bool, fullscreen: bool) -> None:
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    if fullscreen and not windowed:
        cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.resizeWindow(name, renderer.width, renderer.height)


# ---------------------------------------------------------------------------
# preview: 네트워크도 모델도 없이 화면 구성만 조정하는 모드
# ---------------------------------------------------------------------------


def run_preview(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    ensure_ui_config(config)
    renderer = make_renderer(config, getattr(args, "theme", False))
    renderer.ui["debug"]["enabled"] = True
    if not renderer.calibrated:
        print(
            "warning: no driver-viewpoint alignment saved. lanes will not line up\n"
            "         with the real road. run hud.calibrate pick / aim first."
        )

    name = "HUD preview"
    _open_window(name, renderer, args.windowed, bool(config["display"]["fullscreen"]))
    print(
        "preview keys: W warning  L lanes  D debug  H flip  [ ] thickness  S save  Q quit"
    )

    warning_index = 0
    three_lanes = False
    state = normalize_state(getattr(args, "state", 0))
    announced = False
    started = time.monotonic()

    while True:
        elapsed = time.monotonic() - started
        phase = elapsed * 0.8
        if three_lanes:
            lanes = [
                _mock_lane(0.14, 0.42, phase),
                _mock_lane(0.50, 0.54, phase + 0.2),
                _mock_lane(0.86, 0.68, phase + 0.4),
            ]
        else:
            lanes = [
                _mock_lane(0.18, 0.43, phase),
                _mock_lane(0.82, 0.57, phase + 0.3),
            ]

        warning = WARNING_STATES[warning_index]
        start_render = time.perf_counter()
        canvas = renderer.render(
            lanes=lanes,
            warning=warning,
            elapsed=elapsed,
            state=state,
            telemetry={"confidence": 0.7, "fps": 22.0, "inference_ms": 44.6,
                       "departure_distance": 0.3},
            debug=_blank_debug(
                status="preview",
                lanes=len(lanes),
                seq=int(elapsed * 22),
                fps=22.0,
                inference_ms=44.6,
                warning=warning,
            ),
        )
        render_ms = (time.perf_counter() - start_render) * 1000.0
        if not announced:
            announced = True
            print(f"renderer state {renderer.state}")
        cv2.putText(
            canvas,
            f"render {render_ms:.1f}ms",
            (12, renderer.height - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (90, 90, 90),
            1,
            cv2.LINE_AA,
        )

        cv2.imshow(name, canvas)
        key = cv2.waitKey(16) & 0xFF
        if key in (27, ord("q")):
            break
        if key == ord("w"):
            warning_index = (warning_index + 1) % len(WARNING_STATES)
        elif key == ord("l"):
            three_lanes = not three_lanes
        elif key == ord("d"):
            renderer.ui["debug"]["enabled"] = not renderer.ui["debug"]["enabled"]
        elif key == ord("h"):
            config["display"]["flip_horizontal"] = not config["display"]["flip_horizontal"]
            renderer.rebuild(config)
            renderer.ui["debug"]["enabled"] = True
        elif key == ord("["):
            config["ui"]["lane"]["thickness"] = max(
                2, int(config["ui"]["lane"]["thickness"]) - 1
            )
            renderer.rebuild(config)
            renderer.ui["debug"]["enabled"] = True
        elif key == ord("]"):
            config["ui"]["lane"]["thickness"] = min(
                24, int(config["ui"]["lane"]["thickness"]) + 1
            )
            renderer.rebuild(config)
            renderer.ui["debug"]["enabled"] = True
        elif key == ord("s"):
            save_config(args.config, config)
            print(f"saved: {args.config}")

    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# sample: PNG 로 저장해 노트북에서 확인
# ---------------------------------------------------------------------------


def run_sample(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    ensure_ui_config(config)
    renderer = make_renderer(config, getattr(args, "theme", False))
    renderer.ui["debug"]["enabled"] = True
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    state = normalize_state(getattr(args, "state", 0))

    scenes = {
        "01_normal": ("none", False),
        "02_departure_left": ("departure_left", False),
        "03_change_right": ("change_right", True),
        "04_lost": ("none", None),
    }
    for name, (warning, three) in scenes.items():
        if three is None:
            lanes: list[Any] = []
        elif three:
            lanes = [
                _mock_lane(0.14, 0.42, 0.0),
                _mock_lane(0.50, 0.54, 0.2),
                _mock_lane(0.86, 0.68, 0.4),
            ]
        else:
            lanes = [_mock_lane(0.18, 0.43, 0.0), _mock_lane(0.82, 0.57, 0.3)]
        canvas = renderer.render(
            lanes=lanes,
            warning=warning,
            elapsed=0.0,
            state=state,
            telemetry={"confidence": 0.7, "fps": 22.4, "inference_ms": 44.6,
                       "departure_distance": 0.3},
            debug=_blank_debug(
                status="lost" if three is None else "ok",
                lanes=len(lanes),
                seq=120,
                fps=22.4,
                inference_ms=44.6,
                warning=warning,
            ),
        )
        path = out / f"{name}.png"
        cv2.imwrite(str(path), canvas)
        print(f"wrote {path}  state {renderer.state}")


def run_pattern(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    ensure_ui_config(config)
    # 차선을 실제로 그리는 렌더러여야 디자인 좌표 매핑과 굵기가 맞는다.
    renderer = ThemeRenderer(config)
    grid = not args.no_grid
    canvas = build_test_pattern(renderer, grid=grid)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), canvas)
        print(f"wrote {out}  {renderer.width}x{renderer.height}")
        return

    name = "HUD test pattern"
    _open_window(name, renderer, args.windowed, bool(config["display"]["fullscreen"]))
    print("pattern keys: G grid  Q quit")
    while True:
        cv2.imshow(name, canvas)
        key = cv2.waitKey(50) & 0xFF
        if key in (27, ord("q")):
            break
        if key == ord("g"):
            grid = not grid
            canvas = build_test_pattern(renderer, grid=grid)
    cv2.destroyAllWindows()


def run_receive(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    ensure_ui_config(config)
    renderer = make_renderer(config, getattr(args, "theme", False))
    display = config["display"]
    network = config["network"]

    bind_host = args.bind_host or str(network["bind_host"])
    port = args.port or int(network["port"])
    feed = LaneFeed(config, bind_host, port)

    name = str(display["window_name"])
    _open_window(name, renderer, args.windowed, bool(display["fullscreen"]))
    print(f"HUD listening on {bind_host}:{port} / {renderer.width}x{renderer.height}")

    started = time.monotonic()
    stats = _LatencyStats(feed) if args.stats else None

    try:
        while True:
            feed.poll(0.01)
            now = time.monotonic()
            frame_args = feed.frame(now)
            if renderer.ui["debug"]["enabled"]:
                frame_args["debug"]["show"] = True
            canvas = renderer.render(elapsed=now - started, **frame_args)
            if stats is not None:
                stats.add(now, frame_args["debug"])

            cv2.imshow(name, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("d"):
                renderer.ui["debug"]["enabled"] = not renderer.ui["debug"]["enabled"]
    except KeyboardInterrupt:
        pass
    finally:
        feed.close()
        cv2.destroyAllWindows()


def run_bench(args: argparse.Namespace) -> None:
    """실제 장치에서 프레임당 렌더 시간을 잰다. 화면 출력은 하지 않는다."""
    config = load_config(args.config)
    renderer = make_renderer(config, args.theme)
    lanes = [_mock_lane(0.18, 0.43, 0.0), _mock_lane(0.82, 0.57, 0.3)]
    telemetry = {"confidence": 0.7, "fps": 22.4, "inference_ms": 44.6,
                 "departure_distance": 0.3}
    # 부트 애니메이션이 도는 동안은 선이 짧아 측정이 후하게 나온다. 먼저 태운다.
    warm = time.perf_counter()
    while time.perf_counter() - warm < 1.4:
        renderer.render(lanes=lanes, warning="none", elapsed=0.0, telemetry=telemetry)

    # state 마다 그리는 양이 다르다. 2 는 차선을 아예 안 그린다.
    # 무수신 페이드는 상태 바가 2 인 채로 차선을 계속 그리므로 따로 잰다.
    cases = (
        ("none", 0, None, 1.0),
        ("none", 1, None, 1.0),
        ("none", 2, None, 1.0),
        ("none", 2, 0, 0.5),
        ("departure_left", 0, None, 1.0),
    )
    for warning, state, lane_state, opacity in cases:
        kwargs = dict(lanes=lanes, warning=warning, state=state,
                      lane_state=lane_state, lane_opacity=opacity,
                      telemetry=telemetry)
        renderer.render(elapsed=0.0, **kwargs)
        start = time.perf_counter()
        for index in range(args.frames):
            renderer.render(elapsed=index * 0.03, **kwargs)
        each = (time.perf_counter() - start) / args.frames * 1000.0
        label = "fade" if lane_state is not None else ""
        print(f"{warning:16s} state {state} {label:5s} {each:6.1f} ms/frame   "
              f"{1000.0 / each:5.1f} fps")
    print("\ntarget: keep this under 33 ms for 30 fps.")
    print("if it is slower, lower theme.edge_width_ratio.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="반사식 HUD 화면 구성")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preview = subparsers.add_parser("preview", help="네트워크 없이 화면 구성 조정")
    preview.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    preview.add_argument("--windowed", action="store_true")
    preview.add_argument("--theme", action="store_true", help="AR 오버레이 시안으로")
    preview.add_argument("--state", type=int, choices=LANE_STATES, default=0,
                         help="인식 상태 강제 지정. 0 정상 1 주의 2 인식 불가")

    sample = subparsers.add_parser("sample", help="상황별 PNG 저장")
    sample.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    sample.add_argument("--out", default="hud_samples")
    sample.add_argument("--theme", action="store_true", help="AR 오버레이 시안으로")
    sample.add_argument("--state", type=int, choices=LANE_STATES, default=0,
                        help="인식 상태 강제 지정. 0 정상 1 주의 2 인식 불가")

    pattern = subparsers.add_parser(
        "pattern", help="레터박스·반전·굵기 확인용 진단 패턴"
    )
    pattern.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    pattern.add_argument("--windowed", action="store_true")
    pattern.add_argument("--no-grid", action="store_true", help="격자 없이")
    pattern.add_argument("--out", type=Path, help="화면 대신 PNG 로 저장")

    receive = subparsers.add_parser("receive", help="UDP 수신 + 화면 구성")
    receive.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    receive.add_argument("--bind-host")
    receive.add_argument("--port", type=int)
    receive.add_argument("--windowed", action="store_true")
    receive.add_argument("--theme", action="store_true", help="AR 오버레이 시안으로")
    receive.add_argument("--stats", action="store_true",
                         help="2초마다 지연(젯슨 ts 기준)과 버린 패킷 수를 출력")

    bench = subparsers.add_parser("bench", help="이 장치에서 렌더 속도 측정")
    bench.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    bench.add_argument("--theme", action="store_true")
    bench.add_argument("--frames", type=int, default=120)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "preview":
        run_preview(args)
    elif args.command == "sample":
        run_sample(args)
    elif args.command == "pattern":
        run_pattern(args)
    elif args.command == "receive":
        run_receive(args)
    elif args.command == "bench":
        run_bench(args)


if __name__ == "__main__":
    main()
