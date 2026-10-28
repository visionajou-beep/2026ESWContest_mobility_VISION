"""실제 활용 — 캡처보드로 들어오는 화면에서 자차 차선을 인식해 HUD(Raspberry Pi)로 보낸다. 젯슨 Orin Nano 에서 돈다.

    캡처(전용 스레드, 늘 최신 프레임) -> YOLO11s-seg 추론(TensorRT FP16, 960x544) -> 자차 차선 추적기(postprocess)
    -> HUD 메시지(UDP/JSON, 팀원 규격) -> Pi
                                       └ 확인용 화면(브라우저 MJPEG, 그리기 전용 스레드) · 녹화(필요할 때만)

메인 루프는 캡처 -> 추론 -> 후처리 -> HUD 전송(+ 녹화 원본 · 로그)만 한다. 그리기 · JPEG 는 다른 스레드라 HUD 가
그리기를 기다리지 않는다(젯슨 실측 28.4 fps · 캡처 -> HUD 전송 중앙 52.5 ms, 25 W).

사용법 (젯슨에서 — 시연 · 녹화 표준)
    python3 jetson/live.py --model lane_v1_960x544_fp16.engine --device 0 --fps 30 --web 8080 --hud-host 192.168.50.20

    브라우저 http://<젯슨IP>:8080/ (젯슨 모니터면 http://localhost:8080/ + F11). 녹화는 화면의 버튼, 또는 이 터미널에서
    `r 이름` + Enter (정지 `r` + Enter). 녹화 하나 = recordings/<이름>/ 에 raw.mp4 · frames.csv · run.json.

    # 영상 파일로 돌려 보기 (데스크톱에서도 .pt 모델로 돈다)
    python3 jetson/live.py --model best.pt --device 영상.mp4 --save --out result.mp4

캡처는 1280x720 이 표준이다(기본값). 모델 입력(960x540)보다 크고, 학습 · 평가 데이터(BDD100K)가 원래 1280x720 이라
정확도를 잰 것과 같은 경로(1280x720 -> 960x540)로 들어간다. 캡처보드 화면이 상하 반전으로 들어오면 `--flip v`.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]          # 저장소 최상위 — postprocess · jetson 패키지
sys.path.insert(0, str(ROOT))
from postprocess import STATE_HOLD, STATE_LOST, STATE_OK, EgoLaneTracker  # noqa: E402

# 자차 왼쪽 / 오른쪽 / 차로변경 때 셋째 선 (BGR)
COLOR = {"ego_left": (90, 220, 90), "ego_right": (60, 220, 220), "extra": (220, 90, 220)}
STATE_LABEL = {
    STATE_OK:   ("STATE 0 - NORMAL",   (90, 200, 90)),
    STATE_HOLD: ("STATE 1 - HOLDING",  (50, 190, 240)),
    STATE_LOST: ("STATE 2 - FAIL-SAFE", (60, 60, 220)),
}
EMA_ALPHA = 0.15          # 화면에 보이는 신뢰도의 평활 — 판정에는 쓰지 않는다(신뢰도는 정확도가 아니다)

# --web 첫 화면: 실시간 영상 + 녹화 버튼. 영상만 보려면 /stream
PAGE_HTML = """<!doctype html><html><head><meta charset="utf-8"><title>jetson lane</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>body{margin:0;background:#111;color:#eee;font:15px sans-serif}
.bar{display:flex;gap:8px;align-items:center;flex-wrap:wrap;padding:8px 12px}
input{background:#222;color:#eee;border:1px solid #555;padding:6px 8px;width:14em}
button{padding:6px 14px;border:0;border-radius:4px;font-size:15px;cursor:pointer}
#go{background:#c33;color:#fff}#st{background:#555;color:#fff}
#s{margin-left:6px}.rec{color:#f55;font-weight:bold}img{width:100%;display:block}</style></head>
<body><div class="bar"><input id="n" placeholder="이름 (비우면 시각)">
<button id="go" onclick="go('start')">&#9679; 녹화 시작</button>
<button id="st" onclick="go('stop')">&#9632; 정지</button><span id="s"></span></div>
<img src="/stream">
<script>
function show(j){var s=document.getElementById('s');
 if(j.recording){s.className='rec';s.textContent='REC '+j.name+'  '+j.seconds+'s  '+j.frames+' 프레임';}
 else{s.className='';s.textContent='대기'+(j.saved.length?' · 저장됨: '+j.saved.slice(-3).join(', '):'');}}
function go(k){fetch('/rec/'+k+'?name='+encodeURIComponent(document.getElementById('n').value))
 .then(function(r){return r.json()}).then(show);}
setInterval(function(){fetch('/rec/status').then(function(r){return r.json()}).then(show)},1000);
</script></body></html>"""


class WebView:
    """모니터 없이 보기: 브라우저로 MJPEG 스트림을 낸다(판정에는 영향 없음)."""

    def __init__(self, port: int, width: int = 960) -> None:
        self.width = width
        self._jpg: bytes | None = None
        self._lock = threading.Lock()
        self.control: "RecordControl | None" = None     # 녹화 버튼 (main 에서 붙인다)
        view = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, body: bytes, ctype: str) -> None:
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                from urllib.parse import parse_qs, urlparse
                import json
                u = urlparse(self.path)
                if u.path == "/":
                    self._send(PAGE_HTML.encode("utf-8"), "text/html; charset=utf-8")
                    return
                if u.path.startswith("/rec/") and view.control is not None:
                    name = parse_qs(u.query).get("name", [""])[0]
                    if u.path == "/rec/start":
                        view.control.request_start(name)
                    elif u.path == "/rec/stop":
                        view.control.request_stop()
                    self._send(json.dumps(view.control.status(), ensure_ascii=False).encode("utf-8"),
                               "application/json; charset=utf-8")
                    return
                if u.path != "/stream":
                    self.send_response(404); self.end_headers(); return
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while True:
                        with view._lock:
                            jpg = view._jpg
                        if jpg is not None:
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
                        time.sleep(0.03)
                except (BrokenPipeError, ConnectionResetError):
                    return

        self._srv = ThreadingHTTPServer(("0.0.0.0", port), H)
        threading.Thread(target=self._srv.serve_forever, daemon=True).start()
        print(f"브라우저에서 보기: http://<젯슨IP>:{port}/")

    def push(self, frame) -> None:
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (self.width, int(h * self.width / w)))
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            with self._lock:
                self._jpg = buf.tobytes()


class Recorder:
    """녹화 하나: 모델에 들어간 원본(raw.mp4) · 프레임 로그(frames.csv) · 실행 정보(run.json)를 같은 프레임 번호로.

    raw.mp4 는 모델 입력 크기(960x540)로 줄여 저장한다 — ultralytics 와 같은 INTER_LINEAR 라 압축 전 입력이 원래와
    같다. 데스크톱에서 이 파일로 모델과 추적기를 다시 돌려 젯슨의 판정을 프레임 단위로 재현 · 비교한다.
    인코딩은 별도 스레드(라이브를 늦추지 않게), 큐가 차면 기다릴 뿐 버리지 않는다(원본 · 로그 번호가 어긋나면 안 된다).
    """

    # rec_frame = 영상 안 번호, frame = 추적기가 실행 시작부터 센 번호, time_s = 추적기 시계(초).
    # vp_* · lane_w 는 그 프레임의 추적기 상태 — 데스크톱 재실행은 녹화 첫 줄의 값에서 출발한다.
    FIELDS = ["rec_frame", "frame", "time_s", "wall_s", "cap_ms", "proc_ms", "n_masks", "conf_mean",
              "observed", "state", "event", "intent",
              "left_id", "left_age_s", "right_id", "right_age_s",
              "fresh_left", "fresh_right", "center_x", "ref_row", "vp_x", "vp_y", "lane_w", "offset",
              # 왜 그 STATE 인지: 좌 · 우를 몇 초째 못 봤나, 마스크가 어느 필터에서 버려졌나
              "miss_left_s", "miss_right_s", "drop_slope", "drop_reach", "drop_vp", "drop_short",
              "too_far", "vp_warmup", "vp_kept",
              "undrawn_left", "undrawn_right",
              # 캡처: 처리 못 하고 지나간 캡처 프레임 수, 찍힌 뒤 처리 시작까지(ms), 캡처보드가 주는 속도
              "skipped", "frame_age_ms", "cam_fps",
              "view_fps"]

    def __init__(self, folder: str, fps: float, meta: dict, *,
                 raw_size: tuple[int, int] | None = None, overlay: bool = False,
                 codec: str = "mp4v") -> None:
        import csv
        import queue
        self.dir = Path(folder)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.meta = meta
        self.raw_size = raw_size       # (w, h). None 이면 캡처 크기 그대로
        self.overlay = overlay
        self.codec = codec             # 실제로 연 코덱 (avc1 이 안 열리면 mp4v)
        self._writers: dict[str, cv2.VideoWriter] = {}
        self._csv_f = open(self.dir / "frames.csv", "w", newline="", encoding="utf-8")
        self._csv = csv.DictWriter(self._csv_f, fieldnames=self.FIELDS)
        self._csv.writeheader()
        self._q: "queue.Queue" = queue.Queue(maxsize=48)
        self._high = 0
        self._counts = {"raw.mp4": 0, "overlay.mp4": 0}
        self._err: str | None = None
        self._states: dict[int, int] = {}
        self._proc: list[float] = []
        self.n = 0
        self.t0 = time.perf_counter()
        self._th = threading.Thread(target=self._run, daemon=True)
        self._th.start()
        files = "raw.mp4 · " + ("overlay.mp4 · " if overlay else "") + "frames.csv · run.json"
        size = "캡처 크기" if raw_size is None else f"{raw_size[0]}x{raw_size[1]}"
        print(f"기록합니다 -> {self.dir}/ ({files}) · 원본 {size}")

    def _open(self, name: str, frame) -> cv2.VideoWriter:
        h, w = frame.shape[:2]
        for codec in dict.fromkeys([self.codec, "mp4v"]):
            vw = cv2.VideoWriter(str(self.dir / name), cv2.VideoWriter_fourcc(*codec), self.fps, (w, h))
            if vw.isOpened():
                if codec != self.codec:
                    print(f"[주의] {self.codec} 코덱을 열 수 없어 {codec} 로 저장합니다")
                    self.codec = codec
                return vw
        raise RuntimeError(f"{name} 을 열 수 없습니다 (mp4: {self.codec}, mp4v)")

    def _shrink(self, frame):
        if self.raw_size is None or (frame.shape[1], frame.shape[0]) == self.raw_size:
            return None
        return cv2.resize(frame, self.raw_size, interpolation=cv2.INTER_LINEAR)   # ultralytics 와 같은 보간

    def _run(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                return
            name, frame = item
            try:
                if name not in self._writers:
                    self._writers[name] = self._open(name, frame)
                self._writers[name].write(frame)
                self._counts[name] += 1
            except Exception as e:                    # 기록 실패가 라이브를 멈추지 않게
                self._err = self._err or str(e)

    def _put(self, name: str, frame) -> None:
        self._q.put((name, frame))
        self._high = max(self._high, self._q.qsize())

    def write_raw(self, frame) -> None:
        small = self._shrink(frame)
        # 줄인 것은 새 배열이다. 그대로 쓸 때만 복사 — draw() 가 원본 프레임에 직접 그리기도 한다
        self._put("raw.mp4", frame.copy() if small is None else small)

    def write_overlay(self, frame) -> None:
        if not self.overlay:
            return
        small = self._shrink(frame)
        self._put("overlay.mp4", frame if small is None else small)

    @staticmethod
    def _diag_cols(d: dict | None) -> dict:
        if not d:
            return {}
        return {"miss_left_s": f"{d['miss_left']:.3f}", "miss_right_s": f"{d['miss_right']:.3f}",
                "drop_slope": d["slope"], "drop_reach": d["reach"], "drop_vp": d["vp"],
                "drop_short": d["short"], "too_far": d["gate"], "vp_warmup": int(d["warmup"]),
                "vp_kept": d.get("vp_kept", 0),
                "undrawn_left": int(d.get("undrawn_left", False)), "undrawn_right": int(d.get("undrawn_right", False))}

    def log(self, out, tracker, cap_ms: float, proc_ms: float, n_masks: int,
            conf_mean: float | None, extra: dict | None = None) -> None:
        def b_id(b):
            return "" if b is None else b.track_id

        def b_age(b):
            return "" if b is None else f"{b.age_s:.3f}"

        vp, lw = tracker.vp, tracker.lane_w
        self._states[out.state] = self._states.get(out.state, 0) + 1
        self._proc.append(proc_ms)
        self._csv.writerow({
            "rec_frame": self.n,
            "lane_w": "" if lw is None else f"{lw:.4f}",
            "frame": out.frame, "time_s": f"{out.time_s:.4f}",
            "wall_s": f"{time.perf_counter() - self.t0:.3f}",
            "cap_ms": f"{cap_ms:.1f}", "proc_ms": f"{proc_ms:.1f}", "n_masks": n_masks,
            "conf_mean": "" if conf_mean is None else f"{conf_mean:.3f}",
            "observed": out.observed, "state": out.state, "event": out.event, "intent": out.intent,
            "left_id": b_id(out.ego_left), "left_age_s": b_age(out.ego_left),
            "right_id": b_id(out.ego_right), "right_age_s": b_age(out.ego_right),
            "fresh_left": int(out.fresh_left), "fresh_right": int(out.fresh_right),
            "center_x": f"{out.center_x:.4f}", "ref_row": f"{out.ref_row:.4f}",
            "vp_x": "" if vp is None else f"{vp[0]:.4f}",
            "vp_y": "" if vp is None else f"{vp[1]:.4f}",
            "offset": "" if out.offset is None else f"{out.offset:.3f}",
        } | self._diag_cols(out.diag) | (extra or {}))
        self.n += 1

    def close(self, summary: dict | None = None) -> None:
        import json
        wall = time.perf_counter() - self.t0        # 녹화 구간 길이(인코딩 마무리 시간 제외)
        self._q.put(None)
        self._th.join()
        for vw in self._writers.values():
            vw.release()
        self._csv_f.close()
        info = dict(self.meta)
        info.update(summary or {})
        p = np.asarray(self._proc[1:] or self._proc or [0.0])
        info.update({"states": {str(k): v for k, v in sorted(self._states.items())},
                     "proc_ms_mean": round(float(p.mean()), 2),
                     "seconds": round(wall, 2),
                     "codec": self.codec, "overlay": self.overlay,
                     "raw_size": None if self.raw_size is None else list(self.raw_size)})
        info.update({"frames_logged": self.n, "frames_raw": self._counts["raw.mp4"],
                     "frames_overlay": self._counts["overlay.mp4"],
                     "loop_fps_wall": round(self.n / wall, 2) if wall > 0 else None,
                     "writer_queue_high": self._high, "writer_error": self._err})
        (self.dir / "run.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
        ok = (self.n == self._counts["raw.mp4"]
              and (not self.overlay or self.n == self._counts["overlay.mp4"]) and self._err is None)
        ov = f" · 오버레이 {self._counts['overlay.mp4']}" if self.overlay else ""
        print(f"기록 완료: {self.dir}/  로그 {self.n} · 원본 {self._counts['raw.mp4']}{ov} 프레임 · "
              f"루프 {info['loop_fps_wall']} fps" + ("" if ok else "  [주의] 개수 불일치 또는 오류: "
                                                    + str(self._err)))


class RecordControl:
    """라이브 · HUD 전송은 계속 돌리고 녹화만 켰다 껐다 한다(프로세스를 다시 띄우면 HUD 가 끊기고 소실점을 다시 배운다).

    켜고 끄는 곳: 브라우저 버튼, 또는 이 터미널에서 `r` + Enter(`r 이름`). 실제 전환은 메인 루프의 프레임 경계(apply)
    에서만 한다 — 원본 · 로그가 같은 프레임에서 시작하고 끝나야 짝이 맞는다. 멈출 때 남은 인코딩은 뒤에서 마무리한다.
    """

    def __init__(self, root: str, fps: float, meta: dict, **recorder_kw) -> None:
        self.root = Path(root)
        self.fps = fps
        self.meta = meta
        self.recorder_kw = recorder_kw
        self.rec: Recorder | None = None
        self.name = ""
        self._t = 0.0
        self._want: tuple[str, str] | None = None
        self._lock = threading.Lock()
        self._closers: list[threading.Thread] = []
        self.saved: list[str] = []

    def request_start(self, name: str = "") -> None:
        with self._lock:
            self._want = ("start", name)

    def request_stop(self) -> None:
        with self._lock:
            self._want = ("stop", "")

    def toggle(self, name: str = "") -> None:
        if self.rec is not None:
            self.request_stop()
        else:
            self.request_start(name)

    def _folder(self, name: str) -> Path:
        import re
        base = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._") or time.strftime("rec_%m%d_%H%M%S")
        folder, k = self.root / base, 2
        while folder.exists():                     # 덮어쓰지 않는다
            folder, k = self.root / f"{base}_{k}", k + 1
        return folder

    def apply(self) -> "Recorder | None":
        """프레임 시작마다 부른다. 이번 프레임에 쓸 Recorder 를 돌려준다."""
        with self._lock:
            want, self._want = self._want, None
        if want is not None:
            kind, name = want
            if kind == "start" and self.rec is None:
                folder = self._folder(name)
                self.rec = Recorder(str(folder), self.fps,
                                    dict(self.meta, name=folder.name,
                                         started=time.strftime("%Y-%m-%d %H:%M:%S")),
                                    **self.recorder_kw)
                self.name, self._t = folder.name, time.perf_counter()
            elif kind == "stop" and self.rec is not None:
                r, self.rec = self.rec, None
                th = threading.Thread(target=r.close, kwargs={"summary": {"stopped_by": "request"}})
                th.start()
                self._closers.append(th)
                self.saved.append(self.name)
        return self.rec

    def status(self) -> dict:
        rec = self.rec
        return {"recording": rec is not None, "name": self.name if rec else "",
                "seconds": round(time.perf_counter() - self._t, 1) if rec else 0,
                "frames": rec.n if rec else 0, "saved": list(self.saved), "root": str(self.root)}

    def close_all(self) -> None:
        if self.rec is not None:
            self.saved.append(self.name)
            self.rec.close({"stopped_by": "exit"})
            self.rec = None
        for th in self._closers:
            th.join()

    def listen_stdin(self) -> None:
        """터미널에서 r + Enter 로 켜고 끈다(SSH 로 띄운 경우). 터미널이 아니면 아무것도 안 한다."""
        if not sys.stdin or not sys.stdin.isatty():
            return

        def loop():
            for line in sys.stdin:
                parts = line.strip().split(maxsplit=1)
                if parts and parts[0].lower() == "r":
                    self.toggle(parts[1] if len(parts) > 1 else "")
        threading.Thread(target=loop, daemon=True).start()
        print("녹화: 이 터미널에서 r + Enter 로 시작/정지 (r 이름 → 그 이름으로)")


def model_input_size(w: int, h: int, imgsz) -> tuple[int, int]:
    """ultralytics LetterBox 가 이 캡처를 줄여 넣는 크기 (w, h) — 패딩 전, 키우지는 않는다. 1280x720 -> 960x540."""
    th, tw = (imgsz, imgsz) if isinstance(imgsz, int) else imgsz
    r = min(th / h, tw / w, 1.0)
    return int(round(w * r)), int(round(h * r))


class LatestFrame:
    """캡처 전용 스레드: 캡처보드가 주는 프레임을 계속 받아 가장 최근 한 장만 둔다.

    한 바퀴 일(약 35 ms)이 프레임 간격(33 ms)보다 조금 길면, 차례대로 읽는 방식은 매번 다음 프레임을 기다려 두 장
    간격(15 fps)에 묶인다. 따로 받아 두면 루프는 자기 속도로 돌고 늘 최신 프레임을 쓴다.
    get() 은 새 프레임이 올 때까지만 기다리고(같은 프레임을 두 번 처리하지 않는다) 건너뛴 수 · 찍힌 시각을 돌려준다.
    영상 파일에는 쓰지 않는다 — 한 장도 건너뛰면 안 된다(재현).
    """

    def __init__(self, cap, flip_code: int | None = None) -> None:
        self.cap = cap
        self.flip_code = flip_code
        self._cond = threading.Condition()
        self._frame = None
        self._seq = 0
        self._t = 0.0
        self._ok = True
        self._stop = False
        self._gaps: list[float] = []            # 캡처보드가 실제로 주는 간격 — 화면의 cam fps
        self._th = threading.Thread(target=self._run, daemon=True)
        self._th.start()

    def _run(self) -> None:
        while not self._stop:
            ok, f = self.cap.read()
            t = time.perf_counter()
            if ok and self.flip_code is not None:
                f = cv2.flip(f, self.flip_code)
            with self._cond:
                if not ok:
                    self._ok = False
                    self._cond.notify_all()
                    return
                if self._seq:
                    self._gaps.append(t - self._t)
                    if len(self._gaps) > 30:
                        self._gaps.pop(0)
                self._frame, self._t = f, t            # read() 는 매번 새 배열 — 넘긴 것을 덮어쓰지 않는다
                self._seq += 1
                self._cond.notify_all()

    def cam_fps(self) -> float | None:
        with self._cond:
            return 1.0 / float(np.mean(self._gaps)) if self._gaps else None

    def get(self, last_seq: int, timeout: float = 2.0):
        """last_seq 보다 새 프레임: (ok, frame, seq, 찍힌 시각). 끝났거나 시간 초과면 ok=False."""
        with self._cond:
            self._cond.wait_for(lambda: self._seq > last_seq or not self._ok or self._stop, timeout)
            if self._seq <= last_seq:
                return False, None, last_seq, 0.0
            return True, self._frame, self._seq, self._t

    def stop(self) -> None:
        self._stop = True
        self._th.join(timeout=1.0)


class ConfidenceEMA:
    """화면에 보여 줄 신뢰도만 평활한다. State 판정에는 관여하지 않는다."""

    def __init__(self, alpha: float = EMA_ALPHA):
        self.alpha = alpha
        self.value: float | None = None

    def update(self, x: float) -> float:
        self.value = x if self.value is None else self.alpha * x + (1 - self.alpha) * self.value
        return self.value


class DrawWorker:
    """그리기 + 브라우저용 JPEG 를 메인 루프 밖(전용 스레드)에서 한다. 늦으면 그 사이 프레임은 화면에서만 건너뛴다 —
    HUD 전송 · 녹화 원본 · 로그는 한 장도 빠지지 않는다(그리기 13 ms 가 끼면 HUD 가 20 번/초에 묶인다)."""

    def __init__(self, web: "WebView") -> None:
        self.web = web
        self._cond = threading.Condition()
        self._job: tuple | None = None
        self._stop = False
        self.drawn = 0
        self.skipped = 0
        self._err: str | None = None
        self._gaps: list[float] = []     # 화면 갱신 간격 — 화면의 view fps
        self._t: float | None = None
        self._th = threading.Thread(target=self._run, daemon=True)
        self._th.start()

    def submit(self, job: tuple) -> None:
        """job = (draw 의 위치 인자, 키워드 인자). 못 그린 이전 것은 버린다 — 화면은 늘 최신."""
        with self._cond:
            if self._job is not None:
                self.skipped += 1
            self._job = job
            self._cond.notify()

    def _run(self) -> None:
        while True:
            with self._cond:
                self._cond.wait_for(lambda: self._job is not None or self._stop)
                if self._job is None:
                    return
                (args, kw), self._job = self._job, None
            try:
                self.web.push(draw(*args, view_fps=self.view_fps(), **kw))
            except Exception as e:                   # 그리기가 죽어도 HUD 전송은 계속
                if self._err is None:
                    self._err = repr(e)
                    print(f"[주의] 화면 그리기 오류 (HUD 전송 · 녹화는 계속): {e!r}", flush=True)
                continue
            t = time.perf_counter()
            with self._cond:
                if self._t is not None:
                    self._gaps.append(t - self._t)
                    if len(self._gaps) > 30:
                        self._gaps.pop(0)
                self._t = t
                self.drawn += 1

    def view_fps(self) -> float | None:
        with self._cond:
            return 1.0 / float(np.mean(self._gaps)) if self._gaps else None

    def stop(self) -> None:
        with self._cond:
            self._stop = True
            self._cond.notify()
        self._th.join(timeout=2.0)


def _put_right(frame, text: str, row: int, s: float, color, scale: float = 0.7, th: int = 2) -> None:
    """윗줄 막대의 오른쪽 끝에 맞춰 쓴다."""
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale * s, th)
    cv2.putText(frame, text, (frame.shape[1] - tw - int(14 * s), int((34 + 36 * row) * s)),
                cv2.FONT_HERSHEY_SIMPLEX, scale * s, color, th, cv2.LINE_AA)


def draw(frame, out, conf_ema: float | None, fps: float, cap_ms: float,
         loop_fps: float | None = None, conf_now: float | None = None, cam_fps: float | None = None,
         view_fps: float | None = None, rec_text: str = ""):
    """자차 차선만 그린다. 무엇을 그릴지는 `out.display_line(b)` — HUD 메시지와 같은 규칙. State 2 면 아무것도 안 그린다.

    선은 안정화 중심선을 도색 폭만큼의 띠로 그린다(유지 중이면 테두리만). 오른쪽 위: loop = 캡처 -> 추론 -> 후처리 ->
    HUD 전송 한 바퀴의 실제 속도(= HUD 갱신 속도), view = 이 화면의 갱신 속도, cam = 캡처보드가 주는 속도,
    infer = 추론 + 후처리만. 글자 크기는 화면 높이에 맞춰 줄인다. frame 에 직접 그리기도 한다(State 2).
    """
    h, w = frame.shape[:2]
    s = h / 1080.0
    label, color = STATE_LABEL[out.state]

    if out.state != STATE_LOST:
        layer = frame.copy()
        for name, b in (("ego_left", out.ego_left), ("ego_right", out.ego_right), ("extra", out.extra)):
            line = out.display_line(b)
            if line is None:
                continue
            if getattr(b, "inferred", False):
                # 개선 ⑨ 역산 선 — 본 선이 아니라는 것이 보이게 중심선 점선으로(3 행 그리고 2 행 쉼)
                P = np.round(np.stack([line[:, 0] * w, line[:, 1] * h], 1)).astype(np.int32)
                for k in range(len(P) - 1):
                    if (k // 3) % 5 < 3:
                        cv2.line(layer, (int(P[k][0]), int(P[k][1])), (int(P[k + 1][0]), int(P[k + 1][1])),
                                 COLOR[name], max(3, w // 240), cv2.LINE_AA)
                continue
            hw = b.line_halfwidth
            hw = np.full(len(line), 0.004) if hw is None else np.clip(np.asarray(hw), 0.003, 0.03)
            left_e = np.stack([(line[:, 0] - hw) * w, line[:, 1] * h], 1)
            right_e = np.stack([(line[:, 0] + hw) * w, line[:, 1] * h], 1)
            band = np.round(np.vstack([left_e, right_e[::-1]])).astype(np.int32)
            if b.held:
                cv2.polylines(layer, [band], True, COLOR[name], max(2, w // 640), cv2.LINE_AA)
            else:
                cv2.fillPoly(layer, [band], COLOR[name])
        frame = cv2.addWeighted(layer, 0.7, frame, 0.3, 0)

    for i, (text, col, scale, th) in enumerate(status_lines(out, conf_ema, conf_now, label, color)):
        if i == 0:
            cv2.rectangle(frame, (0, 0), (w, int(150 * s)), (20, 20, 20), -1)
        cv2.putText(frame, text, (int(14 * s), int((34 + 36 * i) * s)), cv2.FONT_HERSHEY_SIMPLEX,
                    scale * s, col, th, cv2.LINE_AA)
    perf = [f"loop {loop_fps:4.1f} FPS"] if loop_fps is not None else []
    if view_fps is not None:
        perf.append(f"view {view_fps:4.1f}")
    if cam_fps is not None:
        perf.append(f"cam {cam_fps:4.1f}")
    perf += [f"infer {fps:4.1f}", f"cap {cap_ms:4.1f}ms"]
    _put_right(frame, "   ".join(perf), 0, s, (255, 255, 255))
    if out.event:
        _put_right(frame, out.event.upper(), 1, s, COLOR["extra"])
    if rec_text:
        _put_right(frame, rec_text, 2, s, (40, 40, 235), scale=0.8)
    return frame


def side_status(b, miss_s: float, limit: float, undrawn: bool = False) -> str:
    """한쪽 자차 선: seen(이번 프레임) / no line(봤지만 안정화 선이 없어 안 그림 — 못 본 시간으로 셈) /
    hold(마지막 모양 유지) / missing(아직 허용 안) / LOST(허용 초과 = State 2 의 원인) / none."""
    if b is not None and b.age_s == 0.0:
        if undrawn:
            return f"LOST no line {miss_s:.2f}s" if miss_s > limit else f"no line {miss_s:.2f}s"
        return "seen"
    if b is not None:
        return f"hold {b.age_s:.2f}s"
    if miss_s <= 0:
        return "none"
    return f"LOST {miss_s:.2f}s" if miss_s > limit else f"missing {miss_s:.2f}s"


def status_lines(out, conf_ema, conf_now, label, color):
    """화면 왼쪽 위 네 줄 — STATE 와 왜 그 STATE 인지(좌 · 우를 몇 초째 못 봤나, 마스크가 어디서 버려졌나)."""
    d = out.diag or {}
    ml, mr, lim = d.get("miss_left", 0.0), d.get("miss_right", 0.0), d.get("miss_limit", 0.3)
    head = label
    if out.state == STATE_LOST:
        why = [f"{s} {'no line' if u else 'unseen'} {m:.2f}s"
               for s, m, u in (("L", ml, d.get("undrawn_left")), ("R", mr, d.get("undrawn_right"))) if m > lim]
        head += "   <- " + (", ".join(why) if why else "no ego pair") + f" (> {lim:.2f}s)"
    sides = (f"L {side_status(out.ego_left, ml, lim, bool(d.get('undrawn_left')))}    "
             f"R {side_status(out.ego_right, mr, lim, bool(d.get('undrawn_right')))}")
    if d:
        drops = "  ".join(f"{k} {d[k]}" for k in ("slope", "reach", "vp", "short") if d.get(k))
        mask = f"masks {d.get('masks', 0)} -> used {out.observed}"
        if drops:
            mask += f"   dropped: {drops}"
        if d.get("gate"):
            mask += f"   too far {d['gate']}"
        if d.get("vp_kept"):
            mask += f"   kept by track {d['vp_kept']}"     # 소실점 검사가 버릴 뻔한 추적 중인 자차 선
        if d.get("warmup"):
            mask += "   (vanishing point warming up)"
    else:
        mask = f"used {out.observed}"
    bits = []
    if out.offset is not None:
        bits.append(f"offset {out.offset:+.2f}")
    if out.intent:
        bits.append(f"intent {out.intent}")
    # conf = 이번 프레임 모든 마스크의 평균(표시용). 검출이 없으면 '-'
    bits.append("conf -" if conf_now is None or conf_ema is None else f"conf {conf_ema:.2f}")
    return [(head, color, 0.9, 2),
            (sides, (235, 235, 235), 0.65, 1),
            (mask, (200, 200, 200), 0.6, 1),
            ("  ".join(bits), (200, 200, 200), 0.6, 1)]


def main() -> None:
    ap = argparse.ArgumentParser(description="캡처 -> 추론 -> 자차 차선 추적 -> HUD 전송 (젯슨)")
    ap.add_argument("--model", default="lane_v1_960x544_fp16.engine", help="TensorRT 엔진(.engine) 또는 .pt / .onnx")
    ap.add_argument("--device", default="0", help="/dev/video 번호(예: 0) 또는 영상 파일 경로")
    ap.add_argument("--imgsz", default="544,960", help="정수 하나(960) 또는 h,w (544,960). 엔진 모양과 같아야 한다")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--fourcc", default="MJPG", help="캡처 포맷. USB 캡처보드는 MJPG 여야 1080p30 이 나온다. 끄려면 none")
    ap.add_argument("--flip", default="none", choices=["none", "v", "h", "180"],
                    help="캡처 화면이 뒤집혀 들어올 때: v 상하, h 좌우, 180 회전")
    ap.add_argument("--cap-width", type=int, default=1280)
    ap.add_argument("--cap-height", type=int, default=720)
    ap.add_argument("--fps", type=float, default=0,
                    help="캡처 fps(녹화 파일 fps). 0 이면 장치에 묻고, 실패하면 30. 캡처보드 입력이면 추적기는 실제 찍힌 시각으로 센다")
    ap.add_argument("--hud-host", default=None, help="Pi 주소: 프레임마다 HUD 메시지(UDP/JSON) 전송")
    ap.add_argument("--hud-port", type=int, default=5005)
    ap.add_argument("--web", type=int, default=0, help="이 포트로 브라우저 화면(MJPEG + 녹화 버튼). 예 8080")
    ap.add_argument("--save", action="store_true", help="화면 대신 파일로 저장(--out)")
    ap.add_argument("--out", default="live_result.mp4")
    ap.add_argument("--rec-dir", default="recordings", help="녹화 폴더들을 둘 곳")
    ap.add_argument("--rec-overlay", action="store_true", help="녹화에 오버레이 화면(overlay.mp4)도 저장(인코딩 두 배)")
    ap.add_argument("--rec-full", action="store_true", help="원본을 캡처 크기 그대로 저장(기본은 모델 입력 크기 960x540)")
    ap.add_argument("--rec-codec", default="mp4v", choices=["mp4v", "avc1"])
    ap.add_argument("--record", default="", help="이 이름으로 처음부터 끝까지 녹화")
    ap.add_argument("--seconds", type=int, default=0, help="이만큼 돌고 끝낸다(0 = 계속)")
    args = ap.parse_args()

    parts = [int(x) for x in str(args.imgsz).split(",")]
    imgsz = parts[0] if len(parts) == 1 else tuple(parts)

    if not Path(args.model).exists():
        raise SystemExit(f"[에러] 모델이 없습니다: {args.model}")

    dev = int(args.device) if args.device.isdigit() else args.device
    cap = cv2.VideoCapture(dev)
    if not cap.isOpened():
        raise SystemExit(f"[에러] 캡처 장치를 열 수 없습니다: {args.device}\n"
                         "v4l2-ctl --list-devices 로 장치 번호를 확인하세요.")
    is_camera = isinstance(dev, int)
    if is_camera:
        if args.fourcc and args.fourcc.lower() != "none":
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.fourcc.upper()))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.cap_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cap_height)
        if args.fps:
            cap.set(cv2.CAP_PROP_FPS, args.fps)
        # 캡처 전용 스레드가 계속 꺼내 가므로 쌓이지 않는다. 버퍼가 하나면 캡처보드가 한 장 걸러 채워 15 fps 가 될 수 있다
        set_ok = cap.set(cv2.CAP_PROP_BUFFERSIZE, 4)
        print(f"캡처 버퍼 4개 요청 → {'적용' if set_ok else '이 백엔드에서 지원 안 함'} "
              f"(지금 값 {cap.get(cv2.CAP_PROP_BUFFERSIZE):.0f})")

    fps_nominal = args.fps or float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    got_w, got_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fcc_s = "".join(chr((fcc >> (8 * i)) & 0xFF) for i in range(4)) if fcc else "?"
    try:
        backend = cap.getBackendName()
    except Exception:
        backend = "?"
    print(f"입력 {args.device}: {got_w}x{got_h} @ {cap.get(cv2.CAP_PROP_FPS):.1f} fps 포맷 {fcc_s} · {backend}"
          + ("" if is_camera else " (파일)"))
    if is_camera and (got_w != args.cap_width or got_h != args.cap_height):
        print(f"[주의] 요청한 {args.cap_width}x{args.cap_height} 가 아니라 {got_w}x{got_h} 로 열렸습니다. "
              "캡처보드가 지원하는 해상도인지 v4l2-ctl -d /dev/videoN --list-formats-ext 로 확인하세요.")
    model = YOLO(args.model, task="segment")
    hud = None
    if args.hud_host:
        from jetson.hud_message import HudSender
        hud = HudSender(args.hud_host, args.hud_port)
        print(f"HUD 메시지: {args.hud_host}:{args.hud_port} (좌우 공통 곡률 + 교차 방지)")
    tracker = EgoLaneTracker(fps_nominal)
    ema = ConfidenceEMA()

    print(f"모델 {args.model} · imgsz {imgsz} · 카메라 {fps_nominal:.1f} fps")
    print(f"후처리: 자차 차선 추적기 (유지 {tracker.hold_s:.2f}s · State2 {tracker.miss_s:.2f}s)")
    # 캡처보드 입력은 실제 찍힌 시각으로 센다 — 루프가 30 fps 보다 느려도 유지 · State 2 · 평활이 초를 지킨다.
    # 영상 파일은 한 장씩 차례대로라 '한 장 = 1/fps 초'.
    real_clock = is_camera
    print("추적기 시계: " + ("캡처 시각 (실제 초 — 루프 속도와 무관)" if real_clock
                           else f"프레임 수 x 1/{fps_nominal:.0f} 초 (영상 파일)"))

    web = WebView(args.web) if args.web else None
    writer = None
    if args.save:
        ok, first = cap.read()
        if not ok:
            raise SystemExit("[에러] 캡처 장치에서 프레임을 못 읽었습니다.")
        h, w = first.shape[:2]
        writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps_nominal, (w, h))
        print(f"파일로 저장합니다 -> {args.out}")
    # 데스크톱 재분석은 같은 판이어야 의미가 있다 — 이 실행의 후처리 파일 해시를 남긴다
    sha = {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest()[:16]
           for f in ("postprocess/tracker.py", "postprocess/geometry.py", "postprocess/missing_lane.py", "jetson/hud_message.py", "jetson/live.py")
           if (ROOT / f).exists()}
    control = RecordControl(args.rec_dir, fps_nominal, {
        "sha256_16": sha,
        "model": args.model, "imgsz": list(imgsz) if isinstance(imgsz, tuple) else imgsz,
        "conf": args.conf, "device": args.device, "flip": args.flip,
        "fps_nominal": fps_nominal, "capture": f"{got_w}x{got_h} {fcc_s}",
        "hold_s": tracker.hold_s, "miss_s": tracker.miss_s,
        "hud_host": args.hud_host, "draw": "thread" if web is not None else "sync",
        "tracker_clock": "capture" if real_clock else "frames",
        "preset": "adopted",            # 채택한 후처리 설정 — 추적기에 내장
    }, raw_size=None if args.rec_full else model_input_size(got_w, got_h, imgsz),
        overlay=args.rec_overlay, codec=args.rec_codec)
    if web is not None:
        web.control = control
    if args.record:
        control.request_start(args.record)
    control.listen_stdin()
    if args.save:
        pass
    elif web is not None:
        print("브라우저 화면을 냅니다 (녹화 버튼 포함). 종료는 Ctrl+C.")
    elif args.record:
        print("기록만 합니다(창 없음). 종료는 Ctrl+C 또는 --seconds.")
    else:
        cv2.namedWindow("ego lane", cv2.WINDOW_NORMAL)
        print("창에 표시합니다. 종료하려면 창에서 'q' 키.")
    headless = bool(args.save or web is not None or args.record)

    t_start = time.perf_counter()
    n = 0
    fps_hist: list[float] = []
    proc_ms_all: list[float] = []      # 추론 + 후처리 + HUD 전송 (캡처 제외)
    cap_ms_all: list[float] = []
    states = {STATE_OK: 0, STATE_HOLD: 0, STATE_LOST: 0}
    loop_hist: list[float] = []        # 한 바퀴 간격 — 화면의 loop FPS(= HUD 갱신 속도)
    t_prev: float | None = None
    flip_code = None if args.flip == "none" else {"v": 0, "h": 1, "180": -1}[args.flip]
    grab = LatestFrame(cap, flip_code) if is_camera else None
    print("캡처: " + ("전용 스레드 · 최신 프레임" if grab is not None else "차례대로 읽기 (한 장씩)"))
    # 매 장이 필요한 곳(창 · --save · 녹화 오버레이)은 그 프레임만 메인 루프에서 그린다
    drawer = DrawWorker(web) if web is not None else None
    print("그리기: " + ("전용 스레드 · 최신 프레임 (HUD 전송은 그리기를 기다리지 않음)" if drawer is not None
                       else "메인 루프에서 매 장"))
    last_seq = 0
    try:
        while True:
            t_cap = time.perf_counter()
            if t_prev is not None:
                loop_hist.append(t_cap - t_prev)
                if len(loop_hist) > 30:
                    loop_hist.pop(0)
            t_prev = t_cap
            skipped, age_ms = 0, 0.0
            t_frame = None                              # 추적기 시계: 이 프레임이 찍힌 시각(초)
            if grab is not None:
                ok, frame, seq, t_shot = grab.get(last_seq)
                if ok:
                    skipped = seq - last_seq - 1 if last_seq else 0
                    last_seq = seq
                    age_ms = (time.perf_counter() - t_shot) * 1000    # 찍힌 뒤 처리 시작까지
                    t_frame = t_shot
            else:
                ok, frame = cap.read()
                if ok and flip_code is not None:
                    frame = cv2.flip(frame, flip_code)
            cap_ms = (time.perf_counter() - t_cap) * 1000
            if not ok:
                print("[주의] 프레임을 못 읽었습니다. 캡처보드 연결 또는 영상 끝.")
                break
            rec = control.apply()                       # 녹화 시작 · 정지는 프레임 경계에서만
            if rec is not None:
                rec.write_raw(frame)                    # 모델이 볼 그대로, 그리기 전에

            t0 = time.perf_counter()
            res = model.predict(frame, imgsz=imgsz, conf=args.conf, verbose=False)[0]
            polys: list[np.ndarray] = []
            if res.masks is not None:
                polys = [np.asarray(p, dtype=np.float32) for p in res.masks.xyn if len(p) >= 3]
            out = tracker.update(polys, t=t_frame)
            if hud is not None:
                # fps = 추론 + 후처리 속도(최근 30 장 평균, 화면의 infer 와 같은 값) — 규격의 '모니터링용 FPS'
                hud.send(out, fps=float(np.mean(fps_hist)) if fps_hist else 0.0)
            dt = time.perf_counter() - t0

            fps_hist.append(1.0 / max(dt, 1e-6))
            if len(fps_hist) > 30:
                fps_hist.pop(0)
            proc_ms_all.append(dt * 1000)
            cap_ms_all.append(cap_ms)
            confs = [float(b.conf) for b in (res.boxes or [])]
            if confs:
                ema.update(float(np.mean(confs)))
            states[out.state] += 1

            loop_fps = 1.0 / float(np.mean(loop_hist)) if loop_hist else None
            conf_now = float(np.mean(confs)) if confs else None
            rec_text = ""
            if rec is not None:
                st = control.status()
                rec_text = f"REC {st['seconds']:5.1f}s  {st['name']}"
            job = ((frame, out, ema.value, float(np.mean(fps_hist)), cap_ms, loop_fps, conf_now,
                    grab.cam_fps() if grab is not None else None), {"rec_text": rec_text})
            shown = None
            if drawer is not None and writer is None and not (rec is not None and rec.overlay):
                drawer.submit(job)                      # frame 은 여기서부터 그리기 스레드 것
            else:
                shown = draw(*job[0], **job[1])
                if web is not None:
                    web.push(shown)
                if rec is not None:
                    rec.write_overlay(shown)
            if rec is not None:
                rec.log(out, tracker, cap_ms, dt * 1000, len(polys), conf_now,
                        extra={"skipped": skipped, "frame_age_ms": f"{age_ms:.1f}",
                               "cam_fps": "" if grab is None or grab.cam_fps() is None else f"{grab.cam_fps():.1f}",
                               "view_fps": "" if drawer is None or drawer.view_fps() is None
                               else f"{drawer.view_fps():.1f}"})

            n += 1
            if writer is not None:
                writer.write(shown)
            if headless:
                if n % 50 == 0:
                    print(f"   ...{n}프레임, {STATE_LABEL[out.state][0]}", flush=True)
                if args.seconds and (time.perf_counter() - t_start) >= args.seconds:
                    break
            else:
                cv2.imshow("ego lane", shown)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    except KeyboardInterrupt:
        print("\n중단됨.")

    wall = time.perf_counter() - t_start
    if grab is not None:
        grab.stop()
    if drawer is not None:
        drawer.stop()
    cap.release()
    total = max(sum(states.values()), 1)
    if proc_ms_all:
        p = np.asarray(proc_ms_all[10:] or proc_ms_all)     # 첫 10 프레임은 워밍업
        c = np.asarray(cap_ms_all[10:] or cap_ms_all)
        print(f"\n추론+후처리: 평균 {p.mean():.1f} ms/프레임 (p95 {np.percentile(p, 95):.1f}) -> {1000 / p.mean():.1f} FPS  "
              f"| 캡처/디코드: 평균 {c.mean():.1f} ms  | 벽시계 전체(= HUD 갱신): {len(proc_ms_all) / wall:.1f} FPS")
        if drawer is not None:
            print(f"  화면(그리기 스레드): {drawer.drawn / wall:.1f} FPS · 그린 {drawer.drawn} · 화면에서만 건너뜀 "
                  f"{drawer.skipped} (HUD 전송 · 녹화는 매 장)")
    print(f"\nState 0 {states[STATE_OK]} ({states[STATE_OK]/total:.1%}) · "
          f"1 {states[STATE_HOLD]} ({states[STATE_HOLD]/total:.1%}) · "
          f"2 {states[STATE_LOST]} ({states[STATE_LOST]/total:.1%})")
    control.close_all()
    if control.saved:
        print(f"녹화 {len(control.saved)}개: {control.root}/ " + ", ".join(control.saved))
    if writer is not None:
        writer.release()
        print(f"저장 완료: {args.out} ({n}프레임)")
    else:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
