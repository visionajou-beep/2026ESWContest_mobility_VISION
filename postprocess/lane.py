"""추적기가 쓰고 내보내는 자료형."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import HOLD_SEC

STATE_OK, STATE_HOLD, STATE_LOST = 0, 1, 2   # 0 두 선 정상 · 1 주의(한쪽 유지 또는 한 줄) · 2 안 그림
EGO_ROLES = ("ego_left", "ego_right")


class TrackerError(RuntimeError):
    pass


@dataclass
class Track:
    """프레임을 넘어 따라가는 도색 선 하나."""

    tid: int
    x: float                 # 기준행 위치 (차 가까이에서 읽어 기준행으로 옮긴 값)
    poly: np.ndarray         # 마지막으로 본 마스크 폴리곤
    side: int                # -1 경로 왼쪽 · +1 오른쪽 · 0 경로에 걸침
    first_seen: float        # 처음 본 시각 (추적기 시계)
    last_seen: float
    hits: int = 1
    confident_side: int = 0  # 마지막으로 분명하게 있던 쪽
    # 이중선 처리용: 최근 위치 기록 (시계, x, 아래 끝 행), 배운 두 획 간격, 이번 값이 바깥 획을 옮긴 것인지
    hist: list = field(default_factory=list)
    stroke: tuple | None = None
    from_outer: bool = False
    inner_u: float = -1e9    # 안쪽 획을 마지막으로 본 시계
    inner_row: float = 0.0   # 그 마스크의 아래 끝 행

    def observe(self, x: float, poly: np.ndarray, frame: float, side: int) -> None:
        self.x = x
        self.poly = poly
        self.from_outer = False
        self.side = side
        if side:
            self.confident_side = side
        self.last_seen = frame
        self.hits += 1


@dataclass
class Boundary:
    """화면 · HUD 로 넘기는 선 하나."""

    track_id: int
    poly: np.ndarray
    x: float
    age_s: float             # 마지막으로 본 뒤 지난 시간 (이번 프레임에 봤으면 0)
    held: bool
    line: np.ndarray | None = None             # 다듬은 중심선 (N, 2) — 실제로 그리는 선
    line_halfwidth: np.ndarray | None = None   # 행마다 도색 폭의 절반
    from_outer: bool = False                   # 이중선의 바깥 획을 안쪽으로 옮긴 값인지
    inferred: bool = False                     # 본 선이 아니라 반대쪽에서 역산한 선인지

    def drawable(self, hold_s: float) -> bool:
        return self.age_s <= hold_s


@dataclass
class TrackerOutput:
    """한 프레임의 결과."""

    frame: int
    time_s: float
    ego_left: Boundary | None
    ego_right: Boundary | None
    extra: Boundary | None       # 차로변경 때 가려는 쪽의 셋째 선
    state: int                   # STATE_OK / STATE_HOLD / STATE_LOST
    event: str                   # "" | "change_left" | "change_right" — 이번 프레임에 선을 넘었다
    intent: str                  # "" | "left" | "right" — 그쪽으로 다가가는 중
    center_x: float              # 진행 경로가 기준행을 지나는 위치
    ref_row: float               # 기준행
    offset: float | None         # 차로 안 위치 (-1 왼쪽 선 ~ +1 오른쪽 선)
    observed: int                # 자차 후보가 된 마스크 수
    fresh_left: bool             # 이번 프레임에 실제로 본 선인지 (유지 중이 아니라)
    fresh_right: bool
    # 화면 · 기록용 진단 값 (판정에는 쓰지 않는다): 들어온 마스크 수, 필터별로 버린 수, 각 쪽을 못 본 시간 등
    diag: dict | None = None
    hold_s: float = HOLD_SEC
    inferred: str = ""           # "left" / "right" — 이번 프레임에 그 쪽을 역산했다

    def display_line(self, b: "Boundary | None") -> np.ndarray | None:
        """이 선을 화면 · HUD 에 그릴 모양. State 2 이거나, 유지 시간이 지났거나, 다듬은 선이 없으면 None."""
        if b is None or self.state == STATE_LOST or not b.drawable(self.hold_s):
            return None
        return b.line
