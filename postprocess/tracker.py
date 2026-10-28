"""자차 차선 추적기 — 모델이 낸 차선 마스크(폴리곤)를 프레임마다 받아, 화면 · HUD 에 그릴 자차 좌 · 우 선을 낸다.

모델은 한 장씩 따로 판단하므로 결과를 그대로 그리면 점선 조각이 번갈아 '가장 가까운 선'이 되고,
와이퍼가 한두 프레임 가리면 선이 깜빡인다. 추적기는 여기에 시간 축을 더한다.

    고르기      소실점 · 기준행 · 거리 제한으로 자차 좌 · 우 선을 고른다 (geometry.py)
    추적        선마다 번호를 붙여 프레임을 넘어 따라간다. 자차 선을 다른 선으로 바꾸려면 0.25 초 동안 계속 그래야 한다
    유지        잠깐 안 보인 선은 0.15 초 동안 마지막 모양을 유지하고, 0.3 초 넘게 안 보이면 State 2 (안 그림)
    다듬기      선 조각을 모아 중심선 하나로 만들고 행마다 평활한다 (smooth.py)
    차로변경    넘어간 선은 번호를 지킨 채 반대쪽 역할로 넘긴다. 한쪽으로 다가가면 그쪽 옆 선을 셋째 선으로 낸다
    이중선      번갈아 보이는 두 획 중 차에 가까운 안쪽 선을 쓴다 (double_line.py)
    한쪽 유실   짝으로 확인된 한 줄은 계속 그리고, 반대쪽 선은 역산한다 (missing_lane.py)

원칙은 "틀리게 그리느니 안 그린다"이다.

    tracker = EgoLaneTracker(30.0)
    out = tracker.update(result.masks.xyn, t=capture_time)   # 0 ~ 1 정규화 폴리곤 목록, 찍힌 시각(초)
    line = out.display_line(out.ego_left)                    # 그릴 선 (N, 2) 또는 None

다른 카메라로 옮길 때는 기준행(소실점에서 화면 아래까지의 35.4 %)과
'경로에 걸친 선은 자차 선이 아니다'(차 가까이에서만 성립)를 먼저 확인한다.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from .config import (
    FORGET_MARGIN_SEC,
    GATE_LANES,
    HOLD_SEC,
    INTENT_OFFSET,
    LANE_W_PRIOR,
    LANE_W_WINDOW_SEC,
    MATCH_GATE,
    MAX_REF_ROW,
    MIN_CROSS_HITS,
    MIN_REF_ROW,
    MIN_SPAN,
    MISS_SEC,
    NEAR_X_MARGIN,
    NEAR_X_MIN_BELOW,
    OBS_REACH_RATIO,
    OFFSET_ALPHA,
    REF_ROW_BELOW_HORIZON,
    ROLE_HYSTERESIS_SEC,
    SAME_LINE_DX,
    STEP_MAX,
    VP_MISS_MAX,
    VP_TOGETHER,
    VP_WARMUP_FRAMES,
    VP_WINDOW_SEC,
    X_PAIR,
)
from .double_line import DoubleLineMixin
from .geometry import (
    MIN_LANE_GAP,
    REF_ROW,
    assign_roles,
    estimate_vanishing_point,
    is_longitudinal,
    out_sign,
    path_x_at_row,
    same_line,
    side_of,
    vp_miss,
    vp_together,
    x_at_row,
)
from .lane import (
    EGO_ROLES,
    STATE_HOLD,
    STATE_LOST,
    STATE_OK,
    Boundary,
    Track,
    TrackerError,
    TrackerOutput,
)
from .missing_lane import MissingLaneMixin
from .smooth import ShapeStabilizer, ema_alpha


class EgoLaneTracker(DoubleLineMixin, MissingLaneMixin):
    """프레임마다 차선 폴리곤을 넣으면 그릴 자차 좌 · 우 선(차로변경 때는 셋째 선까지)이 나온다.

    모델 · 영상과 분리되어 있어 GPU 없이 시험할 수 있고, 젯슨에 그대로 올라간다.
    """

    def __init__(self, fps: float, *, hold_s: float = HOLD_SEC, miss_s: float = MISS_SEC,
                 hysteresis_s: float = ROLE_HYSTERESIS_SEC) -> None:
        # fps: 시간 값을 프레임 수로 바꾸는 기준이자, update() 에 시각을 주지 않을 때의 시계
        if fps <= 0:
            raise TrackerError(f"fps must be positive, got {fps}")
        self.fps = float(fps)
        self.dt = 1.0 / self.fps
        self.hold_s = hold_s
        self.miss_s = miss_s
        self.hyst_frames = max(1, int(round(hysteresis_s * self.fps)))
        self.vp_alpha = 1.0 / (VP_WINDOW_SEC * fps)
        self._shape = ShapeStabilizer()

        self.frame = -1
        self._u = -1.0                    # 추적기 시계 (기준 프레임 단위). 시각을 안 주면 프레임 번호
        self._t0: float | None = None     # 시계 0 에 해당하는 시각
        self.step = 1.0                   # 이번 프레임의 길이 (기준 프레임 수)
        self._step_c = 1.0                # STEP_MAX 로 자른 값 (평활 · 누적용)
        self.vp: tuple[float, float] | None = None
        self._vp_n = 0
        self.ref_row = REF_ROW
        self._center_x = 0.5
        # 차로폭은 카메라마다 다르므로 최근 자차 쌍 간격의 중앙값으로 배운다 (EMA 는 잠깐의 잘못된 쌍에 끌려간다)
        self._lane_w_est: float | None = None
        self._lane_w_hist: deque[float] = deque(maxlen=max(15, int(LANE_W_WINDOW_SEC * fps)))
        self.tracks: list[Track] = []
        self._by_id: dict[int, Track] = {}
        self._next_id = 1
        self.committed: dict[str, int | None] = {"ego_left": None, "ego_right": None}
        self._pending: tuple[int | None, int | None] | None = None
        self._pending_n = 0
        self.miss: dict[str, float] = {"ego_left": 0.0, "ego_right": 0.0}
        self._miss_before = dict(self.miss)
        self._paired: dict[int, tuple[float, float]] = {}   # 선 번호 -> (시계, x): 마지막으로 짝이 확인된 때
        self._pair_lines: dict | None = None   # 마지막으로 확인된 쌍 (역산용)
        self._infer_info: dict = {}            # 역산 진단 값 (기록용)
        self._intent = ""
        self._want = ""
        self._intent_n = 0
        self._offset_ema: float | None = None
        self._active_crossing = ""
        self._settling = False            # 막 선을 넘음 — 새 차로에 자리잡을 때까지 새 의도를 보지 않는다
        self._extra_id: int | None = None  # 셋째 선으로 고른 선 (의도가 시작될 때 고정)
        self.last_drop = {"slope": 0, "reach": 0, "vp": 0, "short": 0}
        self._last_warmup, self._last_gated = False, 0

    @property
    def lane_w(self) -> float | None:
        """배운 차로폭 (기준행, 화면 폭 비율). 아직 못 배웠으면 None."""
        return self._lane_w_est

    # ---------------- 공개 ----------------

    def update(self, polys, t: float | None = None) -> TrackerOutput:
        """한 프레임을 처리한다.

        `t` 는 프레임이 찍힌 시각(초, 기준점은 상관없음). 주지 않으면 직전 프레임 + 1/fps 로 본다.
        라이브 입력은 시각을 줘야 프레임이 건너뛰어도 유지 · State 2 · 평활 시간이 맞는다. 영상 파일은 주지 않는다.
        """
        self.frame += 1
        self._tick(t)
        t = self._u * self.dt
        polys = [np.asarray(p, dtype=np.float32) for p in polys if len(p) >= 3]
        n_in = len(polys)

        # 1. 소실점 · 기준행 · 진행 경로 — 선이 어느 쪽인지가 여기에 달렸다.
        #    소실점 추정의 걸침 검사는 직전 프레임의 기준행으로 한다 (이번 기준행은 소실점을 갱신해야 나온다).
        self._update_vp(estimate_vanishing_point(
            [q for q in polys if is_longitudinal(q, 0.5, self.ref_row)]))
        self.ref_row = self._reference_row()
        center_x = path_x_at_row(self.vp, self.ref_row)

        # 2. 후보 거르기
        drop = {"slope": 0, "reach": 0, "vp": 0, "short": 0}   # 진단용 개수 — 판정에는 쓰지 않는다
        cands: list[tuple[float, np.ndarray]] = []
        refs: list[float] = []            # 기준행에서 읽은 값 (cands 의 x 는 차 가까이에서 읽어 옮긴 값)
        for q in polys:
            # 여기서는 기울기만 본다: 밟고 있는 선은 이번 프레임에 쪽이 없지만 추적 기록으로 쪽을 안다
            if not is_longitudinal(q, center_x, self.ref_row, center_test=False):
                drop["slope"] += 1
                continue
            x = x_at_row(q, self.ref_row, max_reach_ratio=OBS_REACH_RATIO)
            if x is None:
                drop["reach"] += 1
                continue                  # 먼 쪽에만 있는 마스크
            x_ref = float(x)
            x = self._near_equiv(q, float(x), center_x)
            cands.append((float(x), q))
            refs.append(x_ref)
        vp_on = self.vp is not None and self._vp_n >= VP_WARMUP_FRAMES
        misses = [vp_miss(q, self.vp) for _, q in cands] if vp_on else []
        obs: list[tuple[float, np.ndarray]] = []
        for k, (x, q) in enumerate(cands):
            if vp_on:
                miss = misses[k]
                if miss is not None and abs(miss) > VP_MISS_MAX:
                    if vp_together(x, miss, [(cands[j][0], misses[j]) for j in range(len(cands)) if j != k],
                                   VP_TOGETHER, center_x):
                        drop["vp_together_kept"] = drop.get("vp_together_kept", 0) + 1
                    elif self._on_tracked_ego(float(x), refs[k]):
                        drop["vp_kept"] = drop.get("vp_kept", 0) + 1
                    else:
                        drop["vp"] += 1
                        continue          # 차선은 모두 소실점으로 모인다 — 이것은 차선이 아니다
            obs.append((float(x), q))
        if obs:
            # 점선은 짧은 조각 여럿으로 온다. 같은 줄(SAME_LINE_DX 안)의 조각을 합친 길이로 판단한다.
            spans = [(float(q[:, 1].min()), float(q[:, 1].max())) for _, q in obs]
            keep = []
            for i, (x, _) in enumerate(obs):
                lo = min(spans[j][0] for j in range(len(obs)) if abs(obs[j][0] - x) <= SAME_LINE_DX)
                hi = max(spans[j][1] for j in range(len(obs)) if abs(obs[j][0] - x) <= SAME_LINE_DX)
                if hi - lo >= MIN_SPAN:
                    keep.append(obs[i])
            drop["short"] = len(obs) - len(keep)
            obs = keep
        self.last_drop = drop

        # 3. 추적 · 역할 · 차로변경
        self._center_x = center_x
        self._match(obs, center_x)
        roles = self._roles(center_x)
        # 선을 넘는 것은 상태가 아니라 순간이다 — 역할이 바뀌는 동안 매 프레임 알리면 한 번이 여러 번이 된다
        crossing = self._crossing()
        event = crossing if crossing and crossing != self._active_crossing else ""
        self._active_crossing = crossing
        self._commit(roles, event)

        left = self._boundary(self.committed["ego_left"])
        right = self._boundary(self.committed["ego_right"])
        self._update_miss(left, right)
        offset = self._offset(left, right, center_x)
        intent = self._update_intent(offset, event)
        extra = None
        if intent:
            # 셋째 선은 의도가 시작될 때 고른 그 선(번호)이다. 선을 넘는 순간 역할이 한 차로씩 밀려
            # 그때그때의 '옆 선'은 다른 선이 되기 때문이다. 고른 선이 사라지면 지금의 옆 선으로 바꾼다.
            adj = roles.get("adj_left" if intent == "left" else "adj_right")
            if adj is not None:
                e = self._by_id.get(self.committed["ego_left" if intent == "left" else "ego_right"] or -1)
                a = self._by_id.get(adj)
                lw = self._lane_w_est or LANE_W_PRIOR
                if e is None or a is None or not (X_PAIR[0] * lw <= abs(a.x - e.x) <= X_PAIR[1] * lw):
                    adj = None             # 한 차로 옆이 아니면 셋째 선을 내지 않는다
            if self._extra_id is None or self._extra_id not in self._by_id:
                self._extra_id = adj
            if adj is not None and adj == self._extra_id:
                extra = self._boundary(adj)
        else:
            self._extra_id = None

        # 4. 선 다듬기
        lon = [q for q in polys if is_longitudinal(q, center_x, self.ref_row)]
        self._shape.vp_y = None if self.vp is None else float(self.vp[1])
        self._shape.lane_w, self._shape.ref_row = (self._lane_w_est or LANE_W_PRIOR), self.ref_row
        for role, b in (("ego_left", left), ("ego_right", right), ("extra", extra)):
            self._shape.update(role, b, lon, self.hold_s, step=self._step_c)
        # 이번 프레임에 봤지만 다듬은 선을 못 만든 쪽은 못 본 것으로 센다 (같은 규칙으로 State 2)
        undrawn = {"ego_left": False, "ego_right": False}
        for role, b in (("ego_left", left), ("ego_right", right)):
            if b is not None and b.age_s == 0.0 and b.line is None:
                self.miss[role] = self._miss_before[role] + self.step * self.dt
                undrawn[role] = True

        self._note_pair(left, right)

        # 5. State 판정 · 한쪽 역산 (역산 선은 출력에만 들어가고 추적 · State 에는 쓰지 않는다)
        state = self._state()
        fresh_left, fresh_right = bool(left and left.age_s == 0.0), bool(right and right.age_s == 0.0)
        inferred = ""
        self._infer_info = {}
        if state != STATE_LOST:
            left, right, inferred = self._infer_other_side(left, right)
        return TrackerOutput(
            frame=self.frame, time_s=t, ego_left=left, ego_right=right,
            extra=extra, state=state, event=event, intent=intent,
            center_x=center_x, ref_row=self.ref_row, offset=offset,
            observed=len(obs),
            fresh_left=fresh_left,
            fresh_right=fresh_right,
            diag={"masks": n_in, **self.last_drop, "gate": self._last_gated,
                  "warmup": self._last_warmup, "miss_left": self.miss["ego_left"],
                  "miss_right": self.miss["ego_right"], "miss_limit": self.miss_s,
                  "undrawn_left": undrawn["ego_left"], "undrawn_right": undrawn["ego_right"], **self._infer_info},
            hold_s=self.hold_s, inferred=inferred,
        )

    # ---------------- 시계 · 소실점 · 기준행 ----------------

    def _tick(self, t: float | None) -> None:
        """시계를 이번 프레임으로 옮기고 step · _step_c 를 정한다."""
        prev = self._u
        if t is None:
            u = prev + 1.0
        else:
            if self._t0 is None:
                self._t0 = float(t) - (prev + 1.0) * self.dt   # 첫 시각은 직전 프레임 + 1 프레임으로 둔다
            # 소수점 여섯째 자리에서 반올림 — 시각이 정확히 k/fps 이면 시계가 k 가 되어, 시각 없이 돌린 결과와 같아진다
            u = max(prev, round((float(t) - self._t0) * self.fps, 6))   # 시계는 거꾸로 가지 않는다
        self._u = u
        self.step = u - prev
        self._step_c = min(self.step, STEP_MAX)

    def _update_vp(self, vp: tuple[float, float] | None) -> None:
        if vp is None:
            return                        # 마지막 값 유지
        if self.vp is None:
            self.vp = vp
            self._vp_n = 1
            return
        # 카메라가 차에 고정되어 있으므로 소실점은 거의 상수다. 처음에는 전체 평균(1/n)으로, 이후는 VP_WINDOW_SEC
        # 시간 상수로 천천히 따라간다 (빨리 따라가면 차로변경 중 차체가 돌 때 소실점이 흔들린다).
        self._vp_n += 1
        a = max(ema_alpha(self.vp_alpha, self._step_c), 1.0 / self._vp_n)
        self.vp = (self.vp[0] + a * (vp[0] - self.vp[0]),
                   self.vp[1] + a * (vp[1] - self.vp[1]))

    def _reference_row(self) -> float:
        """좌우를 판단하는 행 — 소실점에서 화면 아래까지의 REF_ROW_BELOW_HORIZON 지점 (카메라와 상관없이 앞쪽 일정 거리)."""
        if self.vp is None:
            return REF_ROW
        vy = self.vp[1]
        if not 0.0 <= vy < 0.95:
            return REF_ROW
        row = vy + REF_ROW_BELOW_HORIZON * (1.0 - vy)
        return float(min(MAX_REF_ROW, max(MIN_REF_ROW, row)))

    # ---------------- 위치 읽기 ----------------

    def _near_equiv(self, q: np.ndarray, x_ref: float, center_x: float) -> float:
        """마스크 아래 끝 바로 위에서 읽은 경로까지의 거리를, 곧은 도로처럼 기준행으로 옮긴 위치.
        마스크가 기준행 아래로 내려오지 않으면 기준행에서 읽은 값을 쓴다."""
        if self.vp is None:
            return x_ref
        vy = float(self.vp[1])
        y = float(q[:, 1].max()) - NEAR_X_MARGIN
        if y < self.ref_row + NEAR_X_MIN_BELOW or y - vy < 1e-3:
            return x_ref
        xn = x_at_row(q, y, extrapolate=False)
        if xn is None:
            return x_ref
        return float(center_x + (xn - path_x_at_row(self.vp, y)) * (self.ref_row - vy) / (y - vy))

    def _on_tracked_ego(self, x: float, x_ref: float) -> bool:
        """이 위치에 유지 시간 안에 본 자차 선이 있는지 — 급커브에서 크게 휜 자차 선을 소실점 검사로 버리지 않기 위해.
        기준행에서도 그 선과 맞아야 한다 (사선 표시가 우연히 자차 선 근처에 놓이는 경우를 막는다)."""
        for role in EGO_ROLES:
            tr = self._by_id.get(self.committed[role]) if self.committed[role] is not None else None
            if (tr is not None and abs(tr.x - x) <= MATCH_GATE
                    and (self._u - tr.last_seen) * self.dt <= self.hold_s):
                xt = x_at_row(tr.poly, self.ref_row, max_reach_ratio=OBS_REACH_RATIO)
                if xt is None or abs(xt - x_ref) > MATCH_GATE:
                    continue
                return True
        return False

    # ---------------- 추적 ----------------

    def _match(self, obs: list[tuple[float, np.ndarray]], center_x: float) -> None:
        """이번 프레임의 선을 기존 추적에 잇는다 — 가까운 쌍부터 (후보가 적고 차로폭만큼 떨어져 있어 이 방식으로 충분하다)."""
        pairs: list[tuple[float, int, int]] = []
        for oi, (x, _) in enumerate(obs):
            for tr in self.tracks:
                d = abs(x - tr.x)
                gap = self._stroke_gap(tr)
                if gap is not None:
                    d = min(d, abs(x - (tr.x + out_sign(tr.x, center_x) * gap)))   # 이중선은 바깥 획 위치도 본다
                if d <= MATCH_GATE:
                    pairs.append((d, oi, tr.tid))
        pairs.sort()
        used_obs: set[int] = set()
        used_tr: set[int] = set()
        for _, oi, tid in pairs:
            if oi in used_obs or tid in used_tr:
                continue
            used_obs.add(oi)
            used_tr.add(tid)
            x, q = obs[oi]
            tr = self._by_id[tid]
            prev = tr.x
            tr.observe(x, q, self._u, side_of(x, center_x))
            self._stroke_observe(tr, prev, center_x)

        for oi, (x, q) in enumerate(obs):
            if oi in used_obs:
                continue
            side = side_of(x, center_x)
            tr = Track(tid=self._next_id, x=x, poly=q, side=side,
                       confident_side=side, first_seen=self._u,
                       last_seen=self._u)
            self._next_id += 1
            self.tracks.append(tr)
            self._by_id[tr.tid] = tr

        self._merge_duplicates()

        forget = max(self.hold_s, self.miss_s) + FORGET_MARGIN_SEC
        alive = [tr for tr in self.tracks
                 if (self._u - tr.last_seen) * self.dt <= forget]
        self.tracks = alive
        self._by_id = {tr.tid: tr for tr in alive}

    def _merge_duplicates(self) -> None:
        """MIN_LANE_GAP 보다 가까운 두 추적은 한 도색으로 합친다.

        모델은 한 선을 겹친 두 마스크로, 점선을 여러 조각으로 내는 일이 많다. 각자 번호를 가지면
        자차 역할이 둘 사이를 오가며, 선이 보이는데도 유지 상태가 된다.
        """
        groups: list[list[Track]] = []
        for tr in sorted(self.tracks, key=lambda t: t.x):
            prev = groups[-1][-1] if groups else None
            if prev is not None and abs(tr.x - prev.x) < MIN_LANE_GAP:
                groups[-1].append(tr)
            else:
                groups.append([tr])
        if all(len(g) == 1 for g in groups):
            return

        def quality(t: Track) -> tuple[int, float, float]:
            """가장 최근에 본 것, 같으면 경로에 가까운 것 (이중선의 안쪽), 그다음 긴 것."""
            return (t.last_seen, -abs(t.x - self._center_x),
                    float(t.poly[:, 1].max() - t.poly[:, 1].min()))

        survivors: list[Track] = []
        remap: dict[int, int] = {}
        for g in groups:
            keep = min(g, key=lambda t: t.tid)      # 가장 오래 따라간 번호를 남긴다
            fresh = max(g, key=quality)
            if keep.last_seen == self._u and len({t.confident_side for t in g}) > 1:
                # 경로 양쪽에 걸친 무리는 가장 오래 따라간 선이 자기 값을 지킨다
                # ('경로에 가까운 것'을 고르면 선 옆의 빗물 줄기가 그 자리를 차지한다)
                fresh = keep
            if fresh is not keep:
                prev = keep.x
                keep.observe(fresh.x, fresh.poly, fresh.last_seen, fresh.side)
                if fresh.last_seen == self._u:
                    self._stroke_observe(keep, prev, self._center_x)   # 이중선의 두 획이 MATCH_GATE 보다 멀 때 여기로 온다
            keep.hits = max(t.hits for t in g)
            for t in g:
                remap[t.tid] = keep.tid
            survivors.append(keep)
        self.tracks = survivors
        self._by_id = {t.tid: t for t in survivors}
        for role in EGO_ROLES:
            tid = self.committed[role]
            if tid is not None:
                self.committed[role] = remap.get(tid, tid)

    def _roles(self, center_x: float) -> dict[str, int]:
        """원래 마스크가 아니라 추적 중인 선들로 역할을 고른다 — 유지 시간 안의 선은 이번 프레임에 안 보여도 참여한다
        (그렇지 않으면 잠깐 사라진 선의 역할을 바로 옆 선이 가져간다)."""
        self._last_warmup, self._last_gated = False, 0
        if self.vp is not None and 0 < self._vp_n < VP_WARMUP_FRAMES:
            self._last_warmup = True
            return {}                     # 경로를 아직 믿을 수 없으므로 아무것도 고르지 않는다
        avail = [tr for tr in self.tracks
                 if (self._u - tr.last_seen) * self.dt <= self.hold_s]
        lw = self._lane_w_est or LANE_W_PRIOR
        gate = GATE_LANES * lw
        self._last_gated = sum(1 for tr in avail if abs(tr.x - center_x) > gate)
        # 처음부터 경로에 걸친 선은 쪽을 정할 기록이 없다. 차가 그 선을 넘는 중이며, 그 선은 비어 있는 쪽의 경계다.
        sides = []
        for tr in avail:
            geo = side_of(tr.x, center_x)
            side = geo if geo != 0 else tr.confident_side   # 걸쳐 있을 때만 기록을 쓴다
            if side == 0:
                have_l = self.committed["ego_left"] not in (None, tr.tid)
                have_r = self.committed["ego_right"] not in (None, tr.tid)
                if have_l and not have_r:
                    side = 1
                elif have_r and not have_l:
                    side = -1
            sides.append(side)
        roles = assign_roles([tr.poly for tr in avail], [tr.x for tr in avail], sides, center_x, self.ref_row,
                             max_dist=gate, pair_w=lw)
        out = {role: avail[i].tid for role, i in roles.items()}
        if "ego_left" in out and "ego_right" in out:
            # 크게 휜 선 하나의 두 조각이 양쪽에 앉을 수 있다. 한 선이 두 자차 선이 될 수는 없으므로 새 추적이 역할을 내놓는다.
            a, b = self._by_id[out["ego_left"]], self._by_id[out["ego_right"]]
            if same_line(a.poly, b.poly):
                out.pop("ego_left" if a.hits < b.hits else "ego_right")
        if "ego_left" in out and "ego_right" in out:
            l, r = self._by_id[out["ego_left"]], self._by_id[out["ego_right"]]
            w = abs(r.x - l.x)
            # 둘 다 이번 프레임에 본 쌍이면 간격을 모두 배운다 (범위로 걸러 배우면 처음 잘못 배운 값이 진짜 차로폭을 막는다)
            if 0.05 < w < 0.6 and l.last_seen == self._u and r.last_seen == self._u:
                self._lane_w_hist.append(w)
                if len(self._lane_w_hist) >= 15:
                    self._lane_w_est = float(np.median(self._lane_w_hist))
        return out

    def _crossing(self) -> str:
        """그리던 선이 차 밑을 지나 반대쪽으로 갔는지 — 차로변경. 선은 번호를 지킨 채 쪽을 바꾼다."""
        if self._u < self.hyst_frames:
            return ""            # 아직 도로를 찾는 중
        left = self._by_id.get(self.committed["ego_left"] or -1)
        if left is not None and left.confident_side > 0 and left.hits >= MIN_CROSS_HITS:
            return "change_left"
        right = self._by_id.get(self.committed["ego_right"] or -1)
        if right is not None and right.confident_side < 0 and right.hits >= MIN_CROSS_HITS:
            return "change_right"
        return ""

    def _commit(self, roles: dict[str, int], event: str) -> None:
        """고른 역할을 확정한다. 보이던 선을 다른 선으로 바꿀 때만 hyst_frames 동안 기다린다."""
        cand = {r: roles.get(r) for r in EGO_ROLES}
        key = (cand["ego_left"], cand["ego_right"])
        if key == (self.committed["ego_left"], self.committed["ego_right"]):
            self._pending, self._pending_n = None, 0
            return
        # 선이 새로 생기거나 사라지는 것, 선을 넘는 것은 바로 반영한다 (사라지는 선은 이미 유지 시간을 다 썼다)
        replacement = any(self.committed[r] is not None and cand[r] is not None
                          and cand[r] != self.committed[r] for r in EGO_ROLES)
        if not replacement or event:
            self.committed.update(cand)
            self._pending, self._pending_n = None, 0
            return
        # 같은 후보가 이어진 시간을 기준 프레임으로 센다 (첫 프레임은 1)
        if key == self._pending:
            self._pending_n += self._step_c
        else:
            self._pending, self._pending_n = key, 1.0
        if self._pending_n >= self.hyst_frames - 1e-9:
            self.committed.update(cand)
            self._pending, self._pending_n = None, 0

    def _boundary(self, tid: int | None) -> Boundary | None:
        tr = self._by_id.get(tid) if tid is not None else None
        if tr is None:
            return None
        age = (self._u - tr.last_seen) * self.dt
        return Boundary(track_id=tr.tid, poly=tr.poly, x=tr.x,
                        age_s=age, held=age > 0.0, from_outer=tr.from_outer)

    def _update_miss(self, left: Boundary | None, right: Boundary | None) -> None:
        """좌 · 우를 각각 몇 초째 못 봤는지 센다."""
        self._miss_before = dict(self.miss)      # 다듬은 선을 못 만든 쪽은 여기서부터 이어 센다
        for role, b in (("ego_left", left), ("ego_right", right)):
            if b is not None and b.age_s == 0.0:
                self.miss[role] = 0.0
            else:
                self.miss[role] += self.step * self.dt     # 처리가 밀린 시간도 실제 시간대로 센다

    def _offset(self, left: Boundary | None, right: Boundary | None,
                center_x: float) -> float | None:
        """차로 안 위치: 왼쪽 선 -1 ~ 가운데 0 ~ 오른쪽 선 +1."""
        if left is None or right is None:
            return None
        d_l = center_x - left.x
        d_r = right.x - center_x
        total = d_l + d_r
        if total <= 1e-6:
            return None
        return float((d_l - d_r) / total)

    def _update_intent(self, offset: float | None, event: str) -> str:
        """차로변경 의도 — 차로 안 위치가 한쪽 끝으로 가고 있으면 그쪽."""
        if event:
            # 방금 선을 넘었다. 새 차로에 자리잡을 때까지 새 의도는 보지 않는다
            # (방금 넘은 선 근처에 있는 것을 되돌아가려는 것으로 읽지 않게).
            self._intent, self._want, self._intent_n = "", "", 0
            self._offset_ema = offset
            self._settling = True
            return ""
        a_off = ema_alpha(OFFSET_ALPHA, self._step_c)
        if self._settling:
            if offset is not None:
                if self._offset_ema is None:
                    self._offset_ema = offset
                else:
                    self._offset_ema += a_off * (offset - self._offset_ema)
                if abs(offset) < INTENT_OFFSET:
                    self._settling = False
            return ""
        # 위치만으로는 의도가 아니다 — 문턱 밖이면서 가운데로 돌아오는 중이 아니어야 한다 (EMA = 머물던 위치)
        want = ""
        if offset is not None:
            if self._offset_ema is None:
                self._offset_ema = offset
            else:
                self._offset_ema += a_off * (offset - self._offset_ema)
            trend = offset - self._offset_ema
            if offset <= -INTENT_OFFSET and trend <= 0.0:
                want = "left"
            elif offset >= INTENT_OFFSET and trend >= 0.0:
                want = "right"
        # 역할과 같은 지연: 차로 안에서 흔들리는 것으로 셋째 선이 켜졌다 꺼지지 않게
        if want == self._intent:
            self._want, self._intent_n = want, 0
            return self._intent
        if want == self._want:
            self._intent_n += self._step_c
        else:
            self._want, self._intent_n = want, 1.0
        if self._intent_n >= self.hyst_frames - 1e-9:
            self._intent, self._intent_n = want, 0
        return self._intent

    # ---------------- 상태 ----------------

    def _state(self) -> int:
        """State 판정. 확신 없는 선을 그리느니 안 그린다 — 한쪽을 잃으면 둘 다 끄되, 짝으로 확인된 한 줄은 남긴다."""
        if max(self.miss.values()) > self.miss_s:
            if self._one_side_ok():
                return STATE_HOLD
            return STATE_LOST
        if max(self.miss.values()) > 0.0:
            return STATE_HOLD
        return STATE_OK
