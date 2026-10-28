# -*- coding: utf-8 -*-
"""차선 스무딩 — 프레임 사이 차선을 트랙 id 로 이어 HUD 흔들림을 줄인다.
"""
from __future__ import annotations

from typing import Any


class LaneSmoother:
    """차선을 프레임 사이에 대응시켜 HUD 흔들림을 줄인다.

    대응은 트랙 id 를 먼저 본다. 젯슨이 id_left / id_right 로 차선 트랙 id 를
    보내므로, 운전자가 차선을 바꿔 id 가 달라지면 화면상 같은 자리에 있어도
    다른 차선이다. 근접도로만 이으면 EMA 가 두 차선을 하나로 이어 붙여
    차선 변경 순간에 없던 급커브가 생긴다. id 가 바뀌면 그 차선의 히스토리를
    버리고 새 차선으로 시작한다.

    id 가 없는 패킷(mock 송신기)은 아래쪽 x 근접도로 잇는다.

    급변 게이트: 한 프레임에 jump_gate_distance 보다 크게 움직이면 그 변화를
    임계값까지만 반영한다. 급정지처럼 차체가 순간적으로 흔들릴 때 선이 튀는
    것을 막는다. 다만 jump_gate_frames 연속으로 계속 크게 움직이면 실제로
    이동한 것이므로 그대로 받아들인다. 아니면 선이 영영 따라오지 못한다.
    """

    def __init__(
        self,
        alpha: float,
        association_distance: float,
        jump_gate_distance: float = 0.0,
        jump_gate_frames: int = 3,
    ) -> None:
        self.alpha = float(alpha)
        self.association_distance = float(association_distance)
        self.jump_gate_distance = float(jump_gate_distance)
        self.jump_gate_frames = int(jump_gate_frames)
        # 트랙 하나 = {"points": ndarray, "id": Any, "hold": int}
        self.tracks: list[dict[str, Any]] = []

    def reset(self) -> None:
        self.tracks = []

    def _associate(self, current: list[tuple[Any, Any]]) -> list[int | None]:
        """이번 프레임 차선마다 이어 붙일 직전 트랙 번호를 정한다."""
        unused = set(range(len(self.tracks)))
        matches: list[int | None] = [None] * len(current)

        # 1) 트랙 id 가 같으면 거리와 무관하게 같은 차선이다.
        for index, (_, lane_id) in enumerate(current):
            if lane_id is None:
                continue
            for track_index in sorted(unused):
                if self.tracks[track_index]["id"] == lane_id:
                    matches[index] = track_index
                    unused.discard(track_index)
                    break

        # 2) 남은 것만 근접도로 잇는다. 양쪽 다 id 가 있는데 1) 에서 안 붙었다면
        #    id 가 바뀐 것이므로 이어서는 안 된다. 한쪽이라도 id 를 모를 때만
        #    근접도 폴백을 허용한다.
        for index, (lane, lane_id) in enumerate(current):
            if matches[index] is not None:
                continue
            best_index = None
            best_distance = float("inf")
            for track_index in unused:
                if lane_id is not None and self.tracks[track_index]["id"] is not None:
                    continue
                distance = abs(
                    float(lane[-1, 0] - self.tracks[track_index]["points"][-1, 0])
                )
                if distance < best_distance:
                    best_distance = distance
                    best_index = track_index
            if best_index is not None and best_distance <= self.association_distance:
                matches[index] = best_index
                unused.discard(best_index)
        return matches

    def update(
        self,
        lanes: list[list[list[float]]],
        ids: list[Any] | None = None,
    ) -> list[list[list[float]]]:
        import numpy as np

        id_list = list(ids) if ids is not None else []
        current = [
            (
                np.asarray(lane, dtype=np.float32),
                id_list[index] if index < len(id_list) else None,
            )
            for index, lane in enumerate(lanes)
        ]
        current.sort(key=lambda item: float(item[0][-1, 0]))
        matches = self._associate(current)

        tracks: list[dict[str, Any]] = []
        for index, (lane, lane_id) in enumerate(current):
            hold = 0
            track_index = matches[index]
            if track_index is not None:
                previous = self.tracks[track_index]["points"]
                previous_x = np.interp(lane[:, 1], previous[:, 1], previous[:, 0])
                blended = self.alpha * lane[:, 0] + (1.0 - self.alpha) * previous_x
                if self.jump_gate_distance > 0.0:
                    delta = blended - previous_x
                    if float(np.max(np.abs(delta))) > self.jump_gate_distance:
                        hold = int(self.tracks[track_index]["hold"]) + 1
                        if hold <= self.jump_gate_frames:
                            blended = previous_x + np.clip(
                                delta,
                                -self.jump_gate_distance,
                                self.jump_gate_distance,
                            )
                        else:
                            # 계속 크게 움직인다면 흔들림이 아니라 실제 이동이다.
                            hold = 0
                lane[:, 0] = blended
            tracks.append({"points": lane, "id": lane_id, "hold": hold})

        self.tracks = tracks
        return [np.round(track["points"], 5).tolist() for track in tracks]
