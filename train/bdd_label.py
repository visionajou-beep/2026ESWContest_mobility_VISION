"""BDD100K 이미지별 차선 라벨(JSON) 읽기 — 도색 차선 하나를 닫힌 폴리곤 하나로 만든다.

BDD100K 차선 라벨은 도색 하나를 열린 선(``poly2d``) 한두 개로 그린다. 점은 ``[x, y, 명령]`` 이고,
직선 구간은 ``L``, 3 차 베지어 곡선은 ``CCC`` 뒤에 끝점 ``L`` 이 오는 식이다. 이 모듈은 그 경로를 촘촘한 선으로
복원하고, 한 도색의 양쪽 가장자리 두 선을 짝지어 그 사이를 칠한 폴리곤(원본 1280x720 좌표)으로 만든다.

진행 방향(``direction=parallel``)의 흰색 · 노란색 차선만 쓴다. 횡단보도 · 연석 · 가로 방향 선 · 색을 알 수 없는 선은 뺀다.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


__all__ = [
    "LaneBoundary",
    "LaneInstance",
    "ParseError",
    "ParseResult",
    "ParseWarning",
    "ParserConfig",
    "parse_bdd_annotation",
    "parse_bdd_lane_annotation",
    "parse_bdd_lane_file",
    "parse_bdd_lanes",
    "poly2d_to_polyline",
]


@dataclass(frozen=True)
class ParserConfig:
    """1280x720 BDD100K 라벨용 해석 · 기하 기준값."""

    image_width: int = 1280
    image_height: int = 720
    sample_spacing_px: float = 2.0
    min_cubic_samples: int = 8
    max_segment_samples: int = 512
    max_boundary_samples: int = 8192
    max_poly2d_points: int = 4096
    max_objects: int = 10_000
    max_json_bytes: int = 32 * 1024 * 1024

    # 한 도색의 두 가장자리는 대개 24 px 보다 훨씬 가깝다. 24 ~ 27 px 구간은 차 가까이에서 원근으로 넓어진
    # 도색을 위한 것으로, 라벨에서 두 선이 이웃해 있을 때만 쓴다 (_pair_candidate) — 관계없는 평행선이 섞이지 않게.
    pair_profile_samples: int = 48
    pair_min_y_span_px: float = 24.0
    pair_min_overlap_ratio: float = 0.70
    pair_min_span_ratio: float = 0.60
    pair_min_median_distance_px: float = 0.75
    pair_core_max_median_distance_px: float = 24.0
    pair_max_median_distance_px: float = 27.0
    pair_wide_requires_adjacent_objects: bool = True
    pair_max_p90_distance_px: float = 44.0
    pair_min_tangent_cosine: float = 0.94
    pair_min_side_consistency: float = 0.90
    pair_ambiguity_ratio: float = 1.30
    pair_ambiguity_margin_px: float = 2.0

    standalone_width_px: float = 8.0
    max_polygon_side_samples: int = 256
    min_polygon_area_px2: float = 3.0
    strict: bool = False

    def __post_init__(self) -> None:
        if self.image_width < 2 or self.image_height < 2:
            raise ValueError("image dimensions must both be at least 2 pixels")
        if self.sample_spacing_px <= 0:
            raise ValueError("sample_spacing_px must be positive")
        if self.min_cubic_samples < 1 or self.max_segment_samples < 2:
            raise ValueError("invalid curve sampling limits")
        if self.max_boundary_samples < 2 or self.max_poly2d_points < 2:
            raise ValueError("invalid polyline limits")
        if not 0 < self.pair_min_overlap_ratio <= 1:
            raise ValueError("pair_min_overlap_ratio must be in (0, 1]")
        if not 0 < self.pair_min_span_ratio <= 1:
            raise ValueError("pair_min_span_ratio must be in (0, 1]")
        if self.pair_max_median_distance_px <= self.pair_min_median_distance_px:
            raise ValueError("pair distance limits are inconsistent")
        if not (
            self.pair_min_median_distance_px
            < self.pair_core_max_median_distance_px
            <= self.pair_max_median_distance_px
        ):
            raise ValueError("pair core distance limits are inconsistent")
        if self.standalone_width_px <= 0:
            raise ValueError("standalone_width_px must be positive")


@dataclass(frozen=True)
class ParseWarning:
    """이미지 하나를 읽다 발견한 문제 (치명적이지 않음, 보고서용)."""

    code: str
    message: str
    object_ids: tuple[Any, ...] = ()
    details: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "object_ids": list(self.object_ids),
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class LaneBoundary:
    """골라서 촘촘하게 복원한 차선 가장자리 하나."""

    source_id: Any
    object_index: int
    category: str
    color: str
    style: str
    direction: str
    points: np.ndarray

    @property
    def polyline(self) -> np.ndarray:
        return self.points


@dataclass(frozen=True)
class LaneInstance:
    """닫힌 폴리곤으로 나타낸 도색 차선 하나."""

    category: str
    color: str
    style: str
    polygon: np.ndarray
    centerline: np.ndarray
    source_ids: tuple[Any, ...]
    source_indices: tuple[int, ...]
    paired: bool

    @property
    def class_name(self) -> str:
        return "lane_boundary"

    @property
    def points(self) -> np.ndarray:
        """폴리곤 점을 기대하는 쪽을 위한 다른 이름."""

        return self.polygon


@dataclass(frozen=True)
class ParseResult:
    """BDD100K 라벨 파일 하나의 결과 — 차선 목록과 점검 정보."""

    image_name: str
    image_size: tuple[int, int]
    boundaries: tuple[LaneBoundary, ...]
    lanes: tuple[LaneInstance, ...]
    warnings: tuple[ParseWarning, ...]
    stats: Mapping[str, Any]

    @property
    def instances(self) -> tuple[LaneInstance, ...]:
        return self.lanes

    @property
    def ok(self) -> bool:
        fatal_codes = {
            "file_not_found",
            "file_too_large",
            "invalid_json",
            "invalid_root",
            "missing_frames",
            "invalid_objects",
            "paired_polygon_unresolved",
        }
        return not any(warning.code in fatal_codes for warning in self.warnings)


class ParseError(ValueError):
    """``ParserConfig.strict`` 일 때 잘못된 입력에서 발생한다."""


@dataclass(frozen=True)
class _PairCandidate:
    left: int
    right: int
    score: float
    median_distance: float
    p90_distance: float
    overlap_ratio: float
    tangent_cosine: float


def _warning(
    sink: list[ParseWarning],
    config: ParserConfig,
    code: str,
    message: str,
    *,
    object_ids: Sequence[Any] = (),
    details: Mapping[str, Any] | None = None,
    fatal: bool = False,
) -> None:
    warning = ParseWarning(
        code=code,
        message=message,
        object_ids=tuple(object_ids),
        details=dict(details or {}),
    )
    sink.append(warning)
    if fatal and config.strict:
        raise ParseError(f"{code}: {message}")


def _empty_result(
    config: ParserConfig,
    warnings: Sequence[ParseWarning],
    *,
    image_name: str = "",
    stats: Mapping[str, Any] | None = None,
) -> ParseResult:
    return ParseResult(
        image_name=image_name,
        image_size=(config.image_width, config.image_height),
        boundaries=(),
        lanes=(),
        warnings=tuple(warnings),
        stats=dict(stats or {}),
    )


def _normalise_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.strip().casefold().split())


def _safe_source_id(value: Any, fallback: int) -> Any:
    if value is None:
        return fallback
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _append_without_duplicate(parts: list[np.ndarray], samples: np.ndarray) -> None:
    if samples.size == 0:
        return
    if parts and np.allclose(parts[-1][-1], samples[0], atol=1e-9, rtol=0):
        samples = samples[1:]
    if samples.size:
        parts.append(samples)


def _sample_line(
    start: np.ndarray, end: np.ndarray, config: ParserConfig
) -> np.ndarray:
    length = float(np.linalg.norm(end - start))
    intervals = max(1, int(math.ceil(length / config.sample_spacing_px)))
    intervals = min(intervals, config.max_segment_samples)
    t = np.linspace(0.0, 1.0, intervals + 1, dtype=np.float64)[:, None]
    return start[None, :] * (1.0 - t) + end[None, :] * t


def _sample_cubic(
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    config: ParserConfig,
) -> np.ndarray:
    control_length = float(
        np.linalg.norm(p1 - p0)
        + np.linalg.norm(p2 - p1)
        + np.linalg.norm(p3 - p2)
    )
    intervals = max(
        config.min_cubic_samples,
        int(math.ceil(control_length / config.sample_spacing_px)),
    )
    intervals = min(intervals, config.max_segment_samples)
    t = np.linspace(0.0, 1.0, intervals + 1, dtype=np.float64)[:, None]
    one_minus_t = 1.0 - t
    return (
        one_minus_t**3 * p0[None, :]
        + 3.0 * one_minus_t**2 * t * p1[None, :]
        + 3.0 * one_minus_t * t**2 * p2[None, :]
        + t**3 * p3[None, :]
    )


def _dedupe_consecutive(points: np.ndarray, tolerance: float = 1e-7) -> np.ndarray:
    if len(points) < 2:
        return points
    keep = np.ones(len(points), dtype=bool)
    keep[1:] = np.linalg.norm(np.diff(points, axis=0), axis=1) > tolerance
    return points[keep]


def _downsample_preserving_ends(points: np.ndarray, limit: int) -> np.ndarray:
    if len(points) <= limit:
        return points
    indices = np.linspace(0, len(points) - 1, limit)
    indices = np.unique(np.rint(indices).astype(np.int64))
    return points[indices]


def poly2d_to_polyline(
    poly2d: Any,
    *,
    config: ParserConfig | None = None,
    warnings: list[ParseWarning] | None = None,
    object_id: Any = None,
) -> np.ndarray:
    """``poly2d`` 경로를 화면 안으로 자른 촘촘한 선으로 복원한다.

    잘못된 경로는 빈 ``(0, 2)`` 배열을 돌려주고, 경고 목록이 주어지면 경고를 남긴다.
    """

    cfg = config or ParserConfig()
    sink = warnings if warnings is not None else []
    ids = () if object_id is None else (object_id,)

    if (
        not isinstance(poly2d, Sequence)
        or isinstance(poly2d, (str, bytes, bytearray))
        or len(poly2d) < 2
    ):
        _warning(
            sink,
            cfg,
            "invalid_poly2d",
            "poly2d must contain at least two [x, y, command] points",
            object_ids=ids,
        )
        return np.empty((0, 2), dtype=np.float64)
    if len(poly2d) > cfg.max_poly2d_points:
        _warning(
            sink,
            cfg,
            "poly2d_too_large",
            "poly2d exceeds the configured point limit",
            object_ids=ids,
            details={"count": len(poly2d), "limit": cfg.max_poly2d_points},
        )
        return np.empty((0, 2), dtype=np.float64)

    coordinates: list[tuple[float, float]] = []
    commands: list[str] = []
    for point_index, raw_point in enumerate(poly2d):
        if (
            not isinstance(raw_point, Sequence)
            or isinstance(raw_point, (str, bytes, bytearray))
            or len(raw_point) < 3
            or isinstance(raw_point[0], bool)
            or isinstance(raw_point[1], bool)
        ):
            _warning(
                sink,
                cfg,
                "invalid_poly2d_point",
                "poly2d point is not [finite x, finite y, L|C]",
                object_ids=ids,
                details={"point_index": point_index},
            )
            return np.empty((0, 2), dtype=np.float64)
        try:
            x = float(raw_point[0])
            y = float(raw_point[1])
        except (TypeError, ValueError, OverflowError):
            _warning(
                sink,
                cfg,
                "invalid_poly2d_point",
                "poly2d coordinate is not numeric",
                object_ids=ids,
                details={"point_index": point_index},
            )
            return np.empty((0, 2), dtype=np.float64)
        if not math.isfinite(x) or not math.isfinite(y):
            _warning(
                sink,
                cfg,
                "non_finite_coordinate",
                "poly2d contains NaN or infinity",
                object_ids=ids,
                details={"point_index": point_index},
            )
            return np.empty((0, 2), dtype=np.float64)
        command = _normalise_text(raw_point[2]).upper()
        if command not in {"C", "L"}:
            _warning(
                sink,
                cfg,
                "unsupported_poly2d_command",
                "only legacy C and L commands are supported",
                object_ids=ids,
                details={"point_index": point_index, "command": raw_point[2]},
            )
            return np.empty((0, 2), dtype=np.float64)
        coordinates.append((x, y))
        commands.append(command)

    control = np.asarray(coordinates, dtype=np.float64)

    # 일부만 만들기 전에 곡선 구간을 모두 검사한다. CCCL 은 3 차 곡선 하나, CCCCCCL 은 네 번째 점을 공유하는 곡선 둘이다.
    index = 0
    while index < len(commands):
        if commands[index] != "C":
            index += 1
            continue
        end = index
        while end < len(commands) and commands[end] == "C":
            end += 1
        run_length = end - index
        if end >= len(commands) or run_length % 3 != 0:
            _warning(
                sink,
                cfg,
                "malformed_curve_commands",
                "a C run must contain 3*n points followed by an L endpoint",
                object_ids=ids,
                details={"start_index": index, "run_length": run_length},
            )
            return np.empty((0, 2), dtype=np.float64)
        index = end + 1

    parts: list[np.ndarray] = []
    index = 0
    while index < len(control) - 1:
        if commands[index] == "L":
            _append_without_duplicate(
                parts, _sample_line(control[index], control[index + 1], cfg)
            )
            index += 1
            continue

        run_end = index
        while run_end < len(commands) and commands[run_end] == "C":
            run_end += 1
        for cubic_start in range(index, run_end, 3):
            _append_without_duplicate(
                parts,
                _sample_cubic(
                    control[cubic_start],
                    control[cubic_start + 1],
                    control[cubic_start + 2],
                    control[cubic_start + 3],
                    cfg,
                ),
            )
        index = run_end

    if not parts:
        _warning(
            sink,
            cfg,
            "degenerate_poly2d",
            "poly2d did not produce a usable segment",
            object_ids=ids,
        )
        return np.empty((0, 2), dtype=np.float64)

    points = _dedupe_consecutive(np.vstack(parts))
    control_outside = (
        (control[:, 0] < 0)
        | (control[:, 0] > cfg.image_width - 1)
        | (control[:, 1] < 0)
        | (control[:, 1] > cfg.image_height - 1)
    )
    sampled_outside = (
        (points[:, 0] < 0)
        | (points[:, 0] > cfg.image_width - 1)
        | (points[:, 1] < 0)
        | (points[:, 1] > cfg.image_height - 1)
    )
    if np.any(control_outside) or np.any(sampled_outside):
        original_bounds = {
            "min_x": float(np.min(control[:, 0])),
            "max_x": float(np.max(control[:, 0])),
            "min_y": float(np.min(control[:, 1])),
            "max_y": float(np.max(control[:, 1])),
        }
        _warning(
            sink,
            cfg,
            "coordinates_clipped",
            "lane coordinates were clipped to the 1280x720 image",
            object_ids=ids,
            details={
                "control_points_outside": int(np.count_nonzero(control_outside)),
                "sampled_points_outside": int(np.count_nonzero(sampled_outside)),
                "original_bounds": original_bounds,
            },
        )

    points[:, 0] = np.clip(points[:, 0], 0.0, cfg.image_width - 1.0)
    points[:, 1] = np.clip(points[:, 1], 0.0, cfg.image_height - 1.0)
    points = _dedupe_consecutive(points)
    if len(points) > cfg.max_boundary_samples:
        original_count = len(points)
        points = _downsample_preserving_ends(points, cfg.max_boundary_samples)
        _warning(
            sink,
            cfg,
            "boundary_sample_limit",
            "dense boundary was deterministically downsampled",
            object_ids=ids,
            details={"original_count": original_count, "kept": len(points)},
        )
    if len(points) < 2 or float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))) < 1.0:
        _warning(
            sink,
            cfg,
            "degenerate_poly2d",
            "clipped poly2d is shorter than one pixel",
            object_ids=ids,
        )
        return np.empty((0, 2), dtype=np.float64)
    return points


def _monotonic_y_fraction(points: np.ndarray) -> float:
    differences = np.diff(points[:, 1])
    differences = differences[np.abs(differences) > 1e-6]
    if not len(differences):
        return 0.0
    positive = float(np.count_nonzero(differences > 0)) / len(differences)
    return max(positive, 1.0 - positive)


def _unique_y_profile(points: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    order = np.argsort(points[:, 1], kind="stable")
    ys = points[order, 1]
    xs = points[order, 0]
    unique_y, inverse = np.unique(ys, return_inverse=True)
    if len(unique_y) < 2:
        return None
    sums = np.zeros(len(unique_y), dtype=np.float64)
    counts = np.zeros(len(unique_y), dtype=np.float64)
    np.add.at(sums, inverse, xs)
    np.add.at(counts, inverse, 1.0)
    return unique_y, sums / counts


def _pair_candidate(
    left_index: int,
    right_index: int,
    left: LaneBoundary,
    right: LaneBoundary,
    config: ParserConfig,
) -> _PairCandidate | None:
    if left.category != right.category or left.color != right.color or left.style != right.style:
        return None
    if left.style == "unknown":
        return None
    if _monotonic_y_fraction(left.points) < 0.85 or _monotonic_y_fraction(right.points) < 0.85:
        return None

    left_profile = _unique_y_profile(left.points)
    right_profile = _unique_y_profile(right.points)
    if left_profile is None or right_profile is None:
        return None
    left_y, left_x = left_profile
    right_y, right_x = right_profile
    left_span = float(left_y[-1] - left_y[0])
    right_span = float(right_y[-1] - right_y[0])
    if min(left_span, right_span) < config.pair_min_y_span_px:
        return None
    span_ratio = min(left_span, right_span) / max(left_span, right_span)
    if span_ratio < config.pair_min_span_ratio:
        return None

    overlap_start = max(float(left_y[0]), float(right_y[0]))
    overlap_end = min(float(left_y[-1]), float(right_y[-1]))
    overlap = overlap_end - overlap_start
    if overlap <= 0:
        return None
    overlap_ratio = overlap / min(left_span, right_span)
    if overlap_ratio < config.pair_min_overlap_ratio:
        return None

    sample_y = np.linspace(
        overlap_start,
        overlap_end,
        config.pair_profile_samples,
        dtype=np.float64,
    )
    sample_left_x = np.interp(sample_y, left_y, left_x)
    sample_right_x = np.interp(sample_y, right_y, right_x)
    signed_separation = sample_right_x - sample_left_x
    nonzero = np.abs(signed_separation) > 0.25
    if not np.any(nonzero):
        return None
    positive_fraction = float(np.count_nonzero(signed_separation[nonzero] > 0)) / int(
        np.count_nonzero(nonzero)
    )
    side_consistency = max(positive_fraction, 1.0 - positive_fraction)
    if side_consistency < config.pair_min_side_consistency:
        return None

    left_slope = np.gradient(sample_left_x, sample_y)
    right_slope = np.gradient(sample_right_x, sample_y)
    tangent_cosines = (left_slope * right_slope + 1.0) / (
        np.sqrt(left_slope**2 + 1.0) * np.sqrt(right_slope**2 + 1.0)
    )
    tangent_cosine = float(np.median(tangent_cosines))
    if tangent_cosine < config.pair_min_tangent_cosine:
        return None

    mean_slope = 0.5 * (left_slope + right_slope)
    normal_distance = np.abs(signed_separation) / np.sqrt(mean_slope**2 + 1.0)
    median_distance = float(np.median(normal_distance))
    p90_distance = float(np.percentile(normal_distance, 90))
    if not (
        config.pair_min_median_distance_px
        <= median_distance
        <= config.pair_max_median_distance_px
        and p90_distance <= config.pair_max_p90_distance_px
    ):
        return None
    if (
        config.pair_wide_requires_adjacent_objects
        and median_distance > config.pair_core_max_median_distance_px
        and abs(left.object_index - right.object_index) != 1
    ):
        return None

    score = (
        median_distance
        + 0.30 * p90_distance
        + 8.0 * (1.0 - overlap_ratio)
        + 8.0 * (1.0 - tangent_cosine)
    )
    return _PairCandidate(
        left=left_index,
        right=right_index,
        score=score,
        median_distance=median_distance,
        p90_distance=p90_distance,
        overlap_ratio=overlap_ratio,
        tangent_cosine=tangent_cosine,
    )


def _is_ambiguous(
    boundary_index: int,
    chosen_other: int,
    chosen_score: float,
    options: Mapping[int, list[_PairCandidate]],
    config: ParserConfig,
) -> bool:
    alternatives = [
        candidate
        for candidate in options.get(boundary_index, [])
        if candidate.left != chosen_other and candidate.right != chosen_other
    ]
    if not alternatives:
        return False
    alternative_score = min(candidate.score for candidate in alternatives)
    return (
        alternative_score <= chosen_score * config.pair_ambiguity_ratio
        or alternative_score - chosen_score <= config.pair_ambiguity_margin_px
    )


def _find_boundary_pairs(
    boundaries: Sequence[LaneBoundary],
    config: ParserConfig,
    warnings: list[ParseWarning],
) -> list[tuple[int, int]]:
    candidates: list[_PairCandidate] = []
    options: dict[int, list[_PairCandidate]] = defaultdict(list)
    for left_index in range(len(boundaries)):
        for right_index in range(left_index + 1, len(boundaries)):
            candidate = _pair_candidate(
                left_index,
                right_index,
                boundaries[left_index],
                boundaries[right_index],
                config,
            )
            if candidate is None:
                continue
            candidates.append(candidate)
            options[left_index].append(candidate)
            options[right_index].append(candidate)

    best: dict[int, _PairCandidate] = {
        index: min(items, key=lambda item: (item.score, item.left, item.right))
        for index, items in options.items()
    }
    accepted: list[tuple[int, int]] = []
    used: set[int] = set()
    warned_pairs: set[tuple[int, int]] = set()
    for candidate in sorted(candidates, key=lambda item: (item.score, item.left, item.right)):
        left, right = candidate.left, candidate.right
        if left in used or right in used:
            continue
        if best.get(left) is not candidate or best.get(right) is not candidate:
            continue
        ambiguous = _is_ambiguous(left, right, candidate.score, options, config) or _is_ambiguous(
            right, left, candidate.score, options, config
        )
        if ambiguous:
            key = (left, right)
            if key not in warned_pairs:
                warned_pairs.add(key)
                _warning(
                    warnings,
                    config,
                    "ambiguous_boundary_pair",
                    "close boundaries were left separate because pairing was ambiguous",
                    object_ids=(boundaries[left].source_id, boundaries[right].source_id),
                    details={
                        "score": round(candidate.score, 4),
                        "median_distance_px": round(candidate.median_distance, 4),
                    },
                )
            continue
        accepted.append((left, right))
        used.add(left)
        used.add(right)
    return accepted


def _resample_arclength(points: np.ndarray, count: int) -> np.ndarray:
    points = _dedupe_consecutive(points)
    if len(points) < 2:
        return points.copy()
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    if cumulative[-1] <= 1e-9:
        return points[:1].copy()
    targets = np.linspace(0.0, cumulative[-1], count, dtype=np.float64)
    x = np.interp(targets, cumulative, points[:, 0])
    y = np.interp(targets, cumulative, points[:, 1])
    return np.column_stack((x, y))


def _endpoint_border(points: np.ndarray, config: ParserConfig) -> str | None:
    """점 하나가 놓인 화면 가장자리 (없으면 None).

    화면 밖에서 시작한 선은 점마다 화면 안으로 잘려, 화면 가장자리를 따라가는 가짜 구간이 생길 수 있다.
    그러면 폴리곤을 닫는 변이 그 구간과 겹친다.
    """

    x, y = float(points[0]), float(points[1])
    tolerance = 1e-7
    if abs(x) <= tolerance:
        return "left"
    if abs(x - (config.image_width - 1.0)) <= tolerance:
        return "right"
    if abs(y) <= tolerance:
        return "top"
    if abs(y - (config.image_height - 1.0)) <= tolerance:
        return "bottom"
    return None


def _collapse_endpoint_border_runs(
    points: np.ndarray, config: ParserConfig
) -> tuple[np.ndarray, int]:
    """화면 가장자리를 따라 잘린 구간을 실제로 화면에 들어오는 점으로 줄인다."""

    result = points
    removed = 0
    if len(result) < 3:
        return result, removed

    start_border = _endpoint_border(result[0], config)
    if start_border is not None:
        first_inside = 1
        while (
            first_inside < len(result)
            and _endpoint_border(result[first_inside], config) == start_border
        ):
            first_inside += 1
        # 선이 화면 안으로 들어오기 직전의 가장자리 점은 남긴다. 전부 가장자리에 있는 경로는 그대로 둔다.
        if 1 < first_inside < len(result):
            removed_here = first_inside - 1
            result = result[removed_here:]
            removed += removed_here

    if len(result) < 3:
        return result, removed
    end_border = _endpoint_border(result[-1], config)
    if end_border is not None:
        last_inside = len(result) - 2
        while last_inside >= 0 and _endpoint_border(result[last_inside], config) == end_border:
            last_inside -= 1
        if 0 <= last_inside < len(result) - 2:
            keep_through = last_inside + 2
            removed_here = len(result) - keep_through
            result = result[:keep_through]
            removed += removed_here
    return result, removed


def _align_pair(
    left: np.ndarray, right: np.ndarray, config: ParserConfig
) -> tuple[np.ndarray, np.ndarray]:
    same_cost = float(np.linalg.norm(left[0] - right[0]) + np.linalg.norm(left[-1] - right[-1]))
    reverse_cost = float(
        np.linalg.norm(left[0] - right[-1]) + np.linalg.norm(left[-1] - right[0])
    )
    if reverse_cost < same_cost:
        right = right[::-1]
    longest = max(
        float(np.sum(np.linalg.norm(np.diff(left, axis=0), axis=1))),
        float(np.sum(np.linalg.norm(np.diff(right, axis=0), axis=1))),
    )
    count = max(8, int(math.ceil(longest / config.sample_spacing_px)) + 1)
    count = min(count, config.max_polygon_side_samples)
    return _resample_arclength(left, count), _resample_arclength(right, count)


def _polygon_area(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def _clip_polygon(points: np.ndarray, config: ParserConfig) -> np.ndarray:
    result = points.copy()
    result[:, 0] = np.clip(result[:, 0], 0.0, config.image_width - 1.0)
    result[:, 1] = np.clip(result[:, 1], 0.0, config.image_height - 1.0)
    result = _dedupe_consecutive(result)
    if len(result) > 1 and np.linalg.norm(result[0] - result[-1]) <= 1e-7:
        result = result[:-1]
    return result


def _orientation(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
) -> int:
    cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    if abs(cross) <= 1e-12:
        return 0
    return 1 if cross > 0 else -1


def _point_on_segment(
    a: tuple[float, float],
    b: tuple[float, float],
    point: tuple[float, float],
) -> bool:
    epsilon = 1e-12
    return (
        min(a[0], b[0]) - epsilon <= point[0] <= max(a[0], b[0]) + epsilon
        and min(a[1], b[1]) - epsilon <= point[1] <= max(a[1], b[1]) + epsilon
    )


def _segments_intersect(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
    d: tuple[float, float],
) -> bool:
    orientations = (
        _orientation(a, b, c),
        _orientation(a, b, d),
        _orientation(c, d, a),
        _orientation(c, d, b),
    )
    if orientations[0] != orientations[1] and orientations[2] != orientations[3]:
        return True
    return (
        (orientations[0] == 0 and _point_on_segment(a, b, c))
        or (orientations[1] == 0 and _point_on_segment(a, b, d))
        or (orientations[2] == 0 and _point_on_segment(c, d, a))
        or (orientations[3] == 0 and _point_on_segment(c, d, b))
    )


def _polygon_is_simple(points: np.ndarray, config: ParserConfig) -> bool:
    """YOLO 라벨에 쓸 소수 여섯째 자리 좌표로 맞추고, 스스로 꼬인 폴리곤을 걸러 낸다.

    길고 가는 차선 띠는 긴 축을 따라 훑어 검사 비용을 줄인다. 여기서 반올림하는 것은 일부러다 —
    픽셀에서는 다른 점이 소수 여섯째 자리로 쓰면 같아질 수 있다.
    """

    if len(points) < 3 or not np.all(np.isfinite(points)):
        return False
    normalised = np.column_stack(
        (
            np.round(points[:, 0] / config.image_width, 6),
            np.round(points[:, 1] / config.image_height, 6),
        )
    )
    vertices = [tuple(map(float, point)) for point in normalised]
    if len(set(vertices)) != len(vertices):
        return False

    edge_count = len(vertices)
    primary_axis = int(np.ptp(normalised[:, 1]) > np.ptp(normalised[:, 0]))
    secondary_axis = 1 - primary_axis
    edge_bounds: list[tuple[float, float, float, float, int]] = []
    for index in range(edge_count):
        start = vertices[index]
        end = vertices[(index + 1) % edge_count]
        edge_bounds.append(
            (
                min(start[primary_axis], end[primary_axis]),
                max(start[primary_axis], end[primary_axis]),
                min(start[secondary_axis], end[secondary_axis]),
                max(start[secondary_axis], end[secondary_axis]),
                index,
            )
        )

    active: list[tuple[float, float, float, float, int]] = []
    epsilon = 1e-12
    for current in sorted(edge_bounds, key=lambda item: (item[0], item[4])):
        active = [item for item in active if item[1] + epsilon >= current[0]]
        first = current[4]
        a = vertices[first]
        b = vertices[(first + 1) % edge_count]
        for other in active:
            second = other[4]
            if (
                second == first
                or second == (first + 1) % edge_count
                or first == (second + 1) % edge_count
            ):
                continue
            if other[3] + epsilon < current[2] or current[3] + epsilon < other[2]:
                continue
            c = vertices[second]
            d = vertices[(second + 1) % edge_count]
            if _segments_intersect(a, b, c, d):
                return False
        active.append(current)
    return True


def _standalone_ribbon(
    centerline: np.ndarray,
    width: float,
    config: ParserConfig,
) -> np.ndarray:
    tangent = np.gradient(centerline, axis=0)
    tangent_norm = np.linalg.norm(tangent, axis=1)
    tangent_norm[tangent_norm <= 1e-9] = 1.0
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0])) / tangent_norm[:, None]
    half_width = 0.5 * width
    first_edge = centerline + half_width * normal
    second_edge = centerline - half_width * normal
    return _clip_polygon(np.vstack((first_edge, second_edge[::-1])), config)


def _local_segment_fallback(
    centerline: np.ndarray, config: ParserConfig
) -> tuple[np.ndarray, int, float] | None:
    """선 전체로 띠를 만들 수 없을 때 쓰는 보수적인 부분 띠.

    관측된 중심선 한 구간만 덮는다. 고리를 가로질러 닫거나 관계없는 선 사이를 채우지 않으며, 쓸 때마다 보고서에 남긴다.
    """

    differences = np.diff(centerline, axis=0)
    lengths = np.linalg.norm(differences, axis=1)
    for segment_index in np.argsort(lengths)[::-1]:
        length = float(lengths[segment_index])
        if length <= 1e-9:
            continue
        start = centerline[segment_index]
        end = centerline[segment_index + 1]
        tangent = (end - start) / length
        normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float64)
        for width_scale in (1.0, 0.5, 0.25):
            half_width = 0.5 * config.standalone_width_px * width_scale
            polygon = _clip_polygon(
                np.vstack(
                    (
                        start + half_width * normal,
                        end + half_width * normal,
                        end - half_width * normal,
                        start - half_width * normal,
                    )
                ),
                config,
            )
            if (
                _polygon_area(polygon) >= config.min_polygon_area_px2
                and _polygon_is_simple(polygon, config)
            ):
                return polygon, int(segment_index), 2.0 * half_width
    return None


def _safe_standalone_polygon(
    centerline: np.ndarray, config: ParserConfig
) -> tuple[np.ndarray, float, int | None] | None:
    for width_scale in (1.0, 0.75, 0.5, 0.375, 0.25, 0.125, 0.0625):
        effective_width = config.standalone_width_px * width_scale
        polygon = _standalone_ribbon(centerline, effective_width, config)
        if (
            _polygon_area(polygon) >= config.min_polygon_area_px2
            and _polygon_is_simple(polygon, config)
        ):
            return polygon, effective_width, None
    fallback = _local_segment_fallback(centerline, config)
    if fallback is None:
        return None
    polygon, segment_index, effective_width = fallback
    return polygon, effective_width, segment_index


def _align_pair_on_y_overlap(
    left: np.ndarray, right: np.ndarray, config: ParserConfig
) -> tuple[np.ndarray, np.ndarray] | None:
    left_profile = _unique_y_profile(left)
    right_profile = _unique_y_profile(right)
    if left_profile is None or right_profile is None:
        return None
    left_y, left_x = left_profile
    right_y, right_x = right_profile
    overlap_start = max(float(left_y[0]), float(right_y[0]))
    overlap_end = min(float(left_y[-1]), float(right_y[-1]))
    if overlap_end - overlap_start <= 1e-9:
        return None
    count = max(8, int(math.ceil((overlap_end - overlap_start) / config.sample_spacing_px)) + 1)
    count = min(count, config.max_polygon_side_samples)
    sample_y = np.linspace(overlap_start, overlap_end, count, dtype=np.float64)
    left_points = np.column_stack((np.interp(sample_y, left_y, left_x), sample_y))
    right_points = np.column_stack((np.interp(sample_y, right_y, right_x), sample_y))
    return left_points, right_points


def _paired_instance(
    left: LaneBoundary,
    right: LaneBoundary,
    config: ParserConfig,
    warnings: list[ParseWarning],
) -> LaneInstance | None:
    # 원래 띠의 면적이 충분한 쌍만 아래 보정을 받는다 — 보정은 꼭짓점만 바꾸고, 어떤 차선이 들어가는지는 바꾸지 않는다.
    original_left, original_right = _align_pair(left.points, right.points, config)
    if len(original_left) < 2 or len(original_right) < 2:
        return None
    original_polygon = _clip_polygon(
        np.vstack((original_left, original_right[::-1])), config
    )
    if _polygon_area(original_polygon) < config.min_polygon_area_px2:
        _warning(
            warnings,
            config,
            "degenerate_paired_polygon",
            "paired boundaries produced a near-zero-area polygon and were kept separate",
            object_ids=(left.source_id, right.source_id),
        )
        return None

    left_source, left_removed = _collapse_endpoint_border_runs(left.points, config)
    right_source, right_removed = _collapse_endpoint_border_runs(right.points, config)
    left_points, right_points = _align_pair(left_source, right_source, config)
    polygon = _clip_polygon(np.vstack((left_points, right_points[::-1])), config)
    centerline = 0.5 * (left_points + right_points)
    if left_removed or right_removed:
        _warning(
            warnings,
            config,
            "endpoint_border_run_collapsed",
            "clipped border samples were collapsed to the visible lane entry point",
            object_ids=(left.source_id, right.source_id),
            details={"left_vertices_removed": left_removed, "right_vertices_removed": right_removed},
        )

    polygon_usable = (
        _polygon_area(polygon) >= config.min_polygon_area_px2
        and _polygon_is_simple(polygon, config)
    )
    if not polygon_usable:
        realigned = _align_pair_on_y_overlap(left_source, right_source, config)
        if realigned is not None:
            candidate_left, candidate_right = realigned
            candidate_polygon = _clip_polygon(
                np.vstack((candidate_left, candidate_right[::-1])), config
            )
            if (
                _polygon_area(candidate_polygon) >= config.min_polygon_area_px2
                and _polygon_is_simple(candidate_polygon, config)
            ):
                left_points, right_points = candidate_left, candidate_right
                polygon = candidate_polygon
                centerline = 0.5 * (left_points + right_points)
                _warning(
                    warnings,
                    config,
                    "paired_polygon_y_realigned",
                    "paired boundaries were aligned on their shared visible y-range to avoid a crossing",
                    object_ids=(left.source_id, right.source_id),
                )

        polygon_usable = (
            _polygon_area(polygon) >= config.min_polygon_area_px2
            and _polygon_is_simple(polygon, config)
        )
        if not polygon_usable:
            safe = _safe_standalone_polygon(centerline, config)
            if safe is None:
                # 정상 입력이면 여기로 오지 않는다. 기록 없이 차선 수를 바꾸느니 원래 결과를 남긴다.
                polygon = original_polygon
                centerline = 0.5 * (original_left + original_right)
                _warning(
                    warnings,
                    config,
                    "paired_polygon_unresolved",
                    "paired polygon could not be simplified without changing lane membership",
                    object_ids=(left.source_id, right.source_id),
                )
            else:
                polygon, effective_width, segment_index = safe
                _warning(
                    warnings,
                    config,
                    "paired_centerline_fallback",
                    "crossing paired boundaries were replaced by a conservative centerline ribbon",
                    object_ids=(left.source_id, right.source_id),
                    details={
                        "effective_width_px": effective_width,
                        "local_segment_index": segment_index,
                    },
                )
    ordered = sorted((left, right), key=lambda boundary: boundary.object_index)
    return LaneInstance(
        category=left.category,
        color=left.color,
        style=left.style,
        polygon=polygon,
        centerline=centerline,
        source_ids=tuple(boundary.source_id for boundary in ordered),
        source_indices=tuple(boundary.object_index for boundary in ordered),
        paired=True,
    )


def _standalone_instance(
    boundary: LaneBoundary,
    config: ParserConfig,
    warnings: list[ParseWarning],
) -> LaneInstance | None:
    source_points, border_vertices_removed = _collapse_endpoint_border_runs(
        boundary.points, config
    )
    length = float(np.sum(np.linalg.norm(np.diff(source_points, axis=0), axis=1)))
    count = max(8, int(math.ceil(length / config.sample_spacing_px)) + 1)
    count = min(count, config.max_polygon_side_samples)
    centerline = _resample_arclength(source_points, count)
    if len(centerline) < 2:
        return None
    if np.all(np.linalg.norm(np.diff(centerline, axis=0), axis=1) <= 1e-9):
        return None
    safe = _safe_standalone_polygon(centerline, config)
    if safe is None:
        _warning(
            warnings,
            config,
            "degenerate_standalone_polygon",
            "standalone boundary could not be expanded into a valid polygon",
            object_ids=(boundary.source_id,),
        )
        return None
    polygon, effective_width, segment_index = safe
    if border_vertices_removed:
        _warning(
            warnings,
            config,
            "endpoint_border_run_collapsed",
            "clipped border samples were collapsed to the visible lane entry point",
            object_ids=(boundary.source_id,),
            details={"vertices_removed": border_vertices_removed},
        )
    if segment_index is not None:
        _warning(
            warnings,
            config,
            "standalone_local_segment_fallback",
            "a self-crossing centerline was represented by one conservative local segment",
            object_ids=(boundary.source_id,),
            details={
                "segment_index": segment_index,
                "effective_width_px": effective_width,
            },
        )
    elif effective_width < config.standalone_width_px:
        _warning(
            warnings,
            config,
            "standalone_width_reduced",
            "standalone ribbon width was reduced to keep the polygon simple",
            object_ids=(boundary.source_id,),
            details={
                "requested_width_px": config.standalone_width_px,
                "effective_width_px": effective_width,
            },
        )
    return LaneInstance(
        category=boundary.category,
        color=boundary.color,
        style=boundary.style,
        polygon=polygon,
        centerline=centerline,
        source_ids=(boundary.source_id,),
        source_indices=(boundary.object_index,),
        paired=False,
    )


def _parse_mapping(
    annotation: Mapping[str, Any],
    config: ParserConfig,
    initial_warnings: Sequence[ParseWarning] = (),
) -> ParseResult:
    warnings = list(initial_warnings)
    stats: Counter[str] = Counter()
    image_name = annotation.get("name", "")
    if not isinstance(image_name, str):
        image_name = str(image_name)

    frames = annotation.get("frames")
    if not isinstance(frames, Sequence) or isinstance(frames, (str, bytes)) or not frames:
        _warning(
            warnings,
            config,
            "missing_frames",
            "annotation must contain a non-empty frames list",
            fatal=True,
        )
        return _empty_result(config, warnings, image_name=image_name, stats=stats)
    frame = frames[0]
    if not isinstance(frame, Mapping):
        _warning(
            warnings,
            config,
            "invalid_frame",
            "frames[0] must be an object",
            fatal=True,
        )
        return _empty_result(config, warnings, image_name=image_name, stats=stats)
    objects = frame.get("objects")
    if not isinstance(objects, Sequence) or isinstance(objects, (str, bytes)):
        _warning(
            warnings,
            config,
            "invalid_objects",
            "frames[0].objects must be a list",
            fatal=True,
        )
        return _empty_result(config, warnings, image_name=image_name, stats=stats)
    if len(objects) > config.max_objects:
        _warning(
            warnings,
            config,
            "object_limit",
            "objects beyond the configured safety limit were ignored",
            details={"count": len(objects), "limit": config.max_objects},
        )
        objects = objects[: config.max_objects]

    boundaries: list[LaneBoundary] = []
    stats["objects_total"] = len(objects)
    for object_index, item in enumerate(objects):
        if not isinstance(item, Mapping):
            stats["skipped_invalid_object"] += 1
            _warning(
                warnings,
                config,
                "invalid_object",
                "object entry is not a JSON object",
                details={"object_index": object_index},
            )
            continue

        source_id = _safe_source_id(item.get("id"), object_index)
        category = _normalise_text(item.get("category"))
        if not category.startswith("lane/"):
            stats["skipped_non_lane"] += 1
            continue
        stats["lane_objects"] += 1
        lane_kind = category.split("/", 1)[1]
        if "road curb" in lane_kind or "crosswalk" in lane_kind:
            stats["skipped_excluded_category"] += 1
            continue

        attributes = item.get("attributes")
        if not isinstance(attributes, Mapping):
            attributes = {}
            _warning(
                warnings,
                config,
                "invalid_attributes",
                "lane attributes are missing or malformed",
                object_ids=(source_id,),
            )
        direction = _normalise_text(attributes.get("direction"))
        if direction != "parallel":
            stats["skipped_non_parallel"] += 1
            continue

        words = set(lane_kind.replace("/", " ").split())
        if "white" in words:
            color = "white"
        elif "yellow" in words:
            color = "yellow"
        else:
            stats["skipped_unknown_color"] += 1
            continue

        style = _normalise_text(attributes.get("style"))
        if not style:
            style = "unknown"
            _warning(
                warnings,
                config,
                "missing_lane_style",
                "lane style is missing; boundary will not be paired",
                object_ids=(source_id,),
            )

        points = poly2d_to_polyline(
            item.get("poly2d"),
            config=config,
            warnings=warnings,
            object_id=source_id,
        )
        if len(points) < 2:
            stats["skipped_invalid_geometry"] += 1
            continue
        boundaries.append(
            LaneBoundary(
                source_id=source_id,
                object_index=object_index,
                category=category,
                color=color,
                style=style,
                direction=direction,
                points=points,
            )
        )

    pairs = _find_boundary_pairs(boundaries, config, warnings)
    paired_indices: set[int] = set()
    lanes: list[LaneInstance] = []
    for left_index, right_index in pairs:
        instance = _paired_instance(
            boundaries[left_index], boundaries[right_index], config, warnings
        )
        if instance is None:
            continue
        paired_indices.update((left_index, right_index))
        lanes.append(instance)

    for boundary_index, boundary in enumerate(boundaries):
        if boundary_index in paired_indices:
            continue
        instance = _standalone_instance(boundary, config, warnings)
        if instance is not None:
            lanes.append(instance)

    lanes.sort(key=lambda lane: (min(lane.source_indices), lane.source_indices))
    stats["selected_boundaries"] = len(boundaries)
    stats["paired_instances"] = sum(lane.paired for lane in lanes)
    stats["standalone_instances"] = sum(not lane.paired for lane in lanes)
    stats["instances"] = len(lanes)
    stats["warnings"] = len(warnings)
    return ParseResult(
        image_name=image_name,
        image_size=(config.image_width, config.image_height),
        boundaries=tuple(boundaries),
        lanes=tuple(lanes),
        warnings=tuple(warnings),
        stats=dict(stats),
    )


def parse_bdd_lane_file(
    path: str | Path, *, config: ParserConfig | None = None
) -> ParseResult:
    """BDD100K JSON 파일 하나를 읽는다 — 잘못된 라벨이 있어도 예외를 밖으로 내지 않는다."""

    cfg = config or ParserConfig()
    warnings: list[ParseWarning] = []
    source = Path(path)
    try:
        size = source.stat().st_size
    except OSError as error:
        _warning(
            warnings,
            cfg,
            "file_not_found",
            "label file could not be opened",
            details={"path": str(source), "error": str(error)},
            fatal=True,
        )
        return _empty_result(cfg, warnings, image_name=source.stem)
    if size > cfg.max_json_bytes:
        _warning(
            warnings,
            cfg,
            "file_too_large",
            "label file exceeds the configured safety limit",
            details={"path": str(source), "size": size, "limit": cfg.max_json_bytes},
            fatal=True,
        )
        return _empty_result(cfg, warnings, image_name=source.stem)
    try:
        with source.open("r", encoding="utf-8-sig") as stream:
            annotation = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        _warning(
            warnings,
            cfg,
            "invalid_json",
            "label file is not valid UTF-8 JSON",
            details={"path": str(source), "error": str(error)},
            fatal=True,
        )
        return _empty_result(cfg, warnings, image_name=source.stem)
    if not isinstance(annotation, Mapping):
        _warning(
            warnings,
            cfg,
            "invalid_root",
            "legacy per-image annotation root must be a JSON object",
            details={"path": str(source)},
            fatal=True,
        )
        return _empty_result(cfg, warnings, image_name=source.stem)
    return _parse_mapping(annotation, cfg, warnings)


def parse_bdd_lane_annotation(
    source: Mapping[str, Any] | str | Path,
    *,
    config: ParserConfig | None = None,
) -> ParseResult:
    """이미 읽은 라벨(dict) 또는 JSON 파일 경로를 해석한다."""

    cfg = config or ParserConfig()
    if isinstance(source, (str, Path)):
        return parse_bdd_lane_file(source, config=cfg)
    if not isinstance(source, Mapping):
        warnings: list[ParseWarning] = []
        _warning(
            warnings,
            cfg,
            "invalid_root",
            "annotation must be a mapping or a JSON file path",
            fatal=True,
        )
        return _empty_result(cfg, warnings)
    return _parse_mapping(source, cfg)


# 같은 함수의 다른 이름 (둘 다 쓴다)
parse_bdd_annotation = parse_bdd_lane_annotation
parse_bdd_lanes = parse_bdd_lane_annotation
