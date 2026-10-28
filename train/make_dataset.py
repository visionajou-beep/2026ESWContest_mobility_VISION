from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import random
import shutil
import uuid
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import yaml
from PIL import Image, UnidentifiedImageError


ROOT = Path(__file__).resolve().parent
DEFAULT_IMAGES_ROOT = ROOT / "bdd100k_images_100k" / "100k"
DEFAULT_LABELS_ROOT = ROOT / "bdd100k_labels" / "100k"
DEFAULT_OUTPUT_ROOT = ROOT / "bdd100k_yolo11_seg"

STRATIFY_FIELDS = ("weather", "timeofday", "scene")
DEFAULT_SEED = 42
DEFAULT_WORKERS = 4
DEFAULT_VAL_RATIO = 0.10
LEGACY_TRAIN_TOTAL = 70_000
LEGACY_TRAIN_TARGET = 63_014
LEGACY_VAL_TARGET = 6_986

# 학습에 쓴 데이터셋의 나눔별 크기. 이미지 수는 반드시 맞아야 하고, 나눔별 차선 · 빈 라벨 수는 참고값이다.
EXPECTED_FULL_COUNTS: dict[str, dict[str, int]] = {
    "train": {"images": 63_014, "lane_instances": 165_274, "empty_labels": 13_517},
    "val": {"images": 6_986, "lane_instances": 18_148, "empty_labels": 1_497},
    "test": {"images": 10_000, "lane_instances": 26_321, "empty_labels": 2_114},
}

# 전체 변환 결과가 맞아야 하는 합계 (데이터셋을 확정하는 기준).
CALIBRATED_GLOBAL_COUNTS: dict[str, int] = {
    "images": 80_000,
    "lane_instances": 209_670,
    "empty_labels": 17_128,
}
PARSER_CALIBRATION: dict[str, Any] = {
    "selected_boundaries": 323_259,
    "paired": 113_589,
    "lane_instances": 209_670,
    "empty_labels": 17_128,
    "fatal": 0,
    "polygon_expansion_px": 27,
    "pairing_continuity": "consecutive raw object_index only",
    "historical_lane_instance_delta": -73,
    "basis": (
        "Full-parser audit with 27 px polygon expansion and the safety limit "
        "that paired boundaries must be consecutive in raw object_index order."
    ),
}

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
PROBLEM_FIELDS = (
    "severity",
    "code",
    "source_split",
    "output_split",
    "annotation",
    "image",
    "object_ids",
    "message",
    "details",
)

# 라벨 해석기(bdd_label.py)가 '이 이미지의 라벨 전체를 쓸 수 없음'으로 내는 경고 코드.
FATAL_PARSER_WARNING_CODES = frozenset(
    {
        "file_not_found",
        "file_too_large",
        "invalid_json",
        "invalid_root",
        "missing_frames",
        "invalid_frame",
        "invalid_objects",
    }
)

MANIFEST_FIELDS = (
    "output_split",
    "source_split",
    "annotation_file",
    "stem",
    "weather",
    "timeofday",
    "scene",
)


@dataclass(frozen=True)
class SourceRecord:
    annotation: Path
    source_split: str
    stratum: tuple[str, str, str]

    @property
    def stem(self) -> str:
        return self.annotation.stem


@dataclass(frozen=True)
class WorkItem:
    record: SourceRecord
    output_split: str


@dataclass
class Problem:
    severity: str
    code: str
    source_split: str
    output_split: str = ""
    annotation: str = ""
    image: str = ""
    object_ids: str = ""
    message: str = ""
    details: str = ""

    def as_row(self) -> dict[str, str]:
        return {field_name: str(getattr(self, field_name)) for field_name in PROBLEM_FIELDS}


@dataclass
class ConversionResult:
    output_split: str
    success: bool
    lane_instances: int = 0
    empty_label: bool = False
    materialization: str = ""
    parser_warnings: int = 0
    problems: list[Problem] = field(default_factory=list)


ParserFunction = Callable[..., Any]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "BDD100K 이미지별 차선 라벨을 1 클래스(lane_boundary) YOLO 분할 데이터셋으로 바꾼다 "
            "(같은 입력이면 항상 같은 결과)."
        )
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        default=DEFAULT_IMAGES_ROOT,
        help="BDD100K 공식 이미지 폴더 (100k/train · 100k/val · 100k/test 가 있는 곳).",
    )
    parser.add_argument(
        "--labels-root",
        type=Path,
        default=DEFAULT_LABELS_ROOT,
        help="BDD100K 공식 라벨(JSON) 폴더 (100k/train · 100k/val · 100k/test).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="만들 YOLO 데이터셋 폴더.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help=(
            "나눔(train · val · test)마다 이 개수만 변환한다 (시험용). "
            "빼거나 0 이면 8 만 장 전부."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="동시에 처리할 작업 수 (기본 4).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="train / val 나눔에 쓰는 난수 시드 (기본 42).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=DEFAULT_VAL_RATIO,
        help="원본 개수가 표준과 다를 때의 검증 비율 (기본 0.10).",
    )
    return parser.parse_args(argv)


def _load_parser() -> ParserFunction:
    try:
        from bdd_label import parse_bdd_lane_file
    except ImportError as exc:
        raise SystemExit(
            "Unable to import bdd_label.parse_bdd_lane_file. "
            "Keep bdd_label.py beside make_dataset.py and install the local "
            'environment with `.\\.venv\\Scripts\\python.exe -m pip install -e ".[dev]"`. '
            f"Original import error: {exc}"
        ) from exc

    if not callable(parse_bdd_lane_file):
        raise SystemExit(
            "bdd_label was imported, but parse_bdd_lane_file(path, *, config=None) "
            "is missing or is not callable."
        )
    return parse_bdd_lane_file


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalise_attribute(value: Any) -> str:
    text = str(value).strip().lower() if value is not None else ""
    return text or "undefined"


def _read_source_record(path: Path, source_split: str) -> tuple[SourceRecord, Problem | None]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            annotation = json.load(stream)
        attributes = annotation.get("attributes", {})
        if not isinstance(attributes, Mapping):
            raise TypeError("top-level attributes must be an object")
        stratum = tuple(
            _normalise_attribute(attributes.get(field_name))
            for field_name in STRATIFY_FIELDS
        )
        return SourceRecord(path, source_split, stratum), None
    except Exception as exc:
        record = SourceRecord(
            path,
            source_split,
            ("unreadable", "unreadable", "unreadable"),
        )
        problem = Problem(
            severity="error",
            code="stratification_metadata_error",
            source_split=source_split,
            annotation=str(path),
            message=str(exc),
        )
        return record, problem


def _discover_records(
    labels_root: Path,
    source_split: str,
    workers: int,
) -> tuple[list[SourceRecord], list[Problem]]:
    split_dir = labels_root / source_split
    if not split_dir.is_dir():
        raise FileNotFoundError(
            f"Missing BDD100K annotation directory: {split_dir}. "
            "Extract bdd100k_labels.zip so the default layout is "
            "bdd100k_labels/100k/{train,val}."
        )

    paths = sorted(split_dir.glob("*.json"), key=lambda item: item.name)
    if not paths:
        raise FileNotFoundError(f"No JSON annotations found in {split_dir}")
    duplicate_stems = [
        stem for stem, count in Counter(path.stem for path in paths).items() if count > 1
    ]
    if duplicate_stems:
        raise ValueError(
            f"Duplicate annotation stems in {split_dir}: {duplicate_stems[:5]}"
        )

    records: list[SourceRecord] = []
    problems: list[Problem] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        parsed = executor.map(
            lambda source: _read_source_record(source, source_split),
            paths,
        )
        for record, problem in parsed:
            records.append(record)
            if problem is not None:
                problems.append(problem)
    return records, problems


def _stable_seed(seed: int, *parts: str) -> int:
    material = "\0".join((str(seed), *parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _allocate_validation_quotas(
    groups: Mapping[tuple[str, str, str], Sequence[SourceRecord]],
    target_val: int,
) -> dict[tuple[str, str, str], int]:
    total = sum(len(records) for records in groups.values())
    if total <= 0:
        raise ValueError("Cannot split an empty official training set.")
    if not 0 <= target_val <= total:
        raise ValueError(f"Invalid validation target {target_val} for {total} records.")

    quotas: dict[tuple[str, str, str], int] = {}
    remainders: list[tuple[float, tuple[str, str, str]]] = []
    for key in sorted(groups):
        exact = len(groups[key]) * target_val / total
        base = math.floor(exact)
        quotas[key] = base
        remainders.append((exact - base, key))

    remaining = target_val - sum(quotas.values())
    remainders.sort(key=lambda item: (-item[0], item[1]))
    for _, key in remainders[:remaining]:
        quotas[key] += 1
    return quotas


def _stratified_train_val_split(
    records: Sequence[SourceRecord],
    *,
    seed: int,
    val_ratio: float,
) -> tuple[list[SourceRecord], list[SourceRecord], dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[SourceRecord]] = defaultdict(list)
    for record in records:
        groups[record.stratum].append(record)

    legacy_target_applied = (
        len(records) == LEGACY_TRAIN_TOTAL
        and math.isclose(val_ratio, DEFAULT_VAL_RATIO, rel_tol=0.0, abs_tol=1e-12)
    )
    target_val = (
        LEGACY_VAL_TARGET
        if legacy_target_applied
        else int(round(len(records) * val_ratio))
    )
    quotas = _allocate_validation_quotas(groups, target_val)

    train_records: list[SourceRecord] = []
    val_records: list[SourceRecord] = []
    stratum_summary: dict[str, Any] = {}
    for key in sorted(groups):
        shuffled = sorted(groups[key], key=lambda record: record.annotation.name)
        random.Random(_stable_seed(seed, *key)).shuffle(shuffled)
        val_count = quotas[key]
        val_records.extend(shuffled[:val_count])
        train_records.extend(shuffled[val_count:])
        stratum_summary[" | ".join(key)] = {
            "source": len(shuffled),
            "train": len(shuffled) - val_count,
            "val": val_count,
        }

    train_records.sort(key=lambda record: record.annotation.name)
    val_records.sort(key=lambda record: record.annotation.name)
    if len(train_records) + len(val_records) != len(records):
        raise AssertionError("The stratified split lost or duplicated source records.")
    if len(val_records) != target_val:
        raise AssertionError("The stratified split did not reach its validation target.")

    return train_records, val_records, {
        "fields": list(STRATIFY_FIELDS),
        "seed": seed,
        "requested_val_ratio": val_ratio,
        "legacy_target_applied": legacy_target_applied,
        "target_train": len(train_records),
        "target_val": len(val_records),
        "strata": stratum_summary,
    }


def _deterministic_limit(
    records: Sequence[SourceRecord],
    max_files: int | None,
    *,
    seed: int,
    subset: str,
) -> list[SourceRecord]:
    if max_files is None or max_files == 0 or max_files >= len(records):
        return list(records)
    ranked = sorted(
        records,
        key=lambda record: (
            _stable_seed(seed, "max-files", subset, record.annotation.name),
            record.annotation.name,
        ),
    )
    return sorted(ranked[:max_files], key=lambda record: record.annotation.name)


def _find_image(images_root: Path, record: SourceRecord) -> Path:
    split_dir = images_root / record.source_split
    for suffix in IMAGE_SUFFIXES:
        candidate = split_dir / f"{record.stem}{suffix}"
        if candidate.is_file():
            return candidate
        upper_candidate = split_dir / f"{record.stem}{suffix.upper()}"
        if upper_candidate.is_file():
            return upper_candidate
    raise FileNotFoundError(
        f"No image for {record.annotation.name} in {split_dir}; tried "
        + ", ".join(IMAGE_SUFFIXES)
    )


def _image_size(path: Path) -> tuple[int, int]:
    try:
        with Image.open(path) as image:
            width, height = image.size
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"Cannot read image dimensions for {path}: {exc}") from exc
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image dimensions {width}x{height}: {path}")
    return width, height


def _normalise_polygon(
    polygon: Iterable[Any],
    width: int,
    height: int,
) -> tuple[list[tuple[float, float]] | None, bool]:
    points: list[tuple[float, float]] = []
    clipped = False
    try:
        for raw_point in polygon:
            if len(raw_point) < 2:
                continue
            x = float(raw_point[0])
            y = float(raw_point[1])
            if not math.isfinite(x) or not math.isfinite(y):
                continue
            clipped_x = min(max(x, 0.0), float(width))
            clipped_y = min(max(y, 0.0), float(height))
            clipped = clipped or clipped_x != x or clipped_y != y
            point = (clipped_x / width, clipped_y / height)
            if not points or point != points[-1]:
                points.append(point)
    except (TypeError, ValueError, IndexError):
        return None, clipped

    if len(points) > 1 and points[0] == points[-1]:
        points.pop()
    if len(points) < 3 or len(set(points)) < 3:
        return None, clipped
    return points, clipped


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8", newline="")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _same_volume(source: Path, destination_parent: Path) -> bool:
    try:
        return source.stat().st_dev == destination_parent.stat().st_dev
    except OSError:
        return source.drive.casefold() == destination_parent.drive.casefold()


def _materialise_image(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        try:
            if os.path.samefile(source, destination):
                return "reused_hardlink"
        except OSError:
            pass
        source_stat = source.stat()
        destination_stat = destination.stat()
        if (
            source_stat.st_size == destination_stat.st_size
            and source_stat.st_mtime_ns == destination_stat.st_mtime_ns
        ):
            return "reused_copy"

    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    mode = "copy"
    try:
        if _same_volume(source, destination.parent):
            try:
                os.link(source, temporary)
                mode = "hardlink"
            except OSError:
                shutil.copy2(source, temporary)
                mode = "copy_after_link_failure"
        else:
            shutil.copy2(source, temporary)
        os.replace(temporary, destination)
        return mode
    finally:
        temporary.unlink(missing_ok=True)


def _json_text(value: Any) -> str:
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _parser_warning_problem(
    warning: Any,
    item: WorkItem,
    image: Path,
) -> Problem:
    object_ids = getattr(warning, "object_ids", ()) or ()
    if isinstance(object_ids, (str, int)):
        object_ids = (object_ids,)
    code = str(getattr(warning, "code", "parser_warning"))
    declared_severity = str(getattr(warning, "severity", "")).strip().casefold()
    declared_fatal = bool(getattr(warning, "fatal", False))
    severity = (
        "error"
        if code in FATAL_PARSER_WARNING_CODES
        or declared_fatal
        or declared_severity in {"error", "fatal"}
        else "warning"
    )
    return Problem(
        severity=severity,
        code=code,
        source_split=item.record.source_split,
        output_split=item.output_split,
        annotation=str(item.record.annotation),
        image=str(image),
        object_ids=";".join(str(object_id) for object_id in object_ids),
        message=str(getattr(warning, "message", warning)),
        details=_json_text(getattr(warning, "details", None)),
    )


def _convert_one(
    item: WorkItem,
    *,
    images_root: Path,
    output_root: Path,
    parser_function: ParserFunction,
) -> ConversionResult:
    image: Path | None = None
    problems: list[Problem] = []
    try:
        image = _find_image(images_root, item.record)
        width, height = _image_size(image)
        parsed = parser_function(item.record.annotation)
        lanes = getattr(parsed, "lanes", None)
        if lanes is None:
            raise TypeError(
                "parse_bdd_lane_file() returned an object without a lanes attribute"
            )

        parser_warnings = list(getattr(parsed, "warnings", ()) or ())
        parser_problems = [
            _parser_warning_problem(warning, item, image)
            for warning in parser_warnings
        ]
        problems.extend(parser_problems)
        parser_ok = bool(getattr(parsed, "ok", True))
        fatal_codes = [
            problem.code for problem in parser_problems if problem.severity == "error"
        ]
        if not parser_ok or fatal_codes:
            reason = "parser returned ok=False" if not parser_ok else "fatal parser warning"
            codes = ", ".join(sorted(set(fatal_codes))) or "unspecified"
            raise ValueError(f"{reason}; warning codes: {codes}")

        label_rows: list[str] = []
        for lane_index, lane in enumerate(lanes):
            polygon = getattr(lane, "polygon", None)
            if polygon is None:
                problems.append(
                    Problem(
                        severity="warning",
                        code="missing_lane_polygon",
                        source_split=item.record.source_split,
                        output_split=item.output_split,
                        annotation=str(item.record.annotation),
                        image=str(image),
                        message=f"Lane {lane_index} has no polygon and was skipped.",
                    )
                )
                continue
            points, clipped = _normalise_polygon(polygon, width, height)
            if points is None:
                problems.append(
                    Problem(
                        severity="warning",
                        code="invalid_lane_polygon",
                        source_split=item.record.source_split,
                        output_split=item.output_split,
                        annotation=str(item.record.annotation),
                        image=str(image),
                        message=f"Lane {lane_index} has fewer than three valid points.",
                    )
                )
                continue
            if clipped:
                problems.append(
                    Problem(
                        severity="warning",
                        code="polygon_clipped_to_image",
                        source_split=item.record.source_split,
                        output_split=item.output_split,
                        annotation=str(item.record.annotation),
                        image=str(image),
                        message=f"Lane {lane_index} crossed the image boundary and was clipped.",
                    )
                )
            coordinates = " ".join(
                f"{coordinate:.6f}" for point in points for coordinate in point
            )
            label_rows.append(f"0 {coordinates}")

        output_image = output_root / "images" / item.output_split / image.name
        output_label = output_root / "labels" / item.output_split / f"{image.stem}.txt"
        materialization = _materialise_image(image, output_image)
        label_text = "\n".join(label_rows)
        if label_text:
            label_text += "\n"
        _atomic_write_text(output_label, label_text)
        return ConversionResult(
            output_split=item.output_split,
            success=True,
            lane_instances=len(label_rows),
            empty_label=not label_rows,
            materialization=materialization,
            parser_warnings=len(parser_warnings),
            problems=problems,
        )
    except Exception as exc:
        problems.append(
            Problem(
                severity="error",
                code="conversion_error",
                source_split=item.record.source_split,
                output_split=item.output_split,
                annotation=str(item.record.annotation),
                image=str(image) if image is not None else "",
                message=f"{type(exc).__name__}: {exc}",
            )
        )
        return ConversionResult(
            output_split=item.output_split,
            success=False,
            problems=problems,
        )


def _empty_split_stats(selected: int) -> dict[str, Any]:
    return {
        "selected": selected,
        "converted": 0,
        "failed": 0,
        "lane_instances": 0,
        "empty_labels": 0,
        "parser_warnings": 0,
        "materialization": {},
    }


def _write_data_yaml(output_root: Path, *, dataset_root: Path | None = None) -> Path:
    data_path = output_root / "data.yaml"
    payload = {
        "path": (dataset_root or output_root).resolve().as_posix(),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "lane_boundary"},
    }
    _atomic_write_text(
        data_path,
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
    )
    return data_path


def _write_problems_csv(output_root: Path, problems: Sequence[Problem]) -> Path:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=PROBLEM_FIELDS, lineterminator="\n")
    writer.writeheader()
    for problem in problems:
        writer.writerow(problem.as_row())
    path = output_root / "conversion_problems.csv"
    _atomic_write_text(path, output.getvalue())
    return path


def _split_manifest_text(
    selected: Mapping[str, Sequence[SourceRecord]],
) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
    writer.writeheader()
    for output_split in ("train", "val", "test"):
        for record in sorted(
            selected[output_split], key=lambda item: item.annotation.name
        ):
            writer.writerow(
                {
                    "output_split": output_split,
                    "source_split": record.source_split,
                    "annotation_file": record.annotation.name,
                    "stem": record.stem,
                    "weather": record.stratum[0],
                    "timeofday": record.stratum[1],
                    "scene": record.stratum[2],
                }
            )
    return output.getvalue()


def _write_split_manifest(
    output_root: Path,
    selected: Mapping[str, Sequence[SourceRecord]],
) -> tuple[Path, str]:
    text = _split_manifest_text(selected)
    membership_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
    path = output_root / "split_manifest.csv"
    _atomic_write_text(path, text)
    return path, membership_sha256


def _regression_report(
    stats: Mapping[str, Mapping[str, Any]],
    *,
    official_train_count: int,
    official_val_count: int,
    max_files: int | None,
    seed: int,
    val_ratio: float,
) -> dict[str, Any]:
    applicable = (
        (max_files is None or max_files == 0)
        and official_train_count == LEGACY_TRAIN_TOTAL
        and official_val_count == EXPECTED_FULL_COUNTS["test"]["images"]
        and seed == DEFAULT_SEED
        and math.isclose(val_ratio, DEFAULT_VAL_RATIO, rel_tol=0.0, abs_tol=1e-12)
    )
    subset_image_checks: dict[str, Any] = {}
    historical_subset_drift: dict[str, Any] = {}
    actual_global = {
        metric: sum(int(stats[split][source_key]) for split in EXPECTED_FULL_COUNTS)
        for metric, source_key in (
            ("images", "converted"),
            ("lane_instances", "lane_instances"),
            ("empty_labels", "empty_labels"),
        )
    }
    historical_global = {
        metric: sum(
            int(expected[metric]) for expected in EXPECTED_FULL_COUNTS.values()
        )
        for metric in ("images", "lane_instances", "empty_labels")
    }
    expected_global = {
        metric: int(CALIBRATED_GLOBAL_COUNTS[metric])
        for metric in ("images", "lane_instances", "empty_labels")
    }

    for split, expected in EXPECTED_FULL_COUNTS.items():
        actual_images = int(stats[split]["converted"])
        subset_image_checks[split] = {
            "expected": int(expected["images"]),
            "actual": actual_images,
            "matches": actual_images == int(expected["images"]) if applicable else None,
        }
        historical_subset_drift[split] = {}
        for metric in ("lane_instances", "empty_labels"):
            actual_value = int(stats[split][metric])
            expected_value = int(expected[metric])
            historical_subset_drift[split][metric] = {
                "historical": expected_value,
                "actual": actual_value,
                "drift": actual_value - expected_value if applicable else None,
                "matches_historical": (
                    actual_value == expected_value if applicable else None
                ),
                "acceptance_criterion": False,
            }

    global_matches = {
        metric: actual_global[metric] == expected_global[metric] if applicable else None
        for metric in expected_global
    }
    subset_images_match = (
        all(check["matches"] for check in subset_image_checks.values())
        if applicable
        else None
    )
    acceptance_match = (
        bool(subset_images_match)
        and bool(global_matches["lane_instances"])
        and bool(global_matches["empty_labels"])
        if applicable
        else None
    )
    return {
        "applicable": applicable,
        "all_match": acceptance_match,
        "policy": {
            "blocking": [
                "exact train/val/test image counts",
                "exact global lane instance total across 80,000 images",
                "exact global empty-label total across 80,000 images",
            ],
            "informational_only": (
                "Reference values of the dataset used for training (209,743 lane "
                "instances and per-subset instance/empty counts). The current parser "
                "produces 73 fewer because of the 27 px pairing allowance and the "
                "object_index continuity limit, so per-subset counts may differ slightly."
            ),
        },
        "parser_calibration": {
            **PARSER_CALIBRATION,
            "canonical_expected_global_counts": expected_global,
        },
        "acceptance_checks": {
            "subset_images": {
                "all_match": subset_images_match,
                "checks": subset_image_checks,
            },
            "global_totals": {
                "expected": expected_global,
                "actual": actual_global,
                "matches": global_matches,
                "all_match": (
                    all(global_matches.values()) if applicable else None
                ),
            },
        },
        "historical_baseline_drift": {
            "acceptance_criterion": False,
            "global": {
                metric: {
                    "historical": historical_global[metric],
                    "calibrated_expected": expected_global[metric],
                    "actual": actual_global[metric],
                    "calibration_delta": (
                        expected_global[metric] - historical_global[metric]
                    ),
                    "actual_vs_historical": (
                        actual_global[metric] - historical_global[metric]
                        if applicable
                        else None
                    ),
                }
                for metric in ("lane_instances", "empty_labels")
            },
            "subsets": historical_subset_drift,
        },
    }


def _new_staging_root(output_root: Path) -> Path:
    """기존 데이터셋을 건드리지 않는 옆 폴더 이름 하나."""

    output_root.parent.mkdir(parents=True, exist_ok=True)
    return output_root.parent / f".{output_root.name}.staging-{uuid.uuid4().hex}"


def _remove_exact_backup(path: Path, *, expected_parent: Path) -> None:
    """_publish_staging 이 만든 백업만 지운다."""

    if path.parent != expected_parent:
        raise RuntimeError(f"Refusing to remove backup outside {expected_parent}: {path}")
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _publish_staging(staging_root: Path, output_root: Path) -> Path | None:
    """임시 폴더에서 만든 데이터셋으로 output_root 를 한 번에 바꾼다. 실패하면 이전 폴더를 되살린다.

    두 경로는 같은 상위 폴더에 있어야 한다. 이전 결과는 먼저 백업 이름으로 옮기고, 바꾸기에 실패하면 바로 되돌린다.
    실패한 임시 폴더는 지우지 않아 보고서를 확인할 수 있다.
    """

    if staging_root.parent != output_root.parent:
        raise ValueError("staging_root and output_root must be siblings")
    if not staging_root.is_dir():
        raise FileNotFoundError(f"Staging dataset does not exist: {staging_root}")

    backup_root = output_root.parent / f".{output_root.name}.backup-{uuid.uuid4().hex}"
    had_previous_output = output_root.exists()
    if had_previous_output:
        os.replace(output_root, backup_root)
    try:
        os.replace(staging_root, output_root)
    except Exception:
        if had_previous_output and backup_root.exists() and not output_root.exists():
            os.replace(backup_root, output_root)
        raise

    if had_previous_output and backup_root.exists():
        try:
            _remove_exact_backup(backup_root, expected_parent=output_root.parent)
        except OSError as exc:
            # 새 데이터셋은 이미 제자리에 있다. 정상인 새 결과를 되돌리느니 백업을 남겨 둔다.
            print(f"Published successfully; old backup retained at {backup_root}: {exc}")
            return backup_root
    return None


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1.")
    if args.max_files is not None and args.max_files < 0:
        raise SystemExit("--max-files must be 0 or greater.")
    if not 0.0 < args.val_ratio < 1.0:
        raise SystemExit("--val-ratio must be strictly between 0 and 1.")

    images_root = args.images_root.expanduser().resolve()
    labels_root = args.labels_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    for source_root in (images_root, labels_root):
        if (
            output_root == source_root
            or output_root.is_relative_to(source_root)
            or source_root.is_relative_to(output_root)
        ):
            raise SystemExit(
                f"Refusing overlapping raw/generated data paths: {output_root} "
                f"overlaps {source_root}. Choose a separate --output-root."
            )
    for source_split in ("train", "val"):
        image_dir = images_root / source_split
        if not image_dir.is_dir():
            raise SystemExit(
                f"Missing BDD100K image directory: {image_dir}. Finish and extract "
                "bdd100k_images_100k.zip before conversion."
            )

    parser_function = _load_parser()
    started_at = _utc_now()
    print(f"Reading stratification metadata from {labels_root} ...", flush=True)
    official_train, train_metadata_problems = _discover_records(
        labels_root, "train", args.workers
    )
    official_val, val_metadata_problems = _discover_records(
        labels_root, "val", args.workers
    )

    generated_train, generated_val, split_summary = _stratified_train_val_split(
        official_train,
        seed=args.seed,
        val_ratio=args.val_ratio,
    )
    selected: dict[str, list[SourceRecord]] = {
        "train": _deterministic_limit(
            generated_train, args.max_files, seed=args.seed, subset="train"
        ),
        "val": _deterministic_limit(
            generated_val, args.max_files, seed=args.seed, subset="val"
        ),
        "test": _deterministic_limit(
            official_val, args.max_files, seed=args.seed, subset="test"
        ),
    }
    work_items = [
        WorkItem(record, output_split)
        for output_split in ("train", "val", "test")
        for record in selected[output_split]
    ]

    staging_root = _new_staging_root(output_root)
    staging_root.mkdir(exist_ok=False)
    print(f"Staging new dataset at {staging_root}", flush=True)
    manifest_path, membership_sha256 = _write_split_manifest(staging_root, selected)
    stats = {
        split: _empty_split_stats(len(selected[split]))
        for split in ("train", "val", "test")
    }
    materialization_counts = {split: Counter() for split in stats}
    problems: list[Problem] = [*train_metadata_problems, *val_metadata_problems]
    print(
        "Converting "
        + ", ".join(f"{split}={len(selected[split]):,}" for split in stats)
        + f" with {args.workers} workers ...",
        flush=True,
    )

    def convert(item: WorkItem) -> ConversionResult:
        return _convert_one(
            item,
            images_root=images_root,
            output_root=staging_root,
            parser_function=parser_function,
        )

    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for result in executor.map(convert, work_items):
            completed += 1
            split_stats = stats[result.output_split]
            if result.success:
                split_stats["converted"] += 1
                split_stats["lane_instances"] += result.lane_instances
                split_stats["empty_labels"] += int(result.empty_label)
                split_stats["parser_warnings"] += result.parser_warnings
                materialization_counts[result.output_split][result.materialization] += 1
            else:
                split_stats["failed"] += 1
            problems.extend(result.problems)
            if completed % 1_000 == 0 or completed == len(work_items):
                print(f"  {completed:,}/{len(work_items):,}", flush=True)

    for split in stats:
        stats[split]["materialization"] = dict(
            sorted(materialization_counts[split].items())
        )

    # 임시 폴더 전체를 마지막에 이름만 바꾸므로 data.yaml 에는 최종 경로를 쓴다.
    data_yaml = _write_data_yaml(staging_root, dataset_root=output_root)
    problems.sort(
        key=lambda problem: (
            problem.source_split,
            problem.output_split,
            problem.annotation,
            problem.code,
            problem.object_ids,
        )
    )
    problems_csv = _write_problems_csv(staging_root, problems)
    regression = _regression_report(
        stats,
        official_train_count=len(official_train),
        official_val_count=len(official_val),
        max_files=args.max_files,
        seed=args.seed,
        val_ratio=args.val_ratio,
    )
    source_contract = {
        "official_train": {
            "expected": LEGACY_TRAIN_TOTAL,
            "actual": len(official_train),
            "matches": len(official_train) == LEGACY_TRAIN_TOTAL,
        },
        "official_val_as_test": {
            "expected": EXPECTED_FULL_COUNTS["test"]["images"],
            "actual": len(official_val),
            "matches": len(official_val) == EXPECTED_FULL_COUNTS["test"]["images"],
        },
    }
    source_contract_ok = all(item["matches"] for item in source_contract.values())
    conversion_complete = all(
        int(stats[split]["converted"]) == int(stats[split]["selected"])
        and int(stats[split]["failed"]) == 0
        for split in ("train", "val", "test")
    )
    no_error_problems = not any(problem.severity == "error" for problem in problems)
    regression_ok = not regression["applicable"] or bool(regression["all_match"])
    publication_eligible = (
        source_contract_ok
        and conversion_complete
        and no_error_problems
        and regression_ok
    )
    summary = {
        "schema_version": 4,
        "started_at": started_at,
        "completed_at": _utc_now(),
        "class_names": {"0": "lane_boundary"},
        "config": {
            "images_root": str(images_root),
            "labels_root": str(labels_root),
            "output_root": str(output_root),
            "max_files_per_subset": args.max_files,
            "workers": args.workers,
            "seed": args.seed,
            "val_ratio": args.val_ratio,
            "image_materialization": "same-volume hardlink, copy fallback",
        },
        "source_contract": source_contract,
        "split": split_summary,
        "subsets": stats,
        "problem_counts": dict(sorted(Counter(problem.code for problem in problems).items())),
        "problem_rows": len(problems),
        "regression": regression,
        "parser_calibration": regression["parser_calibration"],
        "membership_sha256": membership_sha256,
        "split_manifest": {
            "path": str(output_root / "split_manifest.csv"),
            "rows": len(work_items),
            "membership_sha256": membership_sha256,
        },
        "publication": {
            "eligible": publication_eligible,
            "status": "validated" if publication_eligible else "validation_failed",
            "staging_root": str(staging_root),
            "checks": {
                "source_contract": source_contract_ok,
                "converted_equals_selected_and_zero_failed": conversion_complete,
                "no_error_problems": no_error_problems,
                "canonical_regression_acceptance": regression_ok,
            },
        },
        "outputs": {
            "data_yaml": str(output_root / data_yaml.name),
            "problems_csv": str(output_root / problems_csv.name),
            "split_manifest": str(output_root / manifest_path.name),
        },
    }
    summary_path = staging_root / "conversion_summary.json"
    _atomic_write_text(
        summary_path,
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )

    print(f"Staging summary: {summary_path}")
    for split in ("train", "val", "test"):
        split_stats = stats[split]
        print(
            f"  {split}: converted={split_stats['converted']:,}, "
            f"failed={split_stats['failed']:,}, "
            f"lanes={split_stats['lane_instances']:,}, "
            f"empty={split_stats['empty_labels']:,}"
        )

    if not source_contract_ok:
        print("Source file counts differ from the official 70k/10k contract.")
        print(f"Existing output was not changed. Failed staging retained: {staging_root}")
        return 2
    if not conversion_complete:
        print(f"Conversion failures were recorded in {problems_csv}.")
        print(f"Existing output was not changed. Failed staging retained: {staging_root}")
        return 1
    if not no_error_problems:
        print(f"Source metadata errors were recorded in {problems_csv}.")
        print(f"Existing output was not changed. Failed staging retained: {staging_root}")
        return 1
    if not regression_ok:
        print(
            "Full conversion failed an acceptance regression: subset image "
            "counts or global lane/empty totals differ from the reference targets."
        )
        print(f"Existing output was not changed. Failed staging retained: {staging_root}")
        return 3

    try:
        retained_backup = _publish_staging(staging_root, output_root)
    except Exception as exc:
        print(f"Dataset publication failed: {type(exc).__name__}: {exc}")
        print(
            "Output was not published. The valid staging tree (or a sibling backup "
            f"if restoration itself failed) remains beside: {staging_root}"
        )
        return 4

    summary["publication"]["status"] = "published"
    summary["publication"]["published_at"] = _utc_now()
    if retained_backup is not None:
        summary["publication"]["retained_backup"] = str(retained_backup)
    published_summary_path = output_root / "conversion_summary.json"
    try:
        _atomic_write_text(
            published_summary_path,
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
    except OSError as exc:
        print(f"Dataset published, but summary status update failed: {exc}")
        print(f"Published dataset retained at: {output_root}")
        return 5
    print(f"Published dataset: {output_root}")
    print(f"Summary: {published_summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
