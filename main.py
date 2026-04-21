"""Load and normalize Cabspotting traces for downstream entropy analysis.

This module currently implements the early data-preparation steps of the
project: discovering raw trace files, parsing each GPS observation, sorting
each user trace into ascending timestamp order, assigning discrete location
cells, and computing time-weighted location probabilities per user.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import math
from dataclasses import dataclass, replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "cabspottingdata"


@dataclass(frozen=True, slots=True)
class TracePoint:
    """One parsed Cabspotting GPS observation for a single user.

    Attributes:
        user_id: Cab identifier derived from the source filename.
        latitude: Latitude in decimal degrees.
        longitude: Longitude in decimal degrees.
        occupancy: Fare status flag from the source data (`0` or `1`).
        timestamp: Observation time in Unix epoch seconds.
        location_id: Optional discrete spatial cell identifier assigned during
            location discretization.
    """

    user_id: str
    latitude: float
    longitude: float
    occupancy: int
    timestamp: int
    location_id: str | None = None


@dataclass(frozen=True, slots=True)
class UserTrace:
    """All normalized observations belonging to one user trace file.

    Attributes:
        user_id: Cab identifier derived from the filename.
        records: Parsed observations sorted by ascending timestamp.
    """

    user_id: str
    records: list[TracePoint]


@dataclass(frozen=True, slots=True)
class DatasetLoadResult:
    """Dataset loading output plus summary statistics for validation.

    Attributes:
        traces: Loaded user traces.
        files_seen: Number of trace files processed.
        rows_loaded: Number of valid rows parsed successfully.
        rows_skipped: Number of malformed rows skipped during parsing.
    """

    traces: list[UserTrace]
    files_seen: int
    rows_loaded: int
    rows_skipped: int


@dataclass(frozen=True, slots=True)
class UserLocationProfile:
    """Time-weighted location distribution for one user.

    Attributes:
        user_id: Cab identifier derived from the filename.
        dwell_seconds_by_location: Observed seconds attributed to each location.
        location_probabilities: Normalized probability mass per location.
        total_observed_seconds: Total valid dwell time used for probabilities.
        transitions_used: Number of record-to-record intervals included.
        transitions_skipped: Number of intervals skipped due to invalid timing or
            gap filtering.
    """

    user_id: str
    dwell_seconds_by_location: dict[str, int]
    location_probabilities: dict[str, float]
    total_observed_seconds: int
    transitions_used: int
    transitions_skipped: int


@dataclass(frozen=True, slots=True)
class ProbabilityComputationResult:
    """Dataset-level output for time-weighted location probabilities.

    Attributes:
        user_profiles: Per-user dwell-time and probability summaries.
        total_observed_seconds: Sum of valid dwell time across all users.
        transitions_used: Number of record-to-record intervals included.
        transitions_skipped: Number of intervals skipped due to invalid timing or
            gap filtering.
    """

    user_profiles: list[UserLocationProfile]
    total_observed_seconds: int
    transitions_used: int
    transitions_skipped: int


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the dataset loading step.

    Returns:
        Parsed command-line arguments containing the dataset location and an
        optional user limit for smaller test runs.
    """

    parser = argparse.ArgumentParser(
        description="Load and normalize Cabspotting traces."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing Cabspotting trace files.",
    )
    parser.add_argument(
        "--limit-users",
        type=int,
        default=None,
        help="Optionally load only the first N user trace files.",
    )
    parser.add_argument(
        "--grid-size-degrees",
        type=float,
        default=0.005,
        help="Latitude/longitude grid size used to assign discrete locations.",
    )
    parser.add_argument(
        "--max-gap-seconds",
        type=int,
        default=1800,
        help="Ignore trace intervals larger than this many seconds. Use 0 to disable.",
    )
    return parser.parse_args()


def iter_trace_files(data_dir: Path) -> list[Path]:
    """Return all Cabspotting trace files in deterministic filename order.

    Args:
        data_dir: Directory containing the raw Cabspotting files.

    Returns:
        A sorted list of trace file paths matching `new_*.txt`.

    Raises:
        FileNotFoundError: If the directory contains no matching trace files.
    """

    trace_files = sorted(data_dir.glob("new_*.txt"))
    if not trace_files:
        raise FileNotFoundError(f"No Cabspotting trace files found in {data_dir}.")
    return trace_files


def parse_trace_line(raw_line: str, user_id: str, file_path: Path, line_number: int) -> TracePoint:
    """Parse one raw trace row into a validated `TracePoint`.

    Args:
        raw_line: Unmodified line content from the trace file.
        user_id: Cab identifier assigned to the current trace file.
        file_path: File currently being parsed, used for error messages.
        line_number: One-based line number within the source file.

    Returns:
        A validated `TracePoint` instance.

    Raises:
        ValueError: If the row does not contain exactly four columns or if the
            occupancy flag is not `0` or `1`.
    """

    parts = raw_line.split()
    if len(parts) != 4:
        raise ValueError(
            f"{file_path.name}:{line_number} expected 4 columns, found {len(parts)}."
        )

    latitude = float(parts[0])
    longitude = float(parts[1])
    occupancy = int(parts[2])
    timestamp = int(parts[3])

    if occupancy not in (0, 1):
        raise ValueError(
            f"{file_path.name}:{line_number} occupancy must be 0 or 1, found {occupancy}."
        )

    return TracePoint(
        user_id=user_id,
        latitude=latitude,
        longitude=longitude,
        occupancy=occupancy,
        timestamp=timestamp,
    )


def load_user_trace(file_path: Path) -> tuple[UserTrace, int]:
    """Load one user file, skip malformed rows, and sort by ascending time.

    Args:
        file_path: Path to one Cabspotting trace file.

    Returns:
        A tuple containing the normalized `UserTrace` and the number of skipped
        malformed rows encountered while parsing that file.
    """

    user_id = file_path.stem.removeprefix("new_")
    records: list[TracePoint] = []
    skipped_rows = 0

    for line_number, raw_line in enumerate(file_path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue

        try:
            record = parse_trace_line(line, user_id, file_path, line_number)
        except ValueError:
            skipped_rows += 1
            continue

        records.append(record)

    records.sort(key=lambda record: record.timestamp)
    return UserTrace(user_id=user_id, records=records), skipped_rows


def load_dataset(data_dir: Path, limit_users: int | None = None) -> DatasetLoadResult:
    """Load multiple user traces and aggregate dataset-level row counts.

    Args:
        data_dir: Directory containing the Cabspotting trace files.
        limit_users: Optional cap on how many user files to load, mainly useful
            for faster local validation runs.

    Returns:
        A `DatasetLoadResult` containing all loaded traces and parsing summary
        counts.
    """

    trace_files = iter_trace_files(data_dir)
    if limit_users is not None:
        trace_files = trace_files[:limit_users]

    traces: list[UserTrace] = []
    rows_loaded = 0
    rows_skipped = 0

    for file_path in trace_files:
        trace, skipped_rows = load_user_trace(file_path)
        traces.append(trace)
        rows_loaded += len(trace.records)
        rows_skipped += skipped_rows

    return DatasetLoadResult(
        traces=traces,
        files_seen=len(trace_files),
        rows_loaded=rows_loaded,
        rows_skipped=rows_skipped,
    )


def build_location_id(latitude: float, longitude: float, grid_size_degrees: float) -> str:
    """Convert one latitude/longitude pair into a deterministic grid cell ID.

    Args:
        latitude: Latitude in decimal degrees.
        longitude: Longitude in decimal degrees.
        grid_size_degrees: Spatial bin size applied to both coordinates.

    Returns:
        A stable string identifier for the corresponding grid cell.

    Raises:
        ValueError: If `grid_size_degrees` is not positive.
    """

    if grid_size_degrees <= 0:
        raise ValueError("grid_size_degrees must be greater than zero.")

    lat_index = math.floor(latitude / grid_size_degrees)
    lon_index = math.floor(longitude / grid_size_degrees)
    return f"grid:{grid_size_degrees:.6f}:{lat_index}:{lon_index}"


def assign_locations_to_trace(
    trace: UserTrace, grid_size_degrees: float
) -> UserTrace:
    """Assign a discrete grid cell ID to every record in one user trace.

    Args:
        trace: One user's time-ordered trace.
        grid_size_degrees: Spatial bin size applied to both coordinates.

    Returns:
        A new `UserTrace` whose records include assigned location IDs.
    """

    records_with_locations = [
        replace(
            record,
            location_id=build_location_id(
                record.latitude, record.longitude, grid_size_degrees
            ),
        )
        for record in trace.records
    ]
    return UserTrace(user_id=trace.user_id, records=records_with_locations)


def assign_locations(
    result: DatasetLoadResult, grid_size_degrees: float
) -> DatasetLoadResult:
    """Assign discrete location IDs across all loaded traces.

    Args:
        result: Output from `load_dataset`.
        grid_size_degrees: Spatial bin size applied to both coordinates.

    Returns:
        A new `DatasetLoadResult` with location IDs assigned to every record.
    """

    traces_with_locations = [
        assign_locations_to_trace(trace, grid_size_degrees) for trace in result.traces
    ]
    return DatasetLoadResult(
        traces=traces_with_locations,
        files_seen=result.files_seen,
        rows_loaded=result.rows_loaded,
        rows_skipped=result.rows_skipped,
    )


def normalize_gap_limit(max_gap_seconds: int) -> int | None:
    """Normalize the CLI gap limit into an internal optional value.

    Args:
        max_gap_seconds: Maximum allowed seconds between consecutive records.

    Returns:
        `None` if gap filtering is disabled, otherwise the positive gap limit.
    """

    return None if max_gap_seconds <= 0 else max_gap_seconds


def compute_trace_location_profile(
    trace: UserTrace, max_gap_seconds: int | None
) -> UserLocationProfile:
    """Compute dwell-time location probabilities for one user trace.

    Args:
        trace: One user's time-ordered trace with assigned location IDs.
        max_gap_seconds: Optional upper bound for valid time gaps. Intervals
            larger than this are skipped.

    Returns:
        A `UserLocationProfile` containing dwell times and normalized
        probabilities for the user.

    Raises:
        ValueError: If a record is missing its assigned `location_id`.
    """

    dwell_seconds_by_location: defaultdict[str, int] = defaultdict(int)
    transitions_used = 0
    transitions_skipped = 0

    for current_record, next_record in zip(trace.records, trace.records[1:]):
        if current_record.location_id is None:
            raise ValueError(
                f"Trace '{trace.user_id}' is missing location IDs. Run location assignment first."
            )

        duration_seconds = next_record.timestamp - current_record.timestamp
        if duration_seconds <= 0:
            transitions_skipped += 1
            continue

        if max_gap_seconds is not None and duration_seconds > max_gap_seconds:
            transitions_skipped += 1
            continue

        dwell_seconds_by_location[current_record.location_id] += duration_seconds
        transitions_used += 1

    total_observed_seconds = sum(dwell_seconds_by_location.values())
    if total_observed_seconds == 0:
        location_probabilities: dict[str, float] = {}
    else:
        location_probabilities = {
            location_id: dwell_seconds / total_observed_seconds
            for location_id, dwell_seconds in dwell_seconds_by_location.items()
        }

    return UserLocationProfile(
        user_id=trace.user_id,
        dwell_seconds_by_location=dict(dwell_seconds_by_location),
        location_probabilities=location_probabilities,
        total_observed_seconds=total_observed_seconds,
        transitions_used=transitions_used,
        transitions_skipped=transitions_skipped,
    )


def compute_location_probabilities(
    result: DatasetLoadResult, max_gap_seconds: int | None
) -> ProbabilityComputationResult:
    """Compute time-weighted location probabilities across all loaded traces.

    Args:
        result: Output from `assign_locations`.
        max_gap_seconds: Optional upper bound for valid time gaps. Intervals
            larger than this are skipped.

    Returns:
        A `ProbabilityComputationResult` containing per-user location
        distributions and aggregate interval counts.
    """

    user_profiles = [
        compute_trace_location_profile(trace, max_gap_seconds)
        for trace in result.traces
    ]
    return ProbabilityComputationResult(
        user_profiles=user_profiles,
        total_observed_seconds=sum(
            profile.total_observed_seconds for profile in user_profiles
        ),
        transitions_used=sum(profile.transitions_used for profile in user_profiles),
        transitions_skipped=sum(
            profile.transitions_skipped for profile in user_profiles
        ),
    )


def summarize_dataset(
    result: DatasetLoadResult,
    probabilities: ProbabilityComputationResult,
    grid_size_degrees: float,
    max_gap_seconds: int | None,
) -> str:
    """Build a short human-readable summary of the implemented steps.

    Args:
        result: Output from `assign_locations`.
        probabilities: Output from `compute_location_probabilities`.
        grid_size_degrees: Spatial bin size used for location assignment.
        max_gap_seconds: Optional upper bound used when filtering large gaps.

    Returns:
        A multi-line string summarizing dataset size and one example trace.
    """

    if not result.traces:
        return "No user traces were loaded."

    first_trace = result.traces[0]
    first_profile = probabilities.user_profiles[0]
    first_start = first_trace.records[0].timestamp if first_trace.records else "n/a"
    first_end = first_trace.records[-1].timestamp if first_trace.records else "n/a"
    first_location = (
        first_trace.records[0].location_id if first_trace.records else "n/a"
    )
    unique_locations = {
        record.location_id
        for trace in result.traces
        for record in trace.records
        if record.location_id is not None
    }
    top_location_share = (
        max(first_profile.location_probabilities.values())
        if first_profile.location_probabilities
        else 0.0
    )

    return "\n".join(
        [
            "Step 3 complete: time-weighted location probabilities",
            f"Users loaded: {len(result.traces)}",
            f"Trace files processed: {result.files_seen}",
            f"Rows loaded: {result.rows_loaded}",
            f"Rows skipped: {result.rows_skipped}",
            f"Grid size (degrees): {grid_size_degrees}",
            f"Max gap filter (seconds): {max_gap_seconds if max_gap_seconds is not None else 'disabled'}",
            f"Unique location cells: {len(unique_locations)}",
            f"Observed seconds used: {probabilities.total_observed_seconds}",
            f"Transitions used: {probabilities.transitions_used}",
            f"Transitions skipped: {probabilities.transitions_skipped}",
            "",
            f"Example user: {first_trace.user_id}",
            f"Example records: {len(first_trace.records)}",
            f"Example time range: {first_start} -> {first_end}",
            f"Example first location cell: {first_location}",
            f"Example observed seconds: {first_profile.total_observed_seconds}",
            f"Example probability locations: {len(first_profile.location_probabilities)}",
            f"Example top location share: {top_location_share:.4f}",
        ]
    )


def main() -> int:
    """Run the current step: load traces and compute location probabilities.

    Returns:
        Process exit code, where `0` indicates the loading step completed
        successfully.
    """

    args = parse_args()
    result = load_dataset(args.data_dir, limit_users=args.limit_users)
    result = assign_locations(result, grid_size_degrees=args.grid_size_degrees)
    max_gap_seconds = normalize_gap_limit(args.max_gap_seconds)
    probabilities = compute_location_probabilities(result, max_gap_seconds)
    print(
        summarize_dataset(
            result,
            probabilities,
            grid_size_degrees=args.grid_size_degrees,
            max_gap_seconds=max_gap_seconds,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
