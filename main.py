"""Load and normalize Cabspotting traces for downstream entropy analysis.

This module currently implements the first concrete step of the project:
discovering raw trace files, parsing each GPS observation, sorting each user
trace into ascending timestamp order, and printing a compact validation
summary.
"""

from __future__ import annotations

import argparse
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


def summarize_dataset(
    result: DatasetLoadResult, grid_size_degrees: float
) -> str:
    """Build a short human-readable summary of the implemented steps.

    Args:
        result: Output from `load_dataset`.
        grid_size_degrees: Spatial bin size used for location assignment.

    Returns:
        A multi-line string summarizing dataset size and one example trace.
    """

    if not result.traces:
        return "No user traces were loaded."

    first_trace = result.traces[0]
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

    return "\n".join(
        [
            f"Users loaded: {len(result.traces)}",
            f"Trace files processed: {result.files_seen}",
            f"Rows loaded: {result.rows_loaded}",
            f"Rows skipped: {result.rows_skipped}",
            f"Grid size (degrees): {grid_size_degrees}",
            f"Unique location cells: {len(unique_locations)}",
            "",
            f"Example user: {first_trace.user_id}",
            f"Example records: {len(first_trace.records)}",
            f"Example time range: {first_start} -> {first_end}",
            f"Example first location cell: {first_location}",
        ]
    )


def main() -> int:
    """Run the current step: load traces and assign discrete locations.

    Returns:
        Process exit code, where `0` indicates the loading step completed
        successfully.
    """

    args = parse_args()
    result = load_dataset(args.data_dir, limit_users=args.limit_users)
    result = assign_locations(result, grid_size_degrees=args.grid_size_degrees)
    print(summarize_dataset(result, grid_size_degrees=args.grid_size_degrees))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
