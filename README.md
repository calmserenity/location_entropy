<p style="color: red;"><strong>Warning:</strong> To view the interactive 3D visualisation, you must run <code>location_entropy_analysis.ipynb</code> locally and execute the Step 6 cells. The 3D Plotly output is not fully visible from the repository files alone.</p>

# Location Entropy Analysis

This repository contains a notebook-based analysis of per-user location entropy on spatio-temporal mobility traces. The project computes entropy from time-weighted location probabilities, exports ranked user results, and generates explanatory visualizations, including an interactive 3D trajectory view.

## Project Goal

The analysis addresses a mobility-entropy assignment: estimate each user's location entropy,

`E = -sum(p(i) * log2(p(i)))`

where `p(i)` is the probability of the user being in location `i`, then interpret the results and suggest product ideas from the behavioral patterns in the data.

## What The Notebook Does

The main workflow lives in `location_entropy_analysis.ipynb` and is organized into six steps:

1. Explore the raw trace schema.
2. Load and normalize user traces.
3. Assign latitude/longitude points to discrete grid cells.
4. Compute time-weighted location probabilities.
5. Calculate per-user Shannon entropy and export ranked results.
6. Generate visualizations for interpretation.

The key modeling choice is to use **time-weighted dwell share** instead of raw GPS point counts. This makes the entropy metric better reflect actual mobility behavior.

## Repository Contents

- `location_entropy_analysis.ipynb`: main analysis notebook
- `outputs/stepwise_location_entropy_results.csv`: exported ranked per-user results
- `question.md`: original assignment brief

## Data Setup

The notebook expects the mobility traces to be placed in:

`cabspottingdata/`

inside the project root, with files matching:

`new_*.txt`

The notebook uses:

- `DATA_DIR = PROJECT_ROOT / "cabspottingdata"`
- `OUTPUT_CSV = PROJECT_ROOT / "outputs" / "stepwise_location_entropy_results.csv"`

## Environment

Use Python 3 with Jupyter Notebook or JupyterLab.

Install the main dependencies:

```bash
pip install pandas matplotlib plotly notebook
```

## How To Run

1. Place the trace files in `cabspottingdata/`.
2. Open `location_entropy_analysis.ipynb` in Jupyter or VS Code/Cursor.
3. Run the notebook from top to bottom.
4. Check the exported file at `outputs/stepwise_location_entropy_results.csv`.

## Important Note On The 3D Visualisation

To see the interactive 3D trajectory visualisation, you must **run the notebook locally** and execute the Step 6 cells. The 3D view is produced inside the notebook with Plotly, so it will not be fully visible from the raw repository files alone.

## Outputs

The notebook produces:

- entropy summary statistics across users
- a ranked CSV of per-user metrics
- an entropy distribution plot
- an entropy vs. number-of-locations plot
- representative GPS density maps
- interactive single-day 3D trajectories for low-, median-, and high-entropy examples
- a segment summary table

## Main Metrics

The exported CSV includes:

- `entropy`
- `normalized_entropy`
- `num_locations`
- `top_location_share`
- `total_observed_seconds`
- `transitions_used`
- `transitions_skipped`

## Findings Summary

The notebook shows that some users are highly routine, with most observed time concentrated in a small number of locations, while others spread their time across many locations and exhibit much higher entropy. The combination of entropy values, dominant-location share, density maps, and 3D trajectories helps explain not just how many places users visit, but how evenly their time is distributed across places and across the day.

## Example Product Ideas

- Mobility segmentation for routine vs. exploratory user archetypes
- Dispatch or demand planning based on corridor and zone regularity
- Personalization or anomaly detection using entropy as a behavioral baseline

## Notes

- Set `LIMIT_USERS = None` to run the full dataset.
- Tune `GRID_SIZE_DEGREES` to make location cells coarser or finer.
- Tune `MAX_GAP_SECONDS` to control how long inactive intervals are counted.
- Tune `TRAJECTORY_POINT_LIMIT` if the trajectory plots look too dense or too sparse.