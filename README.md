# MDP Air Traffic Simulation

## Project Description 
Temporary Flight Restrictions (TFRs) are established to create protected airspace regions surrounding launch trajectory. Commercial aircraft must avoid entering restricted zones, maintain safe separation, and minimize delay and fuel consumption. Traditional rule-based rerouting method relying on deterministic procedures that may not optimally balance safety and efficiency under uncertainty. However, real-world operations involve significant uncertainties, including surveillance noise (e.g. ADS-B inaccuracies in position and velocity, or radar latency), weather disturbances, and unpredictable aircraft intent.

This project is an integrated agent-based air traffic simulation framework that addresses collision avoidance and Temporary Flight Restriction (TFR) avoidance in a single decision-making process. It implements baseline, MDP-based and POMDP-based controllers for handling air traffic conflicts for different traffic densities. 

Real flight tracks are replayed through a [Mesa](https://mesa.readthedocs.io/) agent-based model, and each aircraft can either fly its recorded path (`baseline` mode) or be controlled by one of the trained MDP/POMDP policies. The runner compares conflict counts and TFR breaches across both approaches at increasing traffic densities.



## Requirements

- Python 3.10+
- Dependencies:

```bash
pip install mesa numpy pandas openpyxl
```


## Setup

```bash
git clone https://github.com/karrnidh/MDP_Project.git
cd MDP_Project
git checkout code        # the active code lives on the "code" branch
pip install mesa numpy pandas openpyxl
```

Make sure the flight data CSVs are in a folder (default: `dats/`) and that `TFR_Lat_Lon.xlsx` sits in the project root (or pass an explicit path with `--tfr`).

## Running the simulation

Run every density tier with baseline, MDP, and POMDP modes:

```bash
python scenario_runner_combined.py
```

Useful flags:

```bash
python scenario_runner_combined.py --tier tier_50        # one tier only
python scenario_runner_combined.py --mode mdp_vi          # one mode only
python scenario_runner_combined.py --out results/         # custom output folder
python scenario_runner_combined.py --seed 99              # different random aircraft sample
python scenario_runner_combined.py --list-tiers           # show which callsigns are in each tier, then exit
python scenario_runner_combined.py --data path/to/csvs --tfr path/to/TFR_Lat_Lon.xlsx
```

Density tiers are defined by `SCENARIO_SIZES` near the top of `scenario_runner_combined.py` (currently `50, 75, 90, all`). Aircraft sampling is seeded (`SAMPLE_SEED = 42`) so tiers are reproducible, and each tier is a nested superset of the smaller ones.

### Output

Results are written to the `--out` directory (default `scenario_results_combined/`):

- `summary_table.csv` : one row per tier per mode (conflicts, TFR breaches, timing, etc.)
- `tier_sample_seedN.csv` : which callsigns were sampled into each tier
- `{tier}_{mode}_conflicts.csv` : collision event log
- `{tier}_{mode}_tfr_metrics.csv` : per-aircraft TFR summary
- `{tier}_{mode}_mdp_log.csv` : per-step MDP decision log
- `{tier}_{mode}_uncertainty_log.csv` : POMDP belief uncertainty (POMDP mode only)
- `{tier}_{mode}_metrics.csv` : Mesa `DataCollector` time series
- `{tier}_{mode}_snapshots.csv` : per-agent, per-step position/state snapshots

A console summary table and a validation check against known real-world conflicts (e.g. `JBU1052 ↔ SWA219`) print at the end of the run.
