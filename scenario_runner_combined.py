"""
scenario_runner_combined.py
============================
Owner  : Anaya (MDP / Resolution Lead – Group B)
         + Karrnidh / Dev (Group A integration)
Project: MDP Air Traffic Simulation  |  ITSEC 2026

Scenario Runner — Combined TFR + Collision Avoidance
------------------------------------------------------
Evaluates the COMBINED MDP (TFR avoidance + collision avoidance) across
four density tiers, randomly sampled from all available CSVs:

    Development / validation run  (current):
        TIER_1  —  1 aircraft
        TIER_5  —  5 aircraft
        TIER_10 — 10 aircraft
        TIER_20 — 20 aircraft

    Final paper run (change SCENARIO_SIZES at the top of this file):
        TIER_10  —  10 aircraft
        TIER_50  —  50 aircraft
        TIER_100 — 100 aircraft
        TIER_ALL — all available aircraft (146)

    To switch to the paper tiers, change the four numbers in SCENARIO_SIZES
    and re-run — everything else is automatic.

Aircraft selection uses a fixed random seed (SAMPLE_SEED) so results are
fully reproducible and selection bias is avoided.

For each tier, three runs are executed:
    baseline  — aircraft replay FR24 CSV tracks (no MDP)
    mdp_vi    — Group A VI solver (TFR) + Group B ACAS-X (collision)
    pomdp     — Group A POMDP (TFR) + Group B SARSOP POMDP (collision)

Comparison table shows BOTH safety dimensions side-by-side:
    Conflict events  (Group B)
    TFR breaches     (Group A)

Known real conflicts (confirmed by baseline_analysis.py):
    JBU1052 ↔ SWA219  steps 0-12  min h-sep 1.351 NM

Usage
-----
    python scenario_runner_combined.py                     # all tiers
    python scenario_runner_combined.py --tier tier_5
    python scenario_runner_combined.py --mode mdp_vi
    python scenario_runner_combined.py --out results/
    python scenario_runner_combined.py --seed 99           # different sample
    python scenario_runner_combined.py --list-tiers        # show sampled callsigns
"""

import argparse
import os
import random
import time
import tempfile
import shutil
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from air_traffic_abm import load_flight_data
from air_traffic_model_combined import AirTrafficModelCombined, clear_solver_cache
from mdp_collision_env import ObservationNoise


# =============================================================================
# ── TIER CONFIGURATION ────────────────────────────────────────────────────────
#
#  To switch from development tiers to paper tiers, change these four numbers:
#
#  Development:  SCENARIO_SIZES = (1, 5, 10, 20)
#  Paper:        SCENARIO_SIZES = (10, 50, 100, None)   ← None = all available
#
# =============================================================================

SCENARIO_SIZES = (1, 5, 10, 20)   # ← change these for the paper run

# Fixed seed — guarantees the same aircraft are sampled every run.
# Change with --seed if you want a different random sample.
SAMPLE_SEED = 42

# Known real conflicts (for validation section)
KNOWN_CONFLICTS = [
    ("JBU1052", "SWA219", 0, 1.351),
]


# =============================================================================
# Tier helpers
# =============================================================================

def _tier_name(size: Optional[int]) -> str:
    return f"tier_{size}" if size is not None else "tier_all"


def _tier_label(size: Optional[int], total: int) -> str:
    n = size if size is not None else total
    return f"Density Tier — {n} aircraft"


def build_tiers(
    all_callsigns: List[str],
    sizes: tuple,
    seed: int,
) -> Dict[str, List[str]]:
    """
    Build tier→callsign-list mapping by sampling from all_callsigns.

    Sampling strategy
    -----------------
    - Tiers are nested: tier N is a subset of tier N+1.
      This means aircraft added at each tier are incremental, which makes
      the scaling analysis cleaner (you're always adding aircraft, never
      swapping them out).
    - The largest tier gets all available aircraft if its size >= len(all_callsigns).
    - A fixed seed guarantees reproducibility.

    Example with sizes=(1,5,10,20) and 146 callsigns:
        tier_1  = first 1 of shuffled list
        tier_5  = first 5 of shuffled list  (includes tier_1 aircraft)
        tier_10 = first 10                  (includes tier_5 aircraft)
        tier_20 = first 20                  (includes tier_10 aircraft)
    """
    rng = random.Random(seed)
    shuffled = list(all_callsigns)
    rng.shuffle(shuffled)

    tiers = {}
    for size in sizes:
        name = _tier_name(size)
        if size is None or size >= len(shuffled):
            tiers[name] = shuffled          # all aircraft
        else:
            tiers[name] = shuffled[:size]   # nested subset
    return tiers


# =============================================================================
# Result structures
# =============================================================================

@dataclass
class RunResult:
    """Metrics from one simulation run (one tier, one mode)."""
    scenario             : str
    mode                 : str
    n_aircraft           : int
    steps_run            : int
    # Group B — collision
    total_conflicts      : int
    peak_conflicts       : int
    conflict_events      : List[dict]
    mdp_log              : pd.DataFrame
    model_metrics        : pd.DataFrame
    agent_snapshots      : pd.DataFrame
    pomdp_unc_log        : pd.DataFrame
    # Group A — TFR
    tfr_breaches         : int
    tfr_warnings         : int
    avg_min_tfr_dist_nm  : float
    aircraft_entered_tfr : int
    tfr_metrics_df       : pd.DataFrame
    # Timing
    wall_time_s          : float


@dataclass
class ScenarioResult:
    """Paired baseline + MDP + POMDP results for one tier."""
    scenario : str
    label    : str
    baseline : Optional[RunResult] = None
    mdp      : Optional[RunResult] = None
    pomdp    : Optional[RunResult] = None

    def summary(self) -> dict:
        def _r(run: Optional[RunResult]):
            if run is None:
                return {}
            return {
                "n_aircraft"          : run.n_aircraft,
                "steps"               : run.steps_run,
                "total_conflicts"     : run.total_conflicts,
                "peak_conflicts"      : run.peak_conflicts,
                "tfr_breaches"        : run.tfr_breaches,
                "tfr_warnings"        : run.tfr_warnings,
                "avg_min_tfr_dist_nm" : run.avg_min_tfr_dist_nm,
                "wall_time_s"         : round(run.wall_time_s, 1),
            }
        return {
            "scenario": self.scenario,
            "label"   : self.label,
            "baseline": _r(self.baseline),
            "mdp"     : _r(self.mdp),
            "pomdp"   : _r(self.pomdp),
        }


# =============================================================================
# Core runner
# =============================================================================

def _write_tmp_csvs(aircraft_data: dict, tmp_dir: str):
    """Write filtered per-callsign CSVs so AirTrafficModelCombined can load them."""
    for cs, df in aircraft_data.items():
        out = os.path.join(tmp_dir, f"{cs}.csv")
        df.to_csv(out, index=False)


def run_single(
    aircraft_data : dict,
    scenario      : str,
    mode          : str,
    tfr_path      : Optional[str],
) -> RunResult:
    """Execute one simulation run and collect all results."""
    t0      = time.perf_counter()
    tmp_dir = tempfile.mkdtemp(prefix=f"combined_{scenario}_{mode}_")

    try:
        _write_tmp_csvs(aircraft_data, tmp_dir)

        pomdp_noise = ObservationNoise(
            sigma_pos_nm=0.05, sigma_alt_ft=75.0,
            sigma_spd_kt=5.0,  sigma_hdg_deg=1.5, p_delay=0.10,
        ) if mode == "pomdp" else None

        model = AirTrafficModelCombined(
            data_dir      = tmp_dir,
            tfr_path      = tfr_path,
            mode          = mode,
            time_step_s   = 30,
            pomdp_noise   = pomdp_noise,
            train_solvers = (mode not in ("baseline",)),
        )
        model.run(model.max_steps)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    wall_time = time.perf_counter() - t0

    # ── Collect Group B outputs ───────────────────────────────────────────
    det_log       = model.detector.get_event_log()
    model_metrics = model.get_model_metrics()
    snapshots     = model.get_snapshots()
    mdp_log       = model.get_mdp_logs()
    pomdp_unc_log = model.get_pomdp_uncertainty_logs()

    total_conflicts = len(det_log) if not det_log.empty else 0
    peak_conflicts  = int(model_metrics["Total Conflicts"].max()) \
                      if not model_metrics.empty else 0

    # ── Collect Group A outputs ───────────────────────────────────────────
    combined_summary = model.get_combined_summary()
    tfr_df           = model.get_tfr_metrics()

    return RunResult(
        scenario             = scenario,
        mode                 = mode,
        n_aircraft           = len(aircraft_data),
        steps_run            = model.current_step,
        total_conflicts      = total_conflicts,
        peak_conflicts       = peak_conflicts,
        conflict_events      = det_log.to_dict("records") if not det_log.empty else [],
        mdp_log              = mdp_log,
        model_metrics        = model_metrics,
        agent_snapshots      = snapshots,
        pomdp_unc_log        = pomdp_unc_log,
        tfr_breaches         = combined_summary["tfr_breaches"],
        tfr_warnings         = combined_summary["tfr_warnings"],
        avg_min_tfr_dist_nm  = combined_summary["avg_min_tfr_dist_nm"],
        aircraft_entered_tfr = combined_summary["aircraft_entered_tfr"],
        tfr_metrics_df       = tfr_df,
        wall_time_s          = wall_time,
    )


def _filter_data(all_data: dict, callsigns: List[str]) -> dict:
    filtered = {cs: df for cs, df in all_data.items() if cs in callsigns}
    missing  = [cs for cs in callsigns if cs not in all_data]
    if missing:
        print(f"  ⚠  Callsigns not found in data: {missing}")
    return filtered


def run_scenario(
    scenario      : str,
    callsigns     : List[str],
    all_data      : dict,
    tfr_path      : Optional[str],
    run_baseline  : bool = True,
    run_mdp       : bool = True,
    run_pomdp     : bool = False,
    mdp_mode      : str  = "mdp_vi",
) -> ScenarioResult:
    aircraft_data = _filter_data(all_data, callsigns)
    label         = _tier_label(len(callsigns), len(all_data))

    print(f"\n{'─'*62}")
    print(f"  Tier     : {label}")
    print(f"  Aircraft : {list(aircraft_data.keys())}")
    print(f"{'─'*62}")

    result = ScenarioResult(scenario=scenario, label=label)

    if run_baseline:
        print("  [BASELINE] running …")
        result.baseline = run_single(aircraft_data, scenario, "baseline", tfr_path)
        b = result.baseline
        print(f"  [BASELINE] done — {b.steps_run} steps | "
              f"conflicts={b.total_conflicts} | "
              f"TFR breaches={b.tfr_breaches} | "
              f"{b.wall_time_s:.1f}s")

    if run_mdp:
        print(f"  [{mdp_mode.upper()}] running …")
        result.mdp = run_single(aircraft_data, scenario, mdp_mode, tfr_path)
        m = result.mdp
        print(f"  [{mdp_mode.upper()}] done — {m.steps_run} steps | "
              f"conflicts={m.total_conflicts} | "
              f"TFR breaches={m.tfr_breaches} | "
              f"{m.wall_time_s:.1f}s")
        if run_baseline and result.baseline:
            b = result.baseline
            _print_reduction("conflict",  b.total_conflicts, m.total_conflicts)
            _print_reduction("TFR breach", b.tfr_breaches,   m.tfr_breaches)

    if run_pomdp:
        print("  [POMDP] running …")
        result.pomdp = run_single(aircraft_data, scenario, "pomdp", tfr_path)
        p = result.pomdp
        print(f"  [POMDP] done — {p.steps_run} steps | "
              f"conflicts={p.total_conflicts} | "
              f"TFR breaches={p.tfr_breaches} | "
              f"{p.wall_time_s:.1f}s")
        if run_baseline and result.baseline:
            b = result.baseline
            _print_reduction("conflict",  b.total_conflicts, p.total_conflicts)
            _print_reduction("TFR breach", b.tfr_breaches,   p.tfr_breaches)

    return result


def _print_reduction(label: str, before: int, after: int):
    delta = before - after
    pct   = 100.0 * delta / max(before, 1)
    arrow = "↓" if delta >= 0 else "↑"
    print(f"  → {label} reduction : {delta:+d} ({pct:+.0f}%) {arrow}")


# =============================================================================
# Output helpers
# =============================================================================

def print_summary_table(results: List[ScenarioResult]) -> None:
    """Print side-by-side comparison table covering both safety dimensions."""
    W = 84
    print("\n" + "=" * W)
    print(f"  {'TIER':<26} {'MODE':<10} {'AC':>4} {'STEPS':>5} "
          f"{'COLL':>6} {'PEAK':>5} {'TFR-B':>6} {'TFR-W':>6} "
          f"{'MIN-TFR':>8} {'TIME':>7}")
    print("-" * W)
    print(f"  {'':26} {'':10} {'':4} {'':5} "
          f"{'evts':>6} {'sim':>5} {'brch':>6} {'warn':>6} "
          f"{'nm':>8} {'s':>7}")
    print("=" * W)

    for sr in results:
        first = True
        for run in [sr.baseline, sr.mdp, sr.pomdp]:
            if run is None:
                continue
            label = sr.label if first else ""
            first = False
            tfr_d = (f"{run.avg_min_tfr_dist_nm:.1f}"
                     if run.avg_min_tfr_dist_nm < 1e6 else "  N/A")
            print(
                f"  {label:<26} {run.mode:<10} {run.n_aircraft:>4} "
                f"{run.steps_run:>5} {run.total_conflicts:>6} "
                f"{run.peak_conflicts:>5} {run.tfr_breaches:>6} "
                f"{run.tfr_warnings:>6} {tfr_d:>8} "
                f"{run.wall_time_s:>6.1f}s"
            )

        # Reduction rows
        for tag, run_after in [("MDP", sr.mdp), ("POMDP", sr.pomdp)]:
            if sr.baseline and run_after:
                b, a = sr.baseline, run_after
                dc  = b.total_conflicts - a.total_conflicts
                dt  = b.tfr_breaches   - a.tfr_breaches
                pcc = 100.0 * dc / max(b.total_conflicts, 1)
                ptc = 100.0 * dt / max(b.tfr_breaches, 1)
                print(f"  {'':26} → {tag} saved    {'':4} {'':5} "
                      f"{dc:>+6} {'':5} {dt:>+6} {'':6} {'':8}")
                print(f"  {'':26}            {'':4} {'':5} "
                      f"{pcc:>+5.0f}% {'':5} {ptc:>+5.0f}%")
        print()

    print("=" * W)
    print("  COLL=collision events  PEAK=peak simultaneous  TFR-B=TFR breach steps")
    print("  TFR-W=TFR warning steps  MIN-TFR=avg min distance to TFR edge")


def validate_known_conflicts(
    results       : List[ScenarioResult],
    tier_callsigns: Dict[str, List[str]],
) -> None:
    print("\n── Validation: known conflict detection ──────────────────────────────")
    for sr in results:
        if sr.baseline is None or not sr.baseline.conflict_events:
            continue
        det = pd.DataFrame(sr.baseline.conflict_events)
        scenario_cs = tier_callsigns.get(sr.scenario, [])
        for cs_a, cs_b, exp_step, exp_sep in KNOWN_CONFLICTS:
            if cs_a not in scenario_cs or cs_b not in scenario_cs:
                continue
            if det.empty or "callsign_a" not in det.columns:
                found = pd.DataFrame()
            else:
                found = det[
                    ((det["callsign_a"] == cs_a) & (det["callsign_b"] == cs_b)) |
                    ((det["callsign_a"] == cs_b) & (det["callsign_b"] == cs_a))
                ]
            if not found.empty:
                r = found.iloc[0]
                print(f"  ✓  {sr.scenario:>8}  {cs_a} ↔ {cs_b:<10}  "
                      f"detected step {r.get('step','?')} "
                      f"({r.get('h_sep_nm',0):.3f} NM)  "
                      f"[confirmed: step {exp_step}, {exp_sep} NM]")
            else:
                print(f"  ✗  {sr.scenario:>8}  {cs_a} ↔ {cs_b:<10}  "
                      f"NOT DETECTED  [expected step {exp_step}]")


def save_results(
    results        : List[ScenarioResult],
    out_dir        : str,
    tier_callsigns : Dict[str, List[str]],
    seed           : int,
) -> None:
    """
    Save all results to out_dir.

    Files produced
    --------------
    summary_table.csv                      — one row per tier per mode
    tier_sample_seed{N}.csv               — which callsigns were in each tier
    {tier}_{mode}_conflicts.csv            — collision event log
    {tier}_{mode}_tfr_metrics.csv          — per-aircraft TFR summary
    {tier}_{mode}_mdp_log.csv              — per-step MDP log
    {tier}_{mode}_uncertainty_log.csv      — POMDP belief uncertainty
    {tier}_{mode}_metrics.csv              — Mesa DataCollector time-series
    {tier}_{mode}_snapshots.csv            — per-agent per-step snapshots
    """
    os.makedirs(out_dir, exist_ok=True)
    summary_rows = []

    # Save tier composition for reproducibility
    tier_rows = [
        {"tier": tier, "callsign": cs, "seed": seed}
        for tier, callsigns in tier_callsigns.items()
        for cs in callsigns
    ]
    pd.DataFrame(tier_rows).to_csv(
        os.path.join(out_dir, f"tier_sample_seed{seed}.csv"), index=False)

    for sr in results:
        for run in [sr.baseline, sr.mdp, sr.pomdp]:
            if run is None:
                continue
            prefix = f"{run.scenario}_{run.mode}"

            if run.conflict_events:
                pd.DataFrame(run.conflict_events).to_csv(
                    os.path.join(out_dir, f"{prefix}_conflicts.csv"), index=False)

            if not run.tfr_metrics_df.empty:
                run.tfr_metrics_df.to_csv(
                    os.path.join(out_dir, f"{prefix}_tfr_metrics.csv"), index=False)

            if not run.mdp_log.empty:
                run.mdp_log.to_csv(
                    os.path.join(out_dir, f"{prefix}_mdp_log.csv"), index=False)

            if not run.pomdp_unc_log.empty:
                run.pomdp_unc_log.to_csv(
                    os.path.join(out_dir, f"{prefix}_uncertainty_log.csv"), index=False)

            if not run.model_metrics.empty:
                run.model_metrics.to_csv(
                    os.path.join(out_dir, f"{prefix}_metrics.csv"), index=False)

            if not run.agent_snapshots.empty:
                run.agent_snapshots.to_csv(
                    os.path.join(out_dir, f"{prefix}_snapshots.csv"), index=False)

            summary_rows.append({
                "scenario"            : run.scenario,
                "label"               : sr.label,
                "mode"                : run.mode,
                "n_aircraft"          : run.n_aircraft,
                "steps_run"           : run.steps_run,
                "total_conflicts"     : run.total_conflicts,
                "peak_conflicts"      : run.peak_conflicts,
                "tfr_breaches"        : run.tfr_breaches,
                "tfr_warnings"        : run.tfr_warnings,
                "avg_min_tfr_dist_nm" : run.avg_min_tfr_dist_nm,
                "aircraft_entered_tfr": run.aircraft_entered_tfr,
                "wall_time_s"         : round(run.wall_time_s, 1),
                "sample_seed"         : seed,
            })

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(
            os.path.join(out_dir, "summary_table.csv"), index=False)
        print(f"\n  Results saved → {out_dir}/")


# =============================================================================
# Entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Combined TFR + Collision Avoidance Scenario Runner")
    parser.add_argument("--data", default=None,
        help="dats CSV folder")
    parser.add_argument("--tfr", default=None,
        help="TFR_Lat_Lon.xlsx path. Omit to disable TFR avoidance.")
    parser.add_argument("--tier", default="all",
        help="Which tier to run: all | tier_1 | tier_5 | tier_10 | tier_20 "
             "(or whatever SCENARIO_SIZES contains). Default: all")
    parser.add_argument("--mode", default="all",
        choices=["all", "baseline", "mdp_vi", "mdp_ql", "pomdp",
                 "mdp_only", "pomdp_only"])
    parser.add_argument("--seed", type=int, default=SAMPLE_SEED,
        help=f"Random seed for aircraft sampling (default: {SAMPLE_SEED})")
    parser.add_argument("--out", default="scenario_results_combined",
        help="Output directory (default: scenario_results_combined/)")
    parser.add_argument("--list-tiers", action="store_true",
        help="Print the sampled callsigns for each tier and exit")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Locate data directory
    _data_candidates = [
        os.path.join(script_dir, "dats"),
    ]
    data_dir = args.data or next(
        (p for p in _data_candidates if os.path.isdir(p)), _data_candidates[0])

    if not os.path.isdir(data_dir):
        raise SystemExit(
            f"\n  ERROR: Data directory not found: {data_dir}\n"
            "  Pass an explicit path with --data <dir>"
        )

    # Locate TFR file
    _tfr_candidates = [
        os.path.join(script_dir, "TFR_Lat_Lon.xlsx"),
        args.tfr or "",
    ]
    tfr_path = args.tfr or next(
        (p for p in _tfr_candidates if os.path.isfile(p)), None)

    # Load all flight data once
    print("  Loading flight data …")
    _df = load_flight_data(data_dir)
    if _df.empty:
        raise SystemExit(f"No CSV data found in {data_dir}")
    all_data = {cs: grp.reset_index(drop=True)
                for cs, grp in _df.groupby("Callsign")}
    all_callsigns = sorted(all_data.keys())

    # Build tier→callsign mapping
    tier_callsigns = build_tiers(all_callsigns, SCENARIO_SIZES, args.seed)

    # --list-tiers: just print composition and exit
    if args.list_tiers:
        print(f"\n  Tier composition (seed={args.seed}):")
        for tier, cs_list in tier_callsigns.items():
            print(f"    {tier:<12} ({len(cs_list):>3} aircraft): {cs_list}")
        return

    print("=" * 64)
    print("  Combined TFR + Collision Avoidance Scenario Runner")
    print("  Group A (TFR) + Group B (Collision)")
    print("=" * 64)
    print(f"  Data dir   : {data_dir}")
    print(f"  TFR file   : {tfr_path or 'NOT FOUND — TFR avoidance disabled'}")
    print(f"  Tiers      : {list(tier_callsigns.keys())}  "
          f"(sizes {SCENARIO_SIZES})")
    print(f"  Sample seed: {args.seed}  "
          f"(change with --seed N for a different random sample)")
    print(f"  Total CSVs : {len(all_callsigns)} aircraft available")
    print(f"  Mode       : {args.mode}")
    print(f"  Output     : {args.out}/")
    print()
    print("  To switch to paper tiers (10/50/100/146), set in source:")
    print("      SCENARIO_SIZES = (10, 50, 100, None)")
    print()
    print("  Solvers trained once and cached across tiers (Fix 2).")

    # Clear solver cache at startup — retrain fresh each process invocation,
    # but share across tiers within this run.
    clear_solver_cache()

    # Select tiers and modes
    if args.tier == "all":
        tiers_to_run = list(tier_callsigns.keys())
    else:
        if args.tier not in tier_callsigns:
            raise SystemExit(
                f"\n  ERROR: Unknown tier '{args.tier}'. "
                f"Available: {list(tier_callsigns.keys())}"
            )
        tiers_to_run = [args.tier]

    run_baseline = args.mode in ("all", "baseline", "mdp_only", "pomdp_only")
    run_mdp      = args.mode in ("all", "mdp_vi", "mdp_ql", "mdp_only")
    run_pomdp    = args.mode in ("all", "pomdp", "pomdp_only")
    mdp_mode_str = "mdp_ql" if args.mode == "mdp_ql" else "mdp_vi"

    results = []
    for tier in tiers_to_run:
        sr = run_scenario(
            scenario     = tier,
            callsigns    = tier_callsigns[tier],
            all_data     = all_data,
            tfr_path     = tfr_path,
            run_baseline = run_baseline,
            run_mdp      = run_mdp,
            run_pomdp    = run_pomdp,
            mdp_mode     = mdp_mode_str,
        )
        results.append(sr)

    print_summary_table(results)
    validate_known_conflicts(results, tier_callsigns)
    save_results(results, args.out, tier_callsigns, args.seed)


if __name__ == "__main__":
    main()