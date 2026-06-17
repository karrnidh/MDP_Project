"""
air_traffic_model_combined.py
==============================
Merged AirTrafficModel — Group A (TFR Avoidance) + Group B (Collision Avoidance)

Replaces:
  - AirTrafficModel in air_traffic_abm.py  (Group B / Mesa)
  - AirTrafficModel in mdp.py              (Group A / standalone)

Authors:
  Dev / Karrnidh  — Group A: TFR loading, VI/QL/POMDP-A solvers, TFR metrics
  Yuvi / Anaya    — Group B: Mesa, ConflictDetector, CollisionMDP, DataCollector

Usage
-----
    from air_traffic_model_combined import AirTrafficModelCombined

    model = AirTrafficModelCombined(
        data_dir   = "flight_data/",
        tfr_path   = "TFR_Lat_Lon.xlsx",   # Group A — pass None to skip TFR
        mode       = "mdp_vi",              # baseline | mdp_vi | mdp_ql | pomdp
        time_step_s= 30,
    )
    model.run(model.max_steps)

    # Group B outputs (unchanged API)
    model.get_model_metrics()
    model.get_snapshots()
    model.detector.get_event_log()
    model.get_mdp_logs()

    # Group A outputs (new)
    model.get_tfr_metrics()
    model.get_combined_summary()
"""

from __future__ import annotations

import os
from typing import List, Optional

import mesa
import numpy as np
import pandas as pd

# ── Group B imports ───────────────────────────────────────────────────────────
from conflict_detection import (
    ConflictDetector,
    HORIZONTAL_SEP_NM,
    VERTICAL_SEP_FT,
)
from mdp_collision_env import CollisionMDP, CollisionPOMDP, ObservationNoise

# ── Group A imports ───────────────────────────────────────────────────────────
# These are imported lazily inside methods that need them so the model can
# still run in baseline mode even if mdp.py is not on the path.
# Geometry helpers needed at init time are inlined below.

# ── Merged agent ──────────────────────────────────────────────────────────────
from aircraft_agent_combined import AircraftAgent

# ── Data loading (use Group B's loader as canonical) ─────────────────────────
from air_traffic_abm import load_flight_data


# =============================================================================
# Module-level solver cache — keyed by (tfr_path, mode)
# Solvers are trained once and reused across scenarios so VI + QL
# don't re-train from scratch for every density run (high/medium/single).
# =============================================================================

_SOLVER_CACHE: dict = {}   # key: (tfr_path, mode) → (vi, ql, pomdp_a)


def clear_solver_cache() -> None:
    """Evict all cached Group A solvers (call between independent experiments)."""
    _SOLVER_CACHE.clear()
    print("  Solver cache cleared.")


# =============================================================================
# AirTrafficModelCombined
# =============================================================================

class AirTrafficModelCombined(mesa.Model):
    """
    Mesa model that runs real air traffic in four modes, handling both
    TFR avoidance (Group A) and collision avoidance (Group B).

    Parameters
    ----------
    data_dir    : path to FlightRadar24 CSV folder
    tfr_path    : path to TFR_Lat_Lon.xlsx  (pass None to disable TFR)
    mode        : 'baseline' | 'mdp_vi' | 'mdp_ql' | 'pomdp'
    time_step_s : seconds per simulation step (default 30)
    pomdp_noise : ObservationNoise for POMDP mode (uses defaults if None)
    train_solvers: if True, train Group A VI + QL solvers on init (default True)
                  Set False to skip training when TFR is disabled or for speed.

    Shared objects
    --------------
    self.detector    : ConflictDetector — KD-tree, shared across all agents
    self.tfr         : TFRZone | None   — Group A TFR polygon
    self.vi_solver   : ValueIterationSolver | None
    self.ql_solver   : QLearningSolver | None
    self.pomdp_solver_a : POMDPSolver | None   (Group A POMDP)

    Output API
    ----------
    model.get_model_metrics()      → DataCollector time-series (Group B, extended)
    model.get_snapshots()          → per-agent per-step DataFrame
    model.detector.get_event_log() → ConflictEvent log
    model.get_mdp_logs()           → per-agent MDP step log (MDP modes only)
    model.get_tfr_metrics()        → per-agent TFR summary (Group A)
    model.get_combined_summary()   → side-by-side safety comparison dict
    """

    # ── Valid modes ──────────────────────────────────────────────────────────
    VALID_MODES = ("baseline", "mdp_vi", "mdp_ql", "pomdp")

    def __init__(
        self,
        data_dir      : str,
        tfr_path      : Optional[str] = None,
        mode          : str = "baseline",
        time_step_s   : int = 30,
        pomdp_noise   : Optional[ObservationNoise] = None,
        train_solvers : bool = True,
    ):
        super().__init__()

        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {self.VALID_MODES}, got {mode!r}")

        self.mode         = mode
        self.time_step_s  = time_step_s
        self.current_step = 0

        # ── Group B: shared ConflictDetector ──────────────────────────────
        self.detector = ConflictDetector(
            h_sep_nm = HORIZONTAL_SEP_NM,
            v_sep_ft = VERTICAL_SEP_FT,
        )

        # ── Group A: TFR zone ─────────────────────────────────────────────
        self.tfr = None
        if tfr_path is not None:
            self.tfr = self._load_tfr(tfr_path)

        # ── Group A: solvers (trained before agents are created) ──────────
        self.vi_solver      = None
        self.ql_solver      = None
        self.pomdp_solver_a = None

        if mode != "baseline" and self.tfr is not None and train_solvers:
            self._train_group_a_solvers()

        # ── Load flight data ──────────────────────────────────────────────
        df = load_flight_data(data_dir)
        self.callsigns = sorted(df["Callsign"].unique())

        # ── Create merged agents ──────────────────────────────────────────
        self.aircraft: List[AircraftAgent] = []
        for cs in self.callsigns:
            traj  = df[df["Callsign"] == cs].copy()
            agent = AircraftAgent(self, cs, traj, mode=mode)
            self.aircraft.append(agent)

        # ── Attach Group B MDPs (non-baseline modes) ──────────────────────
        if mode != "baseline":
            self._attach_collision_mdps(pomdp_noise)

        # ── DataCollector (Group B API, extended with TFR columns) ────────
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Active Flights"      : lambda m: sum(1 for a in m.aircraft if a.active),
                "Total Conflicts"     : lambda m: len(m.detector.get_active_conflicts()),
                "Flights Airborne"    : lambda m: sum(
                    1 for a in m.aircraft if a.active and a.altitude > 500
                ),
                "Avg Altitude (ft)"   : lambda m: (
                    np.mean([a.altitude for a in m.aircraft
                             if a.active and a.altitude > 0])
                    if any(a.active and a.altitude > 0 for a in m.aircraft) else 0.0
                ),
                "Avg Speed (kts)"     : lambda m: (
                    np.mean([a.speed for a in m.aircraft
                             if a.active and a.speed > 0])
                    if any(a.active and a.speed > 0 for a in m.aircraft) else 0.0
                ),
                # Group A additions
                "In TFR"              : lambda m: sum(
                    1 for a in m.aircraft if a.active and a.in_tfr
                ),
                "TFR Warnings"        : lambda m: sum(
                    1 for a in m.aircraft
                    if a.active and m.tfr is not None
                    and 0 < m.tfr.distance_to_edge(a.lat, a.lon) < 10.0
                ),
                "Mode"                : lambda m: m.mode,
            },
            agent_reporters={
                "Callsign"        : "callsign",
                "Lat"             : "lat",
                "Lon"             : "lon",
                "Altitude"        : "altitude",
                "Speed"           : "speed",
                "Heading"         : "heading",
                "Phase"           : "phase",
                "Active"          : "active",
                "Conflicts"       : "n_conflicts",
                "Progress"        : "progress_pct",
                # Group A additions
                "InTFR"           : "in_tfr",
                "MinTFRDist"      : "min_tfr_distance",
            },
        )
        self.datacollector.collect(self)

    # =========================================================================
    # Group A: TFR loading
    # =========================================================================

    @staticmethod
    def _load_tfr(tfr_path: str):
        """Load TFR polygon from Excel — mirrors mdp.py::load_tfr_from_excel()."""
        try:
            from mdp import load_tfr_from_excel
            tfr = load_tfr_from_excel(tfr_path)
            print(f"  TFR loaded: {tfr.n_vertices} vertices, "
                  f"centroid ({tfr.centroid_lat:.3f}, {tfr.centroid_lon:.3f}), "
                  f"mode={tfr.mode}")
            return tfr
        except Exception as exc:
            print(f"  ! TFR load failed ({exc}); TFR avoidance disabled.")
            return None

    # =========================================================================
    # Group A: solver training
    # =========================================================================

    def _train_group_a_solvers(self):
        """Train VI and QL solvers (Group A), or reuse cached ones.

        Solvers are stored in _SOLVER_CACHE keyed by (tfr_path, mode) so that
        the expensive training (VI + QL) only runs once per unique TFR file,
        regardless of how many scenarios or density levels are evaluated.
        """
        try:
            from mdp import ValueIterationSolver, QLearningSolver, POMDPSolver
        except ImportError:
            print("  ! mdp.py not found; Group A solvers disabled.")
            return

        # Build a stable cache key from TFR geometry only — NOT mode.
        # VI and QL solvers are identical for mdp_vi and pomdp; only the
        # belief update layer differs. Keying by mode caused POMDP to retrain
        # from scratch even when mdp_vi had already trained for the same TFR.
        if self.tfr is not None:
            cache_key = (
                self.tfr.n_vertices,
                round(self.tfr.centroid_lat, 4),
                round(self.tfr.centroid_lon, 4),
            )
        else:
            cache_key = (None,)

        if cache_key in _SOLVER_CACHE:
            vi, ql, pomdp_a = _SOLVER_CACHE[cache_key]
            print(f"  ✓ Reusing cached Group A solvers (key={cache_key})")
            self.vi_solver      = vi
            self.ql_solver      = ql
            self.pomdp_solver_a = pomdp_a
            return

        print("  Training Group A — Value Iteration …")
        vi = ValueIterationSolver(self.tfr)
        vi.build_model()
        vi.solve()

        print("  Training Group A — Q-Learning …")
        ql = QLearningSolver(self.tfr)
        ql.train()

        print("  Group A — POMDP solver ready (uses VI policy + belief state).")
        pomdp_a = POMDPSolver(vi, self.tfr)

        _SOLVER_CACHE[cache_key] = (vi, ql, pomdp_a)
        self.vi_solver      = vi
        self.ql_solver      = ql
        self.pomdp_solver_a = pomdp_a

    # =========================================================================
    # Group B: CollisionMDP attachment
    # =========================================================================

    def _attach_collision_mdps(self, pomdp_noise: Optional[ObservationNoise]):
        """Create and reset a CollisionMDP (or POMDP) for every agent."""
        use_pomdp = (self.mode == "pomdp")
        noise = pomdp_noise or ObservationNoise(
            sigma_pos_nm=0.05, sigma_alt_ft=75.0,
            sigma_spd_kt=5.0, sigma_hdg_deg=1.5, p_delay=0.10,
        )

        for agent in self.aircraft:
            if use_pomdp:
                agent.mdp = CollisionPOMDP(
                    detector         = self.detector,
                    ownship_callsign = agent.callsign,
                    max_steps        = self.max_steps,
                    noise            = noise,
                    n_particles      = 100,
                    n_scenarios      = 15,
                    lookahead        = 2,
                )
            else:
                agent.mdp = CollisionMDP(
                    detector         = self.detector,
                    ownship_callsign = agent.callsign,
                    max_steps        = self.max_steps,
                )

        # Reset all MDPs after all agents exist
        # (so _build_intruder_features can see the full aircraft list)
        for agent in self.aircraft:
            agent.mdp.reset(agent, self.aircraft)

    # =========================================================================
    # Mesa step
    # =========================================================================

    def step(self):
        """
        Advance all agents by one step.

        Order of operations
        -------------------
        1. Each agent steps.
           - In baseline mode: CSV replay + agent-level conflict check.
           - In MDP modes: CSV position is loaded first, then
             AircraftAgent._step_policy() queries both Group A and B solvers
             and delegates kinematics to CollisionMDP.run_step() when active.
        2. In baseline mode: authoritative ConflictDetector.check() runs once.
           In MDP modes: check() is called per-agent inside CollisionMDP.run_step().
        3. DataCollector records the current state.
        """
        for agent in self.aircraft:
            agent.step()

        if self.mode == "baseline":
            self.detector.check(
                step          = self.current_step,
                aircraft_list = self.aircraft,
            )

        self.current_step += 1
        self.datacollector.collect(self)

    def run(self, steps: int) -> "AirTrafficModelCombined":
        """Run the model for up to `steps` steps."""
        for _ in range(steps):
            if not any(a.active for a in self.aircraft):
                break
            self.step()
        return self

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def max_steps(self) -> int:
        """Maximum trajectory length across all agents."""
        return max(len(a._traj) for a in self.aircraft)

    # =========================================================================
    # Output API — Group B (unchanged)
    # =========================================================================

    def get_snapshots(self) -> pd.DataFrame:
        """Agent-level data across all steps (from DataCollector)."""
        return self.datacollector.get_agent_vars_dataframe().reset_index()

    def get_model_metrics(self) -> pd.DataFrame:
        """Model-level metrics across all steps (from DataCollector)."""
        return self.datacollector.get_model_vars_dataframe().reset_index()

    def get_mdp_logs(self) -> pd.DataFrame:
        """
        Concatenated per-step CollisionMDP logs (MDP modes only).
        Returns empty DataFrame in baseline mode.
        """
        if self.mode == "baseline":
            return pd.DataFrame()
        frames = []
        for agent in self.aircraft:
            if agent.mdp is not None:
                df = agent.mdp.get_step_log()
                if not df.empty:
                    frames.append(df)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def get_pomdp_uncertainty_logs(self) -> pd.DataFrame:
        """
        Per-step POMDP belief uncertainty logs (POMDP mode only).
        Returns empty DataFrame otherwise.
        """
        if self.mode != "pomdp":
            return pd.DataFrame()
        frames = []
        for agent in self.aircraft:
            if agent.mdp is not None and hasattr(agent.mdp, "get_uncertainty_log"):
                df = agent.mdp.get_uncertainty_log()
                if not df.empty:
                    df = df.copy()
                    df.insert(0, "callsign", agent.callsign)
                    frames.append(df)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def reset_detector(self):
        """Clear the shared ConflictDetector between runs."""
        self.detector.reset()

    # =========================================================================
    # Output API — Group A (new)
    # =========================================================================

    def get_tfr_metrics(self) -> pd.DataFrame:
        """
        Per-aircraft TFR summary.

        Columns
        -------
        callsign          : aircraft identifier
        steps_in_tfr      : number of steps spent inside TFR polygon
        tfr_entries       : number of times the aircraft entered the TFR
        min_tfr_dist_nm   : closest approach to TFR edge (nm; negative = inside)
        avg_tfr_dist_nm   : average TFR distance over the whole flight
        tfr_warnings      : steps within TFR_WARNING_BUFFER_NM (10 nm)
        """
        rows = []
        for agent in self.aircraft:
            dists     = agent.history_tfr_distance
            in_tfr_h  = agent.history_in_tfr

            # Count entries (False→True transitions)
            entries = sum(
                1 for i in range(1, len(in_tfr_h)) if in_tfr_h[i] and not in_tfr_h[i - 1]
            )
            rows.append({
                "callsign"        : agent.callsign,
                "steps_in_tfr"    : sum(in_tfr_h),
                "tfr_entries"     : entries,
                "min_tfr_dist_nm" : round(min(dists), 3) if dists else float("inf"),
                "avg_tfr_dist_nm" : round(float(np.mean(dists)), 3) if dists else float("inf"),
                "tfr_warnings"    : sum(1 for d in dists if 0 < d < 10.0),
            })
        return pd.DataFrame(rows)

    def get_combined_summary(self) -> dict:
        """
        Side-by-side safety summary covering both Group A (TFR) and
        Group B (collision) metrics. Suitable for the comparison table
        in scenario_runner.py.

        Returns
        -------
        dict with keys:
            mode, aircraft_count, steps,
            # Group B
            total_conflict_events, peak_simultaneous_conflicts,
            # Group A
            tfr_breaches, tfr_warnings, avg_min_tfr_dist_nm,
            aircraft_entered_tfr,
            # Combined
            any_safety_violation  (True if either type occurred)
        """
        det_log   = self.detector.get_event_log()
        tfr_df    = self.get_tfr_metrics()
        metrics   = self.get_model_metrics()

        tfr_breaches = int(tfr_df["steps_in_tfr"].sum()) if not tfr_df.empty else 0
        tfr_warnings = int(tfr_df["tfr_warnings"].sum()) if not tfr_df.empty else 0
        avg_min_tfr  = float(tfr_df["min_tfr_dist_nm"].mean()) \
                       if not tfr_df.empty else float("inf")
        entered_tfr  = int((tfr_df["tfr_entries"] > 0).sum()) \
                       if not tfr_df.empty else 0

        total_conflicts = len(det_log) if not det_log.empty else 0
        peak_conflicts  = int(metrics["Total Conflicts"].max()) \
                          if not metrics.empty else 0

        return {
            "mode"                      : self.mode,
            "aircraft_count"            : len(self.aircraft),
            "steps"                     : self.current_step,
            # Group B
            "total_conflict_events"     : total_conflicts,
            "peak_simultaneous_conflicts": peak_conflicts,
            # Group A
            "tfr_breaches"              : tfr_breaches,
            "tfr_warnings"              : tfr_warnings,
            "avg_min_tfr_dist_nm"       : round(avg_min_tfr, 3),
            "aircraft_entered_tfr"      : entered_tfr,
            # Combined
            "any_safety_violation"      : (total_conflicts > 0 or tfr_breaches > 0),
        }


# =============================================================================
# Entry point — quick smoke test
# =============================================================================

if __name__ == "__main__":
    import sys

    _HERE    = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(_HERE, "flight_data")
    TFR_PATH = os.path.join(_HERE, "TFR_Lat_Lon.xlsx")

    if not os.path.isdir(DATA_DIR):
        print(f"  No flight_data/ found at {DATA_DIR} — skipping smoke test.")
        sys.exit(0)

    for mode in ("baseline", "mdp_vi"):
        print("\n" + "=" * 60)
        print(f"  Mode: {mode}")
        print("=" * 60)

        model = AirTrafficModelCombined(
            data_dir      = DATA_DIR,
            tfr_path      = TFR_PATH if os.path.isfile(TFR_PATH) else None,
            mode          = mode,
            train_solvers = (mode != "baseline"),
        )

        print(f"  Flights    : {len(model.callsigns)}")
        print(f"  Max steps  : {model.max_steps}")
        model.run(model.max_steps)

        summary = model.get_combined_summary()
        print("\n  Combined summary:")
        for k, v in summary.items():
            print(f"    {k:<36}: {v}")

        det_summary = model.detector.summary()
        print("\n  ConflictDetector summary:")
        for k, v in det_summary.items():
            print(f"    {k:<28}: {v}")

        tfr_df = model.get_tfr_metrics()
        if not tfr_df.empty:
            print("\n  TFR metrics:")
            print(tfr_df.to_string(index=False))

        print(f"\n  ✓ {mode} run complete.")