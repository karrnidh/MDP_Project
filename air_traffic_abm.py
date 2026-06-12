"""
air_traffic_abm.py
==================
Owner  : Yuvi/Anaya (shared)
Project: MDP Air Traffic Simulation  |  ITSEC 2026

Air Traffic Agent-Based Model using Mesa
-----------------------------------------
Simulates real flights using FR24 trajectory data.

"""

import pandas as pd
import numpy as np
import glob
import os
from math import radians, cos, sin, asin, sqrt, atan2, degrees
import mesa

from conflict_detection import (
    ConflictDetector,
    HORIZONTAL_SEP_NM,   # 3.0 NM  — single source of truth
    VERTICAL_SEP_FT,     # 1 000 ft
)
from mdp_collision_env import CollisionMDP





# ── Utility Functions ─────────────────────────────────────────────────────────

def haversine_nm(lat1, lon1, lat2, lon2):
    """Distance between two lat/lon points in nautical miles."""
    R = 3440.065
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    return 2 * R * asin(sqrt(a))


def bearing_deg(lat1, lon1, lat2, lon2):
    """Bearing from point 1 to point 2 in degrees (0=N, 90=E)."""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = sin(dlon) * cos(lat2)
    y = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)
    return (degrees(atan2(x, y)) + 360) % 360


def load_flight_data(data_dir: str) -> pd.DataFrame:
    """Load and clean all CSV flight trajectory files from the dats folder.

    Handles two CSV formats:
      Format A (old): combined 'Position' column → 'lat,lon' string
      Format B (new): separate 'lat', 'lon', 'Direction' columns already present
                      e.g. Timestamp,Callsign,lat,lon,Altitude,Speed,Direction
    """
    dfs = []
    for f in glob.glob(os.path.join(data_dir, "*.csv")):
        df = pd.read_csv(f)
        df["source"] = os.path.basename(f)
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    df = pd.concat(dfs, ignore_index=True)

    # Handle both CSV formats
    if "Position" in df.columns and "lat" not in df.columns:
        # Format A: split 'Position' into lat/lon
        df[["lat", "lon"]] = df["Position"].str.split(",", expand=True).astype(float)
    elif "lat" in df.columns and "lon" in df.columns:
        # Format B: lat/lon columns already present — ensure correct types
        df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
        df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    else:
        raise ValueError(
            "CSV files must have either a 'Position' column ('lat,lon' string) "
            "or separate 'lat' and 'lon' columns."
        )

    # Ensure Direction column exists (heading); default 0 if missing
    if "Direction" not in df.columns:
        df["Direction"] = 0.0

    df = df.dropna(subset=["lat", "lon", "Altitude", "Speed"]).reset_index(drop=True)
    df = df.sort_values(["Callsign", "Timestamp"]).reset_index(drop=True)

    # Convert Timestamp to numeric seconds for rel_time calculation.
    # Handles both Unix integer timestamps and ISO string timestamps
    # (e.g. "2024-01-01 03:56:41+00:00" from new FR24 CSVs).
    try:
        ts_numeric = pd.to_numeric(df["Timestamp"])
    except (ValueError, TypeError):
        ts_numeric = pd.to_datetime(df["Timestamp"], utc=True).astype("int64") // 10**9

    df["_ts_numeric"] = ts_numeric
    df["rel_time"] = df.groupby("Callsign")["_ts_numeric"].transform(
        lambda x: x - x.min()
    )
    df = df.drop(columns=["_ts_numeric"])
    return df





# ── Aircraft Agent ─────────────────────────────────────────────────────────────

class AircraftAgent(mesa.Agent):
    """
    Represents a single aircraft.

    In BASELINE mode  — replays the recorded CSV trajectory step by step.
    In MDP mode       — position is controlled by CollisionMDP.run_step().

    Attributes
    ----------
    callsign        : ICAO/FR24 callsign
    lat, lon        : current position (decimal degrees)
    altitude        : current altitude (feet)
    speed           : current speed (knots)
    heading         : current heading (degrees)
    vertical_rate   : current vertical rate (fpm) — updated by MDP write-back
    phase           : 'ground' | 'climb' | 'cruise' | 'descent' | 'landed'
    active          : False once the trajectory ends or the aircraft lands
    conflicts       : set of callsigns currently in conflict (baseline only)
    total_conflicts : cumulative conflict count (baseline _check_conflicts)
    mdp             : CollisionMDP instance, or None in baseline mode
    """

    def __init__(self, model, callsign: str, trajectory: pd.DataFrame):
        super().__init__(model)
        self.callsign      = callsign
        self._traj         = trajectory.reset_index(drop=True)
        self._step_idx     = 0

        # Initialise from first waypoint
        first = self._traj.iloc[0]
        self.lat           = first["lat"]
        self.lon           = first["lon"]
        self.altitude      = float(first["Altitude"])
        self.speed         = float(first["Speed"])
        self.heading       = float(first["Direction"])
        self.vertical_rate = 0.0   # not in CSV; MDP will update this
        self.active        = True
        self.phase         = self._compute_phase()
        self.conflicts     : set = set()
        self.total_conflicts: int = 0
        self.mdp           : CollisionMDP | None = None   # attached by model

    # ── Phase detection ────────────────────────────────────────────────────────

    def _compute_phase(self) -> str:
        if not self.active:
            return "landed"
        if self.altitude < 500:
            return "ground"
        if self.altitude < 10_000:
            return "climb" if self.speed > 150 else "descent"
        if self.altitude > 25_000:
            return "cruise"
        return "climb"

    # ── Step ──────────────────────────────────────────────────────────────────

    def step(self):
        """
        Advance one simulation step.

        BASELINE mode  (self.mdp is None):
            Reads the next row from the CSV trajectory, updates position,
            then runs _check_conflicts() for the agent-level conflict set.
            ConflictDetector.check() is called separately at the model level.

        MDP mode  (self.mdp is not None):
            Delegates entirely to CollisionMDP.run_step(), which:
              1. Calls acas_x_policy() to select an ACAS X RA
              2. Applies kinematics via apply_action_kinematics()
              3. Writes the new position back to this agent
              4. Runs ConflictDetector.check() on the updated positions
              5. Computes and logs the reward
        """
        if not self.active:
            return

        self._step_idx += 1

        if self._step_idx >= len(self._traj):
            self.active = False
            self.phase  = "landed"
            return

        if self.mdp is not None:
            # ── MDP/POMDP mode: CSV is ground truth; MDP is a safety layer ───
            #
            # Always advance the CSV trajectory first so the aircraft follows
            # its real-world flight plan by default.  The MDP only overrides
            # position/altitude/speed when acas_x_policy() detects an active
            # threat — i.e. when it would return something other than MAINTAIN
            # or LEVEL_OFF.  When there is no threat the CSV values written
            # here are left untouched, so speed, altitude and heading evolve
            # exactly as in baseline mode.
            row           = self._traj.iloc[self._step_idx]
            self.lat      = row["lat"]
            self.lon      = row["lon"]
            self.altitude = float(row["Altitude"])
            self.speed    = float(row["Speed"])
            self.heading  = float(row["Direction"])

            # Sync the MDP's internal state to the current CSV position, then
            # run the policy.  run_step() only writes back to the agent when
            # an avoidance action is chosen; MAINTAIN leaves the values above
            # untouched.
            self.mdp.run_step(step_num=self._step_idx)
        else:
            # ── Baseline mode: replay CSV trajectory ──────────────────────────
            row           = self._traj.iloc[self._step_idx]
            self.lat      = row["lat"]
            self.lon      = row["lon"]
            self.altitude = float(row["Altitude"])
            self.speed    = float(row["Speed"])
            self.heading  = float(row["Direction"])
            self._check_conflicts()

        self.phase = self._compute_phase()

    # ── Conflict detection (baseline only) ────────────────────────────────────

    def _check_conflicts(self):
        """
        O(n²) conflict check used in BASELINE mode only.

        Uses the same 3 NM / 1 000 ft thresholds as ConflictDetector so
        baseline agent-level counts are comparable with MDP detector output.
        Populates self.conflicts and increments self.total_conflicts.
        """
        self.conflicts = set()
        for other in self.model.aircraft:
            if other is self or not other.active:
                continue
            h_sep = haversine_nm(self.lat, self.lon, other.lat, other.lon)
            v_sep = abs(self.altitude - other.altitude)
            if h_sep < HORIZONTAL_SEP_NM and v_sep < VERTICAL_SEP_FT:
                self.conflicts.add(other.callsign)
                self.total_conflicts += 1

    # ── Properties for DataCollector ──────────────────────────────────────────

    @property
    def n_conflicts(self) -> int:
        """
        Active conflict count reported to DataCollector.

        Baseline : from _check_conflicts() (agent-level set)
        MDP      : from the shared ConflictDetector via the model
        """
        if self.mdp is not None:
            return sum(
                1 for key in self.model.detector.get_active_conflicts()
                if self.callsign in key.split("|")
            )
        return len(self.conflicts)

    @property
    def progress_pct(self) -> float:
        return self._step_idx / max(len(self._traj) - 1, 1) * 100


# ── Air Traffic Model ──────────────────────────────────────────────────────────

class AirTrafficModel(mesa.Model):
    """
    Mesa model that runs real air traffic trajectories in two modes.

    Parameters
    ----------
    data_dir    : path to dats folder containing CSV files
    time_step_s : seconds per simulation step (default 30)
    mdp_mode    : False = baseline CSV replay (default)
                  True  = MDP policy controls all aircraft positions

    Shared objects
    --------------
    self.detector : ConflictDetector
        Single KD-tree detector shared across all agents and all MDPs.
        Owned here — reset only by AirTrafficModel, never by individual MDPs.
        Call model.detector.get_event_log() after a run for the full log.

    Output
    ------
    After model.run():
        model.get_model_metrics()  → DataCollector time-series DataFrame
        model.get_snapshots()      → per-agent per-step DataFrame
        model.detector.get_event_log() → ConflictEvent log (authoritative)
        model.get_mdp_logs()       → per-agent MDP step logs (MDP mode only)
    """

    def __init__(self, data_dir: str, time_step_s: int = 30,
                 mdp_mode: bool = False):
        super().__init__()
        self.time_step_s  = time_step_s
        self.current_step = 0
        self.mdp_mode     = mdp_mode

        # ── Single shared ConflictDetector (KD-tree, 3 NM / 1 000 ft) ────────
        # This is the authoritative detector for both modes.
        # It is NOT reset by individual MDPs — only by this model.
        self.detector = ConflictDetector(
            h_sep_nm = HORIZONTAL_SEP_NM,
            v_sep_ft = VERTICAL_SEP_FT,
        )

        # ── Load flight data from dats folder ────────────────────────────────────────────
        df = load_flight_data(data_dir)
        self.callsigns = sorted(df["Callsign"].unique())

        # ── Create agents ──────────────────────────────────────────────────────
        self.aircraft: list[AircraftAgent] = []
        for cs in self.callsigns:
            traj  = df[df["Callsign"] == cs].copy()
            agent = AircraftAgent(self, cs, traj)
            self.aircraft.append(agent)

        # ── Attach MDPs in MDP mode ────────────────────────────────────────────
        # All agents must exist before any MDP is reset so that
        # _build_intruder_features() can see the full aircraft list.
        if self.mdp_mode:
            for agent in self.aircraft:
                agent.mdp = CollisionMDP(
                    detector         = self.detector,
                    ownship_callsign = agent.callsign,
                    max_steps        = self.max_steps,
                )
            for agent in self.aircraft:
                agent.mdp.reset(agent, self.aircraft)

        # ── DataCollector ──────────────────────────────────────────────────────
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Active Flights"   : lambda m: sum(1 for a in m.aircraft if a.active),
                # Authoritative conflict count — always from ConflictDetector
                "Total Conflicts"  : lambda m: len(m.detector.get_active_conflicts()),
                "Flights Airborne" : lambda m: sum(
                    1 for a in m.aircraft if a.active and a.altitude > 500
                ),
                "Flights on Ground": lambda m: sum(
                    1 for a in m.aircraft if a.active and a.altitude <= 500
                ),
                "Avg Altitude (ft)": lambda m: (
                    np.mean([a.altitude for a in m.aircraft
                             if a.active and a.altitude > 0])
                    if any(a.active and a.altitude > 0 for a in m.aircraft) else 0
                ),
                "Avg Speed (kts)"  : lambda m: (
                    np.mean([a.speed for a in m.aircraft
                             if a.active and a.speed > 0])
                    if any(a.active and a.speed > 0 for a in m.aircraft) else 0
                ),
                "MDP Mode"         : lambda m: m.mdp_mode,
            },
            agent_reporters={
                "Callsign"  : "callsign",
                "Lat"       : "lat",
                "Lon"       : "lon",
                "Altitude"  : "altitude",
                "Speed"     : "speed",
                "Heading"   : "heading",
                "Phase"     : "phase",
                "Active"    : "active",
                "Conflicts" : "n_conflicts",
                "Progress"  : "progress_pct",
            },
        )

        self.datacollector.collect(self)

    # ── Step ──────────────────────────────────────────────────────────────────

    def step(self):
        """
        Advance all agents by one step.

        Order of operations
        -------------------
        1. Each agent steps (CSV replay or MDP run_step).
           In MDP mode, ConflictDetector.check() is called inside
           each agent's run_step() — so by the time all agents have
           stepped, the detector has the latest picture.
        2. In BASELINE mode, ConflictDetector.check() is called once
           at the model level after all agents have moved, so it sees
           the fully updated positions of every aircraft.
        3. DataCollector records the current state.
        """
        for agent in self.aircraft:
            agent.step()

        # Baseline: run the authoritative KD-tree detector once per step
        # (In MDP mode this is handled inside each agent's run_step())
        if not self.mdp_mode:
            self.detector.check(
                step         = self.current_step,
                aircraft_list= self.aircraft,
            )

        self.current_step += 1
        self.datacollector.collect(self)

    def run(self, steps: int):
        """Run the model for up to `steps` steps."""
        for _ in range(steps):
            if not any(a.active for a in self.aircraft):
                break
            self.step()
        return self

    # ── Convenience ───────────────────────────────────────────────────────────

    @property
    def max_steps(self) -> int:
        """Maximum trajectory length across all agents."""
        return max(len(a._traj) for a in self.aircraft)

    def get_snapshots(self) -> pd.DataFrame:
        """Agent-level data across all steps (from DataCollector)."""
        return self.datacollector.get_agent_vars_dataframe().reset_index()

    def get_model_metrics(self) -> pd.DataFrame:
        """Model-level metrics across all steps (from DataCollector)."""
        return self.datacollector.get_model_vars_dataframe().reset_index()

    def get_mdp_logs(self) -> pd.DataFrame:
        """
        Concatenated per-step MDP logs for all aircraft (MDP mode only).

        Returns a DataFrame with one row per agent per step, containing
        position, action, reward components, and conflict state.
        Merges cleanly with detector.get_event_log() on the 'step' column.

        Returns an empty DataFrame in baseline mode.
        """
        if not self.mdp_mode:
            return pd.DataFrame()
        frames = []
        for agent in self.aircraft:
            if agent.mdp is not None:
                df = agent.mdp.get_step_log()
                if not df.empty:
                    frames.append(df)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def reset_detector(self):
        """
        Clear the shared ConflictDetector between runs.
        Call this before re-running the same model instance.
        """
        self.detector.reset()


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os as _os

    DATA_DIR = _os.path.join(
        _os.path.dirname(_os.path.abspath(__file__)), "flight_data"
    )

    for mode_name, mdp_flag in [("BASELINE", False), ("MDP", True)]:
        print("=" * 60)
        print(f"  Mode: {mode_name}")
        print("=" * 60)

        model = AirTrafficModel(
            data_dir    = DATA_DIR,
            time_step_s = 30,
            mdp_mode    = mdp_flag,
        )

        print(f"  Flights loaded : {len(model.callsigns)}")
        print(f"  Callsigns      : {model.callsigns}")
        print(f"  Max steps      : {model.max_steps}")
        print(f"  Running {model.max_steps} steps …\n")

        model.run(model.max_steps)

        metrics = model.get_model_metrics()
        snaps   = model.get_snapshots()
        det_log = model.detector.get_event_log()

        print("── Model-level summary ──────────────────────────────")
        print(metrics[["Active Flights", "Total Conflicts",
                        "Avg Altitude (ft)", "Avg Speed (kts)"]].describe().round(1))

        print("\n── ConflictDetector summary ─────────────────────────")
        summary = model.detector.summary()
        for k, v in summary.items():
            print(f"  {k:<28}: {v}")

        if mdp_flag:
            mdp_log = model.get_mdp_logs()
            print(f"\n── MDP step log rows : {len(mdp_log)}")
            if not mdp_log.empty:
                print(f"  Columns : {list(mdp_log.columns)}")
                print(f"  Actions (sample):\n{mdp_log['action'].value_counts().head()}")

        print(f"\n✓ {mode_name} run complete.\n")