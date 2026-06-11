"""
conflict_detection.py
=====================
Owner : Yuvi (Conflict Detection Lead – Group B)
Project: MDP Air Traffic Simulation  |  ITSEC 2026

Responsibilities
----------------
- Separation standard enforcement (3 NM horizontal / 1000 ft vertical)
- Nearest-neighbor search using a KD-tree (replaces brute-force O(n²) loop)
- Structured ConflictEvent logging with duration tracking
- Public API consumed by mdp_collision_env.py (Anaya) and model.py (shared)

Usage
-----
    from conflict_detection import ConflictDetector, HORIZONTAL_SEP_NM, VERTICAL_SEP_FT

    detector = ConflictDetector()
    events   = detector.check(step=42, aircraft_list=model.aircraft)
    log_df   = detector.get_event_log()
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


# ── Separation Standards (syllabus-specified) ─────────────────────────────────

HORIZONTAL_SEP_NM = 3.0    # 3 NM  – as per Week 4/22 task spec
VERTICAL_SEP_FT   = 1000   # 1000 ft vertical

# Earth radius in nautical miles – used for lat/lon → Cartesian conversion
_EARTH_R_NM = 3440.065


# ── Haversine helper (kept here so file is self-contained) ────────────────────

def haversine_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points in nautical miles."""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    return 2 * _EARTH_R_NM * math.asin(math.sqrt(a))


def _to_cartesian(lat_deg: float, lon_deg: float) -> tuple[float, float, float]:
    """
    Convert lat/lon to 3-D Cartesian coordinates on the unit sphere,
    scaled by Earth radius in NM. Used to build the KD-tree so that
    Euclidean distance in 3-D approximates great-circle distance.
    """
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    x = _EARTH_R_NM * math.cos(lat) * math.cos(lon)
    y = _EARTH_R_NM * math.cos(lat) * math.sin(lon)
    z = _EARTH_R_NM * math.sin(lat)
    return x, y, z


# ── ConflictEvent dataclass ───────────────────────────────────────────────────

@dataclass
class ConflictEvent:
    """
    A single detected separation violation between two aircraft.

    Fields
    ------
    step            : simulation step when first detected
    callsign_a      : callsign of aircraft A (alphabetically first)
    callsign_b      : callsign of aircraft B
    h_sep_nm        : horizontal separation at detection (nautical miles)
    v_sep_ft        : vertical separation at detection (feet)
    alt_a_ft        : altitude of aircraft A (feet)
    alt_b_ft        : altitude of aircraft B (feet)
    lat_a, lon_a    : position of aircraft A
    lat_b, lon_b    : position of aircraft B
    duration_steps  : how many consecutive steps the conflict lasted
    resolved_step   : step at which separation was restored (None if ongoing)
    severity        : 'CRITICAL' h<1nm | 'HIGH' h<2nm | 'MODERATE' otherwise
    """
    step           : int
    callsign_a     : str
    callsign_b     : str
    h_sep_nm       : float
    v_sep_ft       : float
    alt_a_ft       : float
    alt_b_ft       : float
    lat_a          : float
    lon_a          : float
    lat_b          : float
    lon_b          : float
    duration_steps : int   = 1
    resolved_step  : Optional[int] = None
    severity       : str   = field(init=False)

    def __post_init__(self):
        if self.h_sep_nm < 1.0:
            self.severity = "CRITICAL"
        elif self.h_sep_nm < 2.0:
            self.severity = "HIGH"
        else:
            self.severity = "MODERATE"

    def to_dict(self) -> dict:
        return {
            "step"          : self.step,
            "callsign_a"    : self.callsign_a,
            "callsign_b"    : self.callsign_b,
            "h_sep_nm"      : round(self.h_sep_nm, 4),
            "v_sep_ft"      : round(self.v_sep_ft, 1),
            "alt_a_ft"      : self.alt_a_ft,
            "alt_b_ft"      : self.alt_b_ft,
            "lat_a"         : self.lat_a,
            "lon_a"         : self.lon_a,
            "lat_b"         : self.lat_b,
            "lon_b"         : self.lon_b,
            "duration_steps": self.duration_steps,
            "resolved_step" : self.resolved_step,
            "severity"      : self.severity,
        }


# ── ConflictDetector ──────────────────────────────────────────────────────────

class ConflictDetector:
    """
    Stateful conflict detector using a KD-tree for nearest-neighbor search.

    How it works
    ------------
    1. At each step, active aircraft positions are converted to 3-D Cartesian
       coordinates and loaded into a cKDTree.
    2. The tree's query_pairs() method finds all pairs within a bounding
       Euclidean distance in O(n log n) — far faster than the O(n²) brute-
       force loop used previously, especially as aircraft count grows.
    3. For each candidate pair, the exact haversine distance is computed and
       the vertical separation is checked.
    4. Confirmed conflicts are matched against the active_conflicts dict to
       track duration across steps. New events are appended to event_log.
    5. Resolved conflicts (pair drops out of violation) have their
       resolved_step stamped and are moved to closed_conflicts.

    Public API (used by mdp_collision_env.py)
    -----------------------------------------
    check(step, aircraft_list)  → List[ConflictEvent]  (active this step)
    get_event_log()             → pd.DataFrame
    get_active_conflicts()      → dict  {pair_key: ConflictEvent}
    summary()                   → dict  of headline stats
    reset()                     → clears all state
    """

    def __init__(
        self,
        h_sep_nm: float = HORIZONTAL_SEP_NM,
        v_sep_ft: float = VERTICAL_SEP_FT,
    ):
        self.h_sep_nm = h_sep_nm
        self.v_sep_ft = v_sep_ft

        # KD-tree search radius — slightly larger than h_sep to ensure we
        # don't miss any pairs at the chord/arc boundary
        self._kdtree_radius = h_sep_nm * 1.05

        # Internal state
        self._event_log    : List[ConflictEvent] = []
        self._active       : dict[str, ConflictEvent] = {}   # pair_key → event
        self._closed       : List[ConflictEvent] = []

    # ── Main entry point ──────────────────────────────────────────────────────

    def check(self, step: int, aircraft_list: list) -> List[ConflictEvent]:
        """
        Run conflict detection for one simulation step.

        Parameters
        ----------
        step          : current simulation step number
        aircraft_list : list of AircraftAgent (or any object with
                        .callsign, .lat, .lon, .altitude, .active)

        Returns
        -------
        List of ConflictEvent objects active this step (new + continuing).
        """
        # Only consider airborne aircraft — exclude ground/taxi (altitude < 1000 ft)
        # Ground aircraft at 0 ft triggering false conflicts against low-altitude
        # climb-out is a known data quality issue in the FlightRadar24 CSVs.
        active = [a for a in aircraft_list if a.active and a.altitude >= 1000.0]

        if len(active) < 2:
            self._resolve_all(step)
            return []

        # ── Step 1: Build KD-tree from Cartesian positions ────────────────────
        coords = np.array([_to_cartesian(a.lat, a.lon) for a in active])
        tree   = cKDTree(coords)

        # ── Step 2: Nearest-neighbor search — candidate pairs within radius ───
        # Returns set of (i, j) pairs where i < j
        candidate_pairs = tree.query_pairs(r=self._kdtree_radius)

        # ── Step 3: Precise check on each candidate ───────────────────────────
        conflicts_this_step: dict[str, ConflictEvent] = {}

        for i, j in candidate_pairs:
            a = active[i]
            b = active[j]

            h_sep = haversine_nm(a.lat, a.lon, b.lat, b.lon)
            v_sep = abs(a.altitude - b.altitude)

            if h_sep < self.h_sep_nm and v_sep < self.v_sep_ft:
                # Canonical key: alphabetically ordered so (A,B) == (B,A)
                cs_a, cs_b = sorted([a.callsign, b.callsign])
                pair_key   = f"{cs_a}|{cs_b}"

                # Map back to a/b by sorted order
                if a.callsign == cs_a:
                    fa, fb = a, b
                else:
                    fa, fb = b, a

                event = ConflictEvent(
                    step       = step,
                    callsign_a = cs_a,
                    callsign_b = cs_b,
                    h_sep_nm   = h_sep,
                    v_sep_ft   = v_sep,
                    alt_a_ft   = fa.altitude,
                    alt_b_ft   = fb.altitude,
                    lat_a      = fa.lat,
                    lon_a      = fa.lon,
                    lat_b      = fb.lat,
                    lon_b      = fb.lon,
                )
                conflicts_this_step[pair_key] = event

        # ── Step 4: Update duration tracking ─────────────────────────────────
        active_keys = set(self._active.keys())
        current_keys = set(conflicts_this_step.keys())

        # Continuing conflicts — increment duration, update measurements
        for key in active_keys & current_keys:
            self._active[key].duration_steps += 1
            # Update to latest position/separation reading
            self._active[key].h_sep_nm = conflicts_this_step[key].h_sep_nm
            self._active[key].v_sep_ft = conflicts_this_step[key].v_sep_ft

        # New conflicts — add to active and log
        for key in current_keys - active_keys:
            ev = conflicts_this_step[key]
            self._active[key] = ev
            self._event_log.append(ev)

        # Resolved conflicts — stamp resolved_step and move to closed
        for key in active_keys - current_keys:
            ev = self._active.pop(key)
            ev.resolved_step = step
            self._closed.append(ev)

        return list(self._active.values())

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _resolve_all(self, step: int):
        """Mark all active conflicts as resolved (called when < 2 aircraft active)."""
        for ev in self._active.values():
            ev.resolved_step = step
            self._closed.append(ev)
        self._active.clear()

    # ── Public accessors ──────────────────────────────────────────────────────

    def get_event_log(self) -> pd.DataFrame:
        """
        Returns all conflict events (resolved + still active) as a DataFrame.
        Each row is one conflict episode (not one step — duration is tracked).
        """
        all_events = self._event_log + list(self._active.values())
        if not all_events:
            return pd.DataFrame()
        return pd.DataFrame([e.to_dict() for e in all_events])

    def get_active_conflicts(self) -> dict[str, "ConflictEvent"]:
        """Returns the dict of currently active (unresolved) conflicts."""
        return dict(self._active)

    def is_in_conflict(self, callsign: str) -> bool:
        """Returns True if the given callsign is currently in any active conflict."""
        return any(callsign in key.split("|") for key in self._active)

    def conflict_partners(self, callsign: str) -> List[str]:
        """Returns list of callsigns currently in conflict with the given aircraft."""
        return [
            key.replace(callsign, "").strip("|")
            for key in self._active
            if callsign in key.split("|")
        ]

    def summary(self) -> dict:
        """Headline statistics over the full simulation run so far."""
        log = self.get_event_log()
        if log.empty:
            return {"total_events": 0, "resolved": 0, "active": 0}

        return {
            "total_events"       : len(log),
            "resolved"           : len(self._closed),
            "active"             : len(self._active),
            "critical_events"    : int((log["severity"] == "CRITICAL").sum()),
            "high_events"        : int((log["severity"] == "HIGH").sum()),
            "moderate_events"    : int((log["severity"] == "MODERATE").sum()),
            "avg_duration_steps" : round(log["duration_steps"].mean(), 2),
            "max_duration_steps" : int(log["duration_steps"].max()),
            "min_h_sep_nm"       : round(log["h_sep_nm"].min(), 4),
            "flights_involved"   : sorted(set(log["callsign_a"]) | set(log["callsign_b"])),
        }

    def reset(self):
        """Clear all state — call between simulation runs."""
        self._event_log.clear()
        self._active.clear()
        self._closed.clear()


# ── Standalone test / demo ────────────────────────────────────────────────────

if __name__ == "__main__":
    import glob, os, sys

    # Dynamically points to the flight_data/ subfolder next to this script
    # Works on any machine without changing the path manually
    _HERE    = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(_HERE, "flight_data")

    # Import shared model (must be in the same folder as this script)
    sys.path.insert(0, _HERE)
    from air_traffic_abm import AirTrafficModel

    print("=" * 60)
    print("  conflict_detection.py — standalone test")
    print(f"  Horizontal threshold : {HORIZONTAL_SEP_NM} NM")
    print(f"  Vertical threshold   : {VERTICAL_SEP_FT} ft")
    print("=" * 60)

    # Build model and detector
    model    = AirTrafficModel(data_dir=DATA_DIR, time_step_s=30)
    detector = ConflictDetector(h_sep_nm=HORIZONTAL_SEP_NM, v_sep_ft=VERTICAL_SEP_FT)

    print(f"\nFlights loaded  : {len(model.callsigns)}")
    print(f"Callsigns       : {model.callsigns}")
    print(f"Running {model.max_steps} steps …\n")

    # Run simulation
    for s in range(model.max_steps):
        model.step()
        active_conflicts = detector.check(step=s, aircraft_list=model.aircraft)

        if active_conflicts:
            for ev in active_conflicts:
                marker = "⚠ NEW" if ev.duration_steps == 1 else f"  (step {ev.duration_steps})"
                print(f"  Step {s:>4} | {marker} | {ev.callsign_a} ↔ {ev.callsign_b} | "
                      f"h={ev.h_sep_nm:.2f} nm  v={ev.v_sep_ft:.0f} ft | {ev.severity}")

    # Results
    print("\n" + "=" * 60)
    print("  CONFLICT EVENT LOG")
    print("=" * 60)
    log = detector.get_event_log()
    if log.empty:
        print("  No conflicts detected.")
    else:
        print(log.to_string(index=False))

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    for k, v in detector.summary().items():
        print(f"  {k:<24}: {v}")

    # Save
    out = os.path.join(_HERE, "conflict_events.csv")
    if not log.empty:
        log.to_csv(out, index=False)
        print(f"\n  Saved → {out}")