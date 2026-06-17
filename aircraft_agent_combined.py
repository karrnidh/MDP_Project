"""
aircraft_agent_combined.py
==========================
Merged AircraftAgent — Group A (TFR Avoidance) + Group B (Collision Avoidance)

Combines:
  - Group A (mdp.py):      TFR polygon avoidance, route planning, VI/QL/POMDP solvers
  - Group B (air_traffic_abm.py): Mesa agent, CSV replay, CollisionMDP / CollisionPOMDP

Priority:  Collision avoidance > TFR avoidance > route guidance
           (collision penalties are ~5× larger, so the combined scorer naturally
            prioritises separation over TFR margin when both are active)

Authors:
  Dev / Karrnidh  — Group A logic (TFR, MDP-VI, MDP-QL, POMDP-A)
  Yuvi / Anaya    — Group B logic (Mesa, ConflictDetector, ACAS-X, POMDP-B)
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import mesa
import numpy as np
import pandas as pd

# ── Group B imports ──────────────────────────────────────────────────────────
from conflict_detection import (
    ConflictDetector,
    HORIZONTAL_SEP_NM,
    VERTICAL_SEP_FT,
)
from mdp_collision_env import CollisionMDP, CollisionPOMDP, Action

# ── Group A geometry helpers (copied from mdp.py to keep files independent) ──
# If mdp.py is available in the same package you can instead do:
#   from mdp import haversine, bearing_between, move_by_heading, ...
# but inline copies avoid a circular-import problem.

EARTH_RADIUS_NM = 3440.0


def _haversine(lat1, lon1, lat2, lon2) -> float:
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return float(2 * EARTH_RADIUS_NM * np.arcsin(np.sqrt(a)))


def _bearing(lat1, lon1, lat2, lon2) -> float:
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return float((np.degrees(np.arctan2(x, y)) + 360.0) % 360.0)


def _move(lat, lon, heading_deg, speed_kts, dt_s) -> Tuple[float, float]:
    d_nm = (speed_kts / 3600.0) * dt_s
    ang = d_nm / EARTH_RADIUS_NM
    lat_r = math.radians(lat)
    lon_r = math.radians(lon)
    hdg_r = math.radians(heading_deg)
    new_lat = math.asin(
        math.sin(lat_r) * math.cos(ang)
        + math.cos(lat_r) * math.sin(ang) * math.cos(hdg_r)
    )
    new_lon = lon_r + math.atan2(
        math.sin(hdg_r) * math.sin(ang) * math.cos(lat_r),
        math.cos(ang) - math.sin(lat_r) * math.sin(new_lat),
    )
    return math.degrees(new_lat), math.degrees(new_lon)


def _rel_bearing(my_hdg, target_brg) -> float:
    """Target bearing relative to nose (+right, -left), range [-180, 180]."""
    return (target_brg - my_hdg + 540.0) % 360.0 - 180.0


# ── Constants (kept in sync with mdp.py) ──────────────────────────────────────
TFR_WARNING_BUFFER_NM   = 10.0
TFR_EMERGENCY_BUFFER_NM = 3.0
TFR_REJOIN_BUFFER_NM    = 18.0
TFR_DETOUR_BUFFER_NM    = 28.0
MDP_ACTIVE_TFR_NM       = 45.0
SEPARATION_HARD_NM      = HORIZONTAL_SEP_NM   # 3.0
SEPARATION_WARN_NM      = 5.0
SEPARATION_AWARE_NM     = 10.0
CONFLICT_LOOKAHEAD_STEPS = 8
STEP_DURATION_SECONDS   = 30
WAYPOINT_CAPTURE_NM     = 20.0
GUIDANCE_TURN_DEG       = 20.0
EVASIVE_TURN_DEG        = 45.0
ALT_FLOOR_FT            = 5_000.0
ALT_CEILING_FT          = 45_000.0
ALT_CLIMB_RATE_FT       = 1_000.0


# =============================================================================
# MERGED AircraftAgent
# =============================================================================

class AircraftAgent(mesa.Agent):
    """
    Single Mesa agent that handles BOTH TFR avoidance (Group A) and
    collision avoidance (Group B) simultaneously.

    Modes
    -----
    baseline  — replays CSV trajectory; no MDP
    mdp_vi    — Group A VI solver for TFR + Group B ACAS-X for collisions
    mdp_ql    — Group A QL solver for TFR + Group B ACAS-X for collisions
    pomdp     — Group A POMDP for TFR + Group B SARSOP POMDP for collisions

    Action resolution priority (highest first)
    -------------------------------------------
    1. Collision avoidance  (ACAS-X / CollisionMDP)  — overrides everything
    2. TFR avoidance        (Group A MDP solvers)
    3. Route guidance       (waypoints toward destination)

    When both Group A and Group B recommend non-MAINTAIN actions the combined
    candidate scorer picks the action with the best composite score, which
    naturally favours collision avoidance because its penalty weights are ~5×
    larger than TFR penalties.

    Attributes (Group B originals, extended)
    ----------------------------------------
    callsign        : ICAO/FR24 callsign
    lat, lon        : current position (decimal degrees)
    altitude        : current altitude (feet)
    speed           : current speed (knots)
    heading         : current heading (degrees)
    vertical_rate   : current vertical rate (fpm)
    phase           : 'ground' | 'climb' | 'cruise' | 'descent' | 'landed'
    active          : False once the trajectory ends
    conflicts       : set of callsigns currently in conflict (baseline)
    total_conflicts : cumulative conflict count (baseline)
    mdp             : CollisionMDP / CollisionPOMDP instance (None = baseline)

    New Group A attributes
    ----------------------
    route_waypoints : TFR detour waypoints
    destination_lat / destination_lon : final waypoint from CSV
    in_tfr          : True if currently inside TFR boundary
    history_*       : per-step history lists for metrics / plotting
    """

    def __init__(
        self,
        model: "AirTrafficModel",          # type: ignore[name-defined]
        callsign: str,
        trajectory: pd.DataFrame,
        mode: str = "baseline",
    ):
        super().__init__(model)
        self.callsign   = callsign
        self._traj      = trajectory.reset_index(drop=True)
        self._step_idx  = 0
        self.mode       = mode

        # ── Initialise from first CSV row ──────────────────────────────────
        first = self._traj.iloc[0]
        self.lat           = float(first["lat"])
        self.lon           = float(first["lon"])
        self.altitude      = float(first["Altitude"])
        self.speed         = float(first["Speed"])
        self.heading       = float(first.get("Direction", 0.0))
        self.vertical_rate = 0.0
        self.active        = True
        self.phase         = self._compute_phase()

        # Group B conflict tracking
        self.conflicts      : set = set()
        self.total_conflicts: int = 0

        # Group B MDP handle (attached externally by AirTrafficModel)
        self.mdp: Optional[CollisionMDP] = None

        # Group A — destination / route
        last = self._traj.iloc[-1]
        self.destination_lat = float(last["lat"])
        self.destination_lon = float(last["lon"])
        self.route_waypoints: List[Tuple[float, float]] = []

        # Group A — altitude bookkeeping
        cruise = self._traj[self._traj["Altitude"] >= ALT_FLOOR_FT]
        self.nominal_altitude = float(
            cruise["Altitude"].mean() if len(cruise) else 35_000.0
        )
        self.nominal_speed = float(self._traj["Speed"].mean())

        # Group A — TFR state
        tfr = getattr(model, "tfr", None)
        self.in_tfr = tfr.contains(self.lat, self.lon) if tfr else False

        # Group A — per-step history (for metrics and visualisation)
        self.history_lat          = [self.lat]
        self.history_lon          = [self.lon]
        self.history_altitude     = [self.altitude]
        self.history_time         = [0]
        self.history_in_tfr       = [self.in_tfr]
        self.history_tfr_distance = [
            tfr.distance_to_edge(self.lat, self.lon) if tfr else float("inf")
        ]
        self.history_action       = ["INIT"]

        # Track last MDP action for POMDP belief update
        self._last_mdp_action = "MAINTAIN"

    # =========================================================================
    # Phase detection (Group B)
    # =========================================================================

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

    # =========================================================================
    # Group A — TFR helpers
    # =========================================================================

    def _tfr_conflict_ahead(self, buffer_nm: float = TFR_WARNING_BUFFER_NM) -> bool:
        """Return True if current heading will penetrate the TFR within lookahead."""
        tfr = getattr(self.model, "tfr", None)
        if tfr is None:
            return False
        if tfr.distance_to_edge(self.lat, self.lon) < buffer_nm:
            return True
        for k in range(1, CONFLICT_LOOKAHEAD_STEPS + 1):
            lat, lon = self._predict_position(self.heading, k)
            if tfr.distance_to_edge(lat, lon) < buffer_nm:
                return True
        return False

    def _predict_position(self, heading: float, steps_ahead: int) -> Tuple[float, float]:
        lat, lon = self.lat, self.lon
        for _ in range(steps_ahead):
            lat, lon = _move(lat, lon, heading, self.speed, STEP_DURATION_SECONDS)
        return lat, lon

    def _separation_threats(self) -> List["AircraftAgent"]:
        threats = []

        for other in self.model.aircraft:

            if other.callsign == self.callsign or not other.active:
                continue

            # Fast distance filter FIRST
            dist = _haversine(
                self.lat, self.lon,
                other.lat, other.lon
            )

            if dist > 15:
                continue

            # Expensive prediction only for nearby aircraft
            for k in range(1, CONFLICT_LOOKAHEAD_STEPS + 1):

                a_lat, a_lon = self._predict_position(self.heading, k)
                b_lat, b_lon = other._predict_position(other.heading, k)

                if _haversine(a_lat, a_lon, b_lat, b_lon) < SEPARATION_WARN_NM:
                    threats.append(other)
                    break

        return threats

    def _refresh_route(self):
        """Update TFR detour waypoints, drop ones already captured."""
        tfr = getattr(self.model, "tfr", None)
        if tfr is None:
            return

        # Drop waypoints we have passed
        while self.route_waypoints:
            wp = self.route_waypoints[0]
            if _haversine(self.lat, self.lon, wp[0], wp[1]) <= WAYPOINT_CAPTURE_NM:
                self.route_waypoints.pop(0)
            else:
                break

        start  = (self.lat, self.lon)
        dest   = (self.destination_lat, self.destination_lon)
        buffer = TFR_WARNING_BUFFER_NM

        direct_clear = tfr.segment_clear(self.lat, self.lon,
                                          self.destination_lat, self.destination_lon,
                                          buffer)
        far_from_tfr = tfr.distance_to_edge(self.lat, self.lon) > TFR_REJOIN_BUFFER_NM

        if self.route_waypoints and direct_clear and far_from_tfr:
            self.route_waypoints = []
        if not self.route_waypoints and not direct_clear:
            # Import plan_tfr_detour lazily to avoid circular import if needed
            try:
                from mdp import plan_tfr_detour
                self.route_waypoints = plan_tfr_detour(tfr, start, dest, buffer)
            except ImportError:
                pass  # TFR detour planning unavailable

    def _guidance_target(self) -> Tuple[Tuple[float, float], str]:
        self._refresh_route()
        if self.route_waypoints:
            return self.route_waypoints[0], "ROUTE"
        return (self.destination_lat, self.destination_lon), "DEST"

    def _guidance_action(self, max_turn: float = GUIDANCE_TURN_DEG) -> Tuple[str, str]:
        target, label = self._guidance_target()
        target_hdg    = _bearing(self.lat, self.lon, target[0], target[1])
        diff          = _rel_bearing(self.heading, target_hdg)
        turn          = max(-max_turn, min(max_turn, diff))
        if turn < -32.5:   return "TURN_LEFT_45",  label
        if turn < -8.0:    return "TURN_LEFT_20",   label
        if turn > 32.5:    return "TURN_RIGHT_45",  label
        if turn > 8.0:     return "TURN_RIGHT_20",  label
        return "MAINTAIN", label

    # =========================================================================
    # Group A — MDP state builder
    # =========================================================================

    def _state_for_group_a_mdp(self):
        """Build discrete (tfr_b, tfr_rel_b, dest_b, threat_b, sep_b, traffic_b) state."""
        # Import Group A helpers
        try:
            from mdp import discretize_state, bearing_between, relative_bearing
        except ImportError:
            return None

        tfr = getattr(self.model, "tfr", None)
        if tfr is None:
            return None

        tfr_dist = tfr.distance_to_edge(self.lat, self.lon)
        tfr_brg  = bearing_between(self.lat, self.lon,
                                    tfr.centroid_lat, tfr.centroid_lon)
        tfr_rel  = relative_bearing(self.heading, tfr_brg)

        target, _ = self._guidance_target()
        dest_brg  = bearing_between(self.lat, self.lon, target[0], target[1])
        dest_rel  = relative_bearing(self.heading, dest_brg)

        threats = self._separation_threats()
        if threats:
            t          = threats[0]
            sep        = _haversine(self.lat, self.lon, t.lat, t.lon)
            t_brg      = bearing_between(self.lat, self.lon, t.lat, t.lon)
            threat_rel = relative_bearing(self.heading, t_brg)
        else:
            sep, threat_rel = 999.0, None

        return discretize_state(
            tfr_dist, tfr_rel, dest_rel,
            self._tfr_conflict_ahead(), sep, threat_rel
        )

    # =========================================================================
    # Combined candidate scorer
    # =========================================================================

    # A_ACTIONS that map to Group B Action enum for kinematics scoring
    _ACTION_TO_B = {
        "MAINTAIN":      Action.MAINTAIN,
        "TURN_LEFT_20":  Action.TURN_LEFT_5,    # closest match
        "TURN_RIGHT_20": Action.TURN_RIGHT_5,
        "TURN_LEFT_45":  Action.TURN_LEFT_15,
        "TURN_RIGHT_45": Action.TURN_RIGHT_15,
        "CLIMB":         Action.CLIMB_1500,
        "DESCEND":       Action.DESCEND_1500,
    }

    def _combined_score(
        self,
        action: str,
        preferred_a: Optional[str] = None,   # Group A solver suggestion
        preferred_b: Optional[Action] = None, # Group B ACAS-X suggestion
    ) -> float:
        """
        Score a candidate action using a 12-step lookahead.

        Penalties (highest priority first):
          - Separation loss (collision)  : up to −50 000
          - TFR breach                   : up to −50 000
          - TFR warning zone             : up to  −1 200
          - Manoeuvre cost               : small

        Rewards:
          - Progress toward destination  : +8 per NM
          - Alignment bonuses if action
            matches Group A or B solver   : +20 each
        """
        tfr = getattr(self.model, "tfr", None)

        # ── Apply action to heading / altitude ─────────────────────────────
        h   = self.heading
        alt = self.altitude
        if action == "TURN_LEFT_20":    h = (h - 20.0) % 360.0
        elif action == "TURN_RIGHT_20": h = (h + 20.0) % 360.0
        elif action == "TURN_LEFT_45":  h = (h - 45.0) % 360.0
        elif action == "TURN_RIGHT_45": h = (h + 45.0) % 360.0
        elif action == "CLIMB":         alt = min(alt + ALT_CLIMB_RATE_FT, ALT_CEILING_FT)
        elif action == "DESCEND":       alt = max(alt - ALT_CLIMB_RATE_FT, ALT_FLOOR_FT)

        target, _    = self._guidance_target()
        start_d      = _haversine(self.lat, self.lon, target[0], target[1])
        sim_lat, sim_lon = _move(self.lat, self.lon, h, self.speed, STEP_DURATION_SECONDS)

        min_tfr_dist = tfr.distance_to_edge(sim_lat, sim_lon) if tfr else float("inf")
        min_sep      = 999.0
        look_h       = h

        for step in range(1, 13):
            target_h = _bearing(sim_lat, sim_lon, target[0], target[1])
            diff     = _rel_bearing(look_h, target_h)
            look_h   = (look_h + max(-GUIDANCE_TURN_DEG, min(GUIDANCE_TURN_DEG, diff))) % 360.0
            sim_lat, sim_lon = _move(sim_lat, sim_lon, look_h, self.speed, STEP_DURATION_SECONDS)

            if tfr:
                min_tfr_dist = min(min_tfr_dist, tfr.distance_to_edge(sim_lat, sim_lon))

            for other in self.model.aircraft:
                if _haversine(
                    self.lat,
                    self.lon,
                    other.lat,
                    other.lon
                    ) > 15:
                    continue
                if other.callsign == self.callsign or not other.active:
                    continue
                o_lat, o_lon = other._predict_position(other.heading, step)
                min_sep = min(min_sep, _haversine(sim_lat, sim_lon, o_lat, o_lon))

        end_d    = _haversine(sim_lat, sim_lon, target[0], target[1])
        progress = start_d - end_d

        score  = 8.0 * progress
        score -= 0.02 * abs(_rel_bearing(self.heading, h))
        score -= 0.003 * abs(alt - self.nominal_altitude)

        # ── Collision penalty (Group B priority) ──────────────────────────
        if min_sep < SEPARATION_HARD_NM:
            score -= 50_000.0
        elif min_sep < SEPARATION_WARN_NM:
            score -= 1_800.0 * (SEPARATION_WARN_NM - min_sep + 1.0)

        # ── TFR penalty (Group A priority) ────────────────────────────────
        if tfr:
            if min_tfr_dist < 0:
                score -= 50_000.0
            elif min_tfr_dist < TFR_EMERGENCY_BUFFER_NM:
                score -= 8_000.0
            elif min_tfr_dist < TFR_WARNING_BUFFER_NM:
                score -= 1_200.0 * (TFR_WARNING_BUFFER_NM - min_tfr_dist + 1.0)

        # ── Solver alignment bonuses ──────────────────────────────────────
        if preferred_a and action == preferred_a:
            score += 20.0
        if preferred_b is not None:
            # Map Group B Action enum name back to Group A action string
            b_name = preferred_b.name  # e.g. "CLIMB_1500"
            if "CLIMB" in b_name and action == "CLIMB":
                score += 20.0
            elif "DESCEND" in b_name and action == "DESCEND":
                score += 20.0
            elif "TURN_LEFT" in b_name and "LEFT" in action:
                score += 20.0
            elif "TURN_RIGHT" in b_name and "RIGHT" in action:
                score += 20.0

        # ── Manoeuvre cost ────────────────────────────────────────────────
        if action in ("TURN_LEFT_20", "TURN_RIGHT_20"):
            score -= 2.0
        elif action in ("TURN_LEFT_45", "TURN_RIGHT_45"):
            score -= 7.0
        elif action in ("CLIMB", "DESCEND"):
            score -= 10.0

        return score

    def _best_combined_action(
        self,
        preferred_a: Optional[str] = None,
        preferred_b: Optional[Action] = None,
    ) -> str:
        """Pick the best action from all Group A actions + solver suggestions."""
        from mdp import ACTIONS as GROUP_A_ACTIONS  # type: ignore
        candidates = set(GROUP_A_ACTIONS)
        if preferred_a:
            candidates.add(preferred_a)
        return max(candidates,
                   key=lambda a: self._combined_score(a, preferred_a, preferred_b))

    # =========================================================================
    # Apply action to agent state (Group A kinematics)
    # =========================================================================

    def _apply_group_a_action(self, action: str):
        base = action.lstrip("VI_").lstrip("QL_").lstrip("POMDP_")
        if base == "TURN_LEFT_20":
            self.heading = (self.heading - 20.0) % 360.0
        elif base == "TURN_RIGHT_20":
            self.heading = (self.heading + 20.0) % 360.0
        elif base == "TURN_LEFT_45":
            self.heading = (self.heading - 45.0) % 360.0
        elif base == "TURN_RIGHT_45":
            self.heading = (self.heading + 45.0) % 360.0
        elif base == "CLIMB":
            self.altitude = min(self.altitude + ALT_CLIMB_RATE_FT, ALT_CEILING_FT)
        elif base == "DESCEND":
            self.altitude = max(self.altitude - ALT_CLIMB_RATE_FT, ALT_FLOOR_FT)
        else:  # MAINTAIN — drift back toward nominal
            if self.altitude > self.nominal_altitude + 100:
                self.altitude -= 100.0
            elif self.altitude < self.nominal_altitude - 100:
                self.altitude += 100.0
        self.lat, self.lon = _move(self.lat, self.lon,
                                    self.heading, self.speed, STEP_DURATION_SECONDS)

    # =========================================================================
    # Conflict check (baseline mode, Group B)
    # =========================================================================

    def _check_conflicts_baseline(self):
        self.conflicts = set()
        for other in self.model.aircraft:
            if other is self or not other.active:
                continue
            h_sep = _haversine(self.lat, self.lon, other.lat, other.lon)
            v_sep = abs(self.altitude - other.altitude)
            if h_sep < HORIZONTAL_SEP_NM and v_sep < VERTICAL_SEP_FT:
                self.conflicts.add(other.callsign)
                self.total_conflicts += 1

    # =========================================================================
    # Step
    # =========================================================================

    def step(self):
        if not self.active:
            return

        self._step_idx += 1

        if self._step_idx >= len(self._traj):
            self.active = False
            self.phase  = "landed"
            self._record_history("DONE")
            return

        row = self._traj.iloc[self._step_idx]

        if self.mode == "baseline":
            # ── Baseline: full CSV replay is ground truth ──────────────────
            self.lat      = float(row["lat"])
            self.lon      = float(row["lon"])
            self.altitude = float(row["Altitude"])
            self.speed    = float(row["Speed"])
            self.heading  = float(row.get("Direction", self.heading))
            self._check_conflicts_baseline()
            action = "REPLAY"

        else:
            # ── MDP modes: CSV supplies speed only; MDP owns position/alt/hdg
            # Speed from CSV keeps aircraft kinematics realistic, but
            # lat/lon/altitude/heading are driven by the MDP policy so that
            # TFR and collision corrections are NOT overwritten next step.
            self.speed = float(row["Speed"])
            action = self._step_policy()

        self.phase = self._compute_phase()

        # Update TFR state
        tfr = getattr(self.model, "tfr", None)
        self.in_tfr = tfr.contains(self.lat, self.lon) if tfr else False

        self._record_history(action)

    # ─────────────────────────────────────────────────────────────────────────

    def _step_policy(self) -> str:
        """
        Run the combined MDP policy.

        Returns a string action label for logging.
        """
        # ── 1. Check whether each system is active ─────────────────────────
        tfr = getattr(self.model, "tfr", None)
        tfr_active = (
            tfr is not None and (
                tfr.distance_to_edge(self.lat, self.lon) < MDP_ACTIVE_TFR_NM
                or self._tfr_conflict_ahead()
                or bool(self.route_waypoints)
            )
        )
        collision_active = bool(self._separation_threats())

        # If neither system is active, use pure route guidance
        if not tfr_active and not collision_active:
            guidance, _ = self._guidance_action(GUIDANCE_TURN_DEG)
            self._apply_group_a_action(guidance)   # moves lat/lon
            self._last_mdp_action = guidance
            return guidance

        # ── 2. Query Group A solver ────────────────────────────────────────
        preferred_a: Optional[str] = None
        if tfr_active:
            state_a = self._state_for_group_a_mdp()
            if state_a is not None:
                if self.mode == "mdp_vi":
                    vi = getattr(self.model, "vi_solver", None)
                    if vi:
                        preferred_a = vi.act(state_a)
                elif self.mode == "mdp_ql":
                    ql = getattr(self.model, "ql_solver", None)
                    if ql:
                        preferred_a = ql.act(state_a)
                elif self.mode == "pomdp":
                    pomdp_a = getattr(self.model, "pomdp_solver_a", None)
                    if pomdp_a:
                        preferred_a = pomdp_a.act(
                            self.callsign, state_a, self._last_mdp_action,
                            getattr(getattr(self.model, "vi_solver", None), "T", None)
                        )

        # ── 3. Query Group B solver (CollisionMDP / CollisionPOMDP) ────────
        preferred_b: Optional[Action] = None
        if self.mdp is not None and (collision_active or tfr_active):
            # Sync MDP state then get its suggestion without writing back yet
            # (write-back happens after we pick the final action below)
            from mdp_collision_env import acas_x_policy
            tmp_state = self.mdp._build_state(self, self.model.aircraft)
            preferred_b = acas_x_policy(tmp_state)
            if preferred_b == Action.MAINTAIN:
                preferred_b = None   # neutral — no preference

        # ── 4. Pick best combined action ───────────────────────────────────
        if preferred_b is not None:
            # Collision threat active: run combined scorer but let Group B
            # safety layer handle the actual kinematics via run_step()
            action_str = self._best_combined_action(preferred_a, preferred_b)
            # Delegate kinematics + write-back to CollisionMDP.run_step()
            self.mdp.run_step(step_num=self._step_idx)
            label = f"COMB_{action_str}"
        elif preferred_a is not None:
            # TFR threat only: use Group A combined scorer, apply directly
            action_str = self._best_combined_action(preferred_a, None)
            self._apply_group_a_action(action_str)
            label = f"TFR_{action_str}"
        else:
            # Neither solver has a preference — guidance only
            guidance, _ = self._guidance_action(EVASIVE_TURN_DEG)
            self._apply_group_a_action(guidance)
            action_str = guidance
            label = guidance

        self._last_mdp_action = action_str
        return label

    # =========================================================================
    # History / metrics recording (Group A style, extended)
    # =========================================================================

    def _record_history(self, action: str):
        tfr = getattr(self.model, "tfr", None)
        self.history_lat.append(self.lat)
        self.history_lon.append(self.lon)
        self.history_altitude.append(self.altitude)
        self.history_time.append(self.model.current_step)
        self.history_in_tfr.append(self.in_tfr)
        self.history_tfr_distance.append(
            tfr.distance_to_edge(self.lat, self.lon) if tfr else float("inf")
        )
        self.history_action.append(action)

    # =========================================================================
    # Properties for DataCollector (Group B API, unchanged)
    # =========================================================================

    @property
    def n_conflicts(self) -> int:
        if self.mdp is not None:
            return sum(
                1 for key in self.model.detector.get_active_conflicts()
                if self.callsign in key.split("|")
            )
        return len(self.conflicts)

    @property
    def progress_pct(self) -> float:
        return self._step_idx / max(len(self._traj) - 1, 1) * 100

    @property
    def min_tfr_distance(self) -> float:
        """Minimum TFR distance seen so far (for metrics)."""
        return min(self.history_tfr_distance) if self.history_tfr_distance else float("inf")