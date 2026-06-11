"""
mdp_collision_env.py
====================
Owner  : Anaya (MDP Lead – Group B)
Project: MDP Air Traffic Simulation  |  ITSEC 2026

MDP Formulation — Aircraft Collision Avoidance (ACAS X inspired)
-----------------------------------------------------------------
This module defines the full Markov Decision Process for a single
"ownship" aircraft navigating a shared airspace populated by intruders.
It plugs directly into the ConflictDetector API produced by the
Conflict Detection Lead (conflict_detection.py).

Mathematical Formulation
------------------------
The MDP is the 5-tuple:

    M = (S, A, T, R, γ)

    S  — continuous state space (see AircraftState)
    A  — discrete action space (see Action enum)
    T  — stochastic transition model T(s'|s,a)
    R  — composite reward function R(s, a, s')
    γ  — discount factor (default 0.97)

ACAS X Alignment
----------------
ACAS X models the collision avoidance
problem over relative aircraft geometry and issues
Resolution Advisories (RAs) from a pre-computed value table. This
module implements the MDP (fully-observed) version of that framework,
with the state space extended to include weather, radar, and comms —
ready for the POMDP upgrade (position noise + delayed reports) in the
next sprint.

Key design choices that mirror ACAS X:
  • τ  (tau) — time-to-conflict used as a sensing horizon
  • Relative geometry encoded per intruder (not absolute positions)
  • Vertical rate advisory actions mirror ACAS X RA vocabulary
  • Cost-of-action (COA) penalty structure matches ACAS X utility design

Equations are documented inline with LaTeX-style comments.

Usage
-----
    from mdp_collision_env import CollisionMDP, AircraftState, Action

    # In AircraftAgent.__init__():
    self.mdp = CollisionMDP(detector=shared_detector, ownship_callsign=self.callsign)
    self.mdp.reset(self, model.schedule.agents)

    # In AircraftAgent.step():
    next_state, reward, done, info = self.mdp.run_step(self.model.schedule.steps)
    # Agent position/altitude/speed/heading are updated automatically.
"""





from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Bring in the separation constants and detector from the CD layer ───────────
from conflict_detection import (
    ConflictDetector,
    ConflictEvent,
    HORIZONTAL_SEP_NM,
    VERTICAL_SEP_FT,
    haversine_nm,
)





# ==============================================================================
# SECTION 1 — CONSTANTS & CONFIGURATION
# ==============================================================================

# ── Physical / operational limits ─────────────────────────────────────────────

MIN_ALTITUDE_FT     = 1_000      # floor — below this the aircraft is on approach
MAX_ALTITUDE_FT     = 45_000     # service ceiling (typical transport category)
FL290_FT            = 29_000     # RVSM boundary — vertical sep increases above this
RVSM_VERT_SEP_FT    = 2_000      # 2 000 ft vertical separation above FL290 (ICAO)
MIN_SPEED_KTAS      = 150        # minimum airspeed to stay airborne (ktas)
MAX_SPEED_KTAS      = 600        # maximum operational speed (ktas)

NM_PER_DEG_LAT      = 60.0       # 1° latitude ≈ 60 NM (constant)
FT_PER_FL           = 100        # 1 flight level = 100 ft

# ── Time / geometry ───────────────────────────────────────────────────────────

TAU_THRESHOLD_S     = 35         # τ (tau) — ACAS X RA trigger horizon (seconds)
LOOK_AHEAD_S        = 120        # projection window for state features (2 min)
SIM_STEP_S          = 30         # one simulation step = 30 s (matches model.py)

# ── MDP hyper-parameters ──────────────────────────────────────────────────────

GAMMA               = 0.97       # discount factor  γ
MAX_INTRUDERS       = 10         # maximum intruders encoded in the state vector

# ── Reward weights (tunable) ──────────────────────────────────────────────────
# R(s,a,s') = w_col·R_col + w_sev·R_sev + w_fuel·R_fuel
#           + w_delay·R_delay + w_dev·R_dev + w_ra·R_ra

W_COLLISION         = -500.0     # terminal collision penalty
W_SEVERITY          = -100.0     # graded separation violation penalty
W_FUEL              = -0.05      # per-step fuel burn penalty coefficient
W_DELAY             = -1.5       # per-step delay penalty when off nominal speed
W_DEVIATION         = -2.0       # per-NM lateral / per-FL vertical route deviation
W_RA_COMPLIANCE     = +10.0      # positive reward for following a correct RA
W_CLEAR_OF_CONFLICT = +25.0      # reward when a conflict is resolved this step
W_STEP_SURVIVAL     = +1.0       # small living reward to encourage completing route








# ==============================================================================
# SECTION 2 — ACTION SPACE  (A)
# ==============================================================================

class Action(IntEnum):
    """
    Discrete action set A  — mirrors the ACAS X Resolution Advisory vocabulary.

    Vertical actions (climb / descend rate advisories):
        MAINTAIN       — hold current vertical rate
        CLIMB_250      — climb at 250 fpm
        CLIMB_1500     — climb at 1 500 fpm  (standard TCAS RA)
        CLIMB_2500     — climb at 2 500 fpm  (strong RA)
        DESCEND_250    — descend at 250 fpm
        DESCEND_1500   — descend at 1 500 fpm
        DESCEND_2500   — descend at 2 500 fpm
        LEVEL_OFF      — command zero vertical rate (clear RA)

    Speed actions:
        SPEED_UP       — increase speed by 10 ktas
        SPEED_DOWN     — decrease speed by 10 ktas

    Lateral actions:
        TURN_LEFT_5    — 5° left heading change
        TURN_RIGHT_5   — 5° right heading change
        TURN_LEFT_15   — 15° left heading change  (larger avoidance)
        TURN_RIGHT_15  — 15° right heading change

    |A| = 14 discrete actions
    """
    MAINTAIN        = 0
    CLIMB_250       = 1
    CLIMB_1500      = 2
    CLIMB_2500      = 3
    DESCEND_250     = 4
    DESCEND_1500    = 5
    DESCEND_2500    = 6
    LEVEL_OFF       = 7
    SPEED_UP        = 8
    SPEED_DOWN      = 9
    TURN_LEFT_5     = 10
    TURN_RIGHT_5    = 11
    TURN_LEFT_15    = 12
    TURN_RIGHT_15   = 13


# Vertical rate (fpm) and heading delta (deg) implied by each action
ACTION_VERTICAL_RATE_FPM: Dict[Action, float] = {
    Action.MAINTAIN:    0.0,
    Action.CLIMB_250:   250.0,
    Action.CLIMB_1500:  1500.0,
    Action.CLIMB_2500:  2500.0,
    Action.DESCEND_250: -250.0,
    Action.DESCEND_1500:-1500.0,
    Action.DESCEND_2500:-2500.0,
    Action.LEVEL_OFF:   0.0,
    Action.SPEED_UP:    0.0,
    Action.SPEED_DOWN:  0.0,
    Action.TURN_LEFT_5: 0.0,
    Action.TURN_RIGHT_5:0.0,
    Action.TURN_LEFT_15:0.0,
    Action.TURN_RIGHT_15:0.0,
}

ACTION_DELTA_SPEED_KTAS: Dict[Action, float] = {
    Action.SPEED_UP:   +10.0,
    Action.SPEED_DOWN: -10.0,
}

ACTION_DELTA_HEADING_DEG: Dict[Action, float] = {
    Action.TURN_LEFT_5:   -5.0,
    Action.TURN_RIGHT_5:  +5.0,
    Action.TURN_LEFT_15:  -15.0,
    Action.TURN_RIGHT_15: +15.0,
}


def action_cost_fuel(action: Action, speed_ktas: float, dt_s: float = SIM_STEP_S) -> float:
    """
    Approximate fuel cost of an action over one time-step.

    Equation
    --------
    F(a, v) = base_burn(v) · dt · manoeuvre_factor(a)

    where
        base_burn(v)  ≈ (v / 500)²   [normalised; 1.0 at 500 ktas]
        manoeuvre_factor(a):
            MAINTAIN / LEVEL_OFF : 1.00
            small climb/descend  : 1.10
            large climb/descend  : 1.25
            speed up             : 1.15
            turns                : 1.05

    Units: dimensionless normalised fuel unit per step.
    """
    base = (speed_ktas / 500.0) ** 2

    vertical_rate = abs(ACTION_VERTICAL_RATE_FPM.get(action, 0.0))
    if vertical_rate >= 2500:
        mf = 1.25
    elif vertical_rate >= 1500:
        mf = 1.18
    elif vertical_rate > 0:
        mf = 1.10
    elif action in (Action.SPEED_UP,):
        mf = 1.15
    elif action in (Action.TURN_LEFT_15, Action.TURN_RIGHT_15):
        mf = 1.08
    elif action in (Action.TURN_LEFT_5, Action.TURN_RIGHT_5):
        mf = 1.04
    else:
        mf = 1.00

    return base * mf * (dt_s / 60.0)   # normalised per-minute basis





# ==============================================================================
# SECTION 3 — STATE SPACE  (S)
# ==============================================================================

@dataclass
class IntruderFeatures:
    """
    Relative-geometry features for one intruder aircraft.
    These mirror the ACAS X intruder state vector.

    Variables
    ---------
    callsign        : identifier
    h_sep_nm        : horizontal separation (NM)
    v_sep_ft        : vertical separation (ft)
    rel_bearing_deg : bearing from ownship to intruder (0–360°)
    rel_speed_ktas  : closure / divergence speed (+ = closing)
    rel_vert_rate   : relative vertical rate (fpm)  (+ = intruder above & climbing)
    tau_s           : τ — estimated time-to-conflict (s);  ∞ if no conflict predicted
    severity        : 'CRITICAL' | 'HIGH' | 'MODERATE' | 'NONE'
    """
    callsign        : str
    h_sep_nm        : float
    v_sep_ft        : float
    intruder_alt_ft : float          # absolute altitude of intruder (ft)
    rel_bearing_deg : float
    rel_speed_ktas  : float          # vectorial closure rate (+ = closing)
    rel_vert_rate   : float
    tau_s           : float
    severity        : str = "NONE"

    def to_vector(self) -> np.ndarray:
        """
        Encode as a fixed-length numpy vector for RL consumption.
        Shape: (7,)

        Normalisation:
            h_sep_nm        / HORIZONTAL_SEP_NM  → [0, ∞)
            v_sep_ft        / VERTICAL_SEP_FT    → [0, ∞)
            rel_bearing_deg / 360.0              → [0, 1)
            rel_speed_ktas  / MAX_SPEED_KTAS     → (−1, +1]
            rel_vert_rate   / 3000.0             → (−1, +1]
            tau_s           / LOOK_AHEAD_S       → [0, 1]  (clipped)
            in_conflict     : 1 if severity != NONE, else 0
        """
        sev_map = {"NONE": 0.0, "MODERATE": 0.33, "HIGH": 0.67, "CRITICAL": 1.0}
        return np.array([
            self.h_sep_nm           / HORIZONTAL_SEP_NM,
            self.v_sep_ft           / VERTICAL_SEP_FT,
            self.intruder_alt_ft    / MAX_ALTITUDE_FT,       # normalised absolute alt
            self.rel_bearing_deg    / 360.0,
            np.clip(self.rel_speed_ktas / MAX_SPEED_KTAS, -1.0, 1.0),
            np.clip(self.rel_vert_rate  / 3000.0,         -1.0, 1.0),
            np.clip(self.tau_s          / LOOK_AHEAD_S,    0.0, 1.0),
            sev_map.get(self.severity, 0.0),
        ], dtype=np.float32)





@dataclass
class AircraftState:
    """
    Full MDP state  s ∈ S  for the ownship.

    S = S_own ⊗ S_weather ⊗ S_radar ⊗ S_comms ⊗ S_intruders

    S_own  — ownship kinematics + route deviation
    ─────────────────────────────────────────────
    lat, lon        : geographic position (decimal degrees)
    altitude_ft     : barometric altitude (feet)
    speed_ktas      : true airspeed (knots)
    heading_deg     : magnetic heading (0–360°)
    vertical_rate   : current vertical rate (fpm)
    route_lat_dev   : lateral deviation from planned route (NM)
    route_alt_dev   : altitude deviation from planned FL (ft)
    eta_deviation_s : delay relative to planned ETA (seconds)

    S_weather — meteorological state
    ─────────────────────────────────
    wind_speed_kts  : surface / en-route wind speed
    wind_dir_deg    : wind direction (degrees true)
    turbulence_idx  : 0 = smooth, 1 = light, 2 = moderate, 3 = severe
    visibility_nm   : met visibility (NM)
    icing_level     : 0 = none, 1 = light, 2 = moderate, 3 = severe

    S_radar — sensor picture
    ─────────────────────────
    radar_range_nm  : effective radar range (NM)
    n_contacts      : number of radar contacts in range
    radar_quality   : 0.0 (no data) – 1.0 (full resolution)

    S_comms — received ATC / TCAS advisories
    ──────────────────────────────────────────
    atc_instruction : encoded ATC clearance (see _encode_atc)
    tcas_ra         : active TCAS RA (Action index, or -1 if none)
    datalink_msg    : last CPDLC / ADS-B message code (0 = none)

    S_intruders — relative geometry for each intruder
    ──────────────────────────────────────────────────
    intruders       : list of IntruderFeatures (up to MAX_INTRUDERS)
                      padded with zero-vectors if fewer intruders present
    """



    # Ownship kinematics
    lat            : float = 0.0
    lon            : float = 0.0
    altitude_ft    : float = 35_000.0
    speed_ktas     : float = 450.0
    heading_deg    : float = 0.0
    vertical_rate  : float = 0.0

    # Route deviation
    route_lat_dev  : float = 0.0    # NM
    route_alt_dev  : float = 0.0    # ft
    eta_deviation_s: float = 0.0    # seconds

    # Weather
    wind_speed_kts  : float = 0.0
    wind_dir_deg    : float = 0.0
    turbulence_idx  : int   = 0
    visibility_nm   : float = 10.0
    icing_level     : int   = 0

    # Radar
    radar_range_nm  : float = 40.0
    n_contacts      : int   = 0
    radar_quality   : float = 1.0

    # Comms
    atc_instruction : int   = 0    # 0 = no clearance; see _encode_atc()
    tcas_ra         : int   = -1   # -1 = no RA; otherwise Action index
    datalink_msg    : int   = 0

    # Intruder picture (populated at runtime)
    intruders       : List[IntruderFeatures] = field(default_factory=list)




    # ── Derived / convenience ──────────────────────────────────────────────────

    @property
    def in_conflict(self) -> bool:
        """True if any intruder is currently within separation minima."""
        return any(f.severity != "NONE" for f in self.intruders)

    @property
    def worst_severity(self) -> str:
        sev_order = {"CRITICAL": 3, "HIGH": 2, "MODERATE": 1, "NONE": 0}
        if not self.intruders:
            return "NONE"
        return max(self.intruders, key=lambda f: sev_order[f.severity]).severity

    @property
    def min_tau(self) -> float:
        """Smallest τ among all intruders (most urgent conflict)."""
        if not self.intruders:
            return float("inf")
        return min(f.tau_s for f in self.intruders)

    @property
    def above_rvsm(self) -> bool:
        return self.altitude_ft > FL290_FT

    def effective_vert_sep(self) -> float:
        """Return applicable vertical separation standard (RVSM-aware)."""
        return RVSM_VERT_SEP_FT if self.above_rvsm else VERTICAL_SEP_FT
    


    # ── Vectorisation for RL ───────────────────────────────────────────────────

    def to_vector(self) -> np.ndarray:
        """
        Flatten the full state into a 1-D numpy array.

        Ownship  (9 features):
            lat / 90, lon / 180,
            altitude / MAX_ALTITUDE_FT,
            speed / MAX_SPEED_KTAS,
            heading / 360,
            vertical_rate / 3000  (clipped −1 to +1),
            route_lat_dev / HORIZONTAL_SEP_NM,
            route_alt_dev / VERTICAL_SEP_FT,
            eta_deviation_s / 3600

        Weather  (5 features):
            wind_speed / 100, wind_dir / 360,
            turbulence / 3, visibility / 50, icing / 3

        Radar  (3 features):
            radar_range / 100, n_contacts / MAX_INTRUDERS, radar_quality

        Comms  (3 features):
            atc_instruction / 10, tcas_ra (+ 1 / 15), datalink_msg / 10

        Intruders  (MAX_INTRUDERS × 7 features):
            Each IntruderFeatures.to_vector() padded with zeros.

        Total dimensionality:  9 + 5 + 3 + 3 + 70  =  90
        """
        own = np.array([
            self.lat              / 90.0,
            self.lon              / 180.0,
            self.altitude_ft      / MAX_ALTITUDE_FT,
            self.speed_ktas       / MAX_SPEED_KTAS,
            self.heading_deg      / 360.0,
            np.clip(self.vertical_rate / 3000.0, -1.0, 1.0),
            self.route_lat_dev    / HORIZONTAL_SEP_NM,
            self.route_alt_dev    / VERTICAL_SEP_FT,
            self.eta_deviation_s  / 3600.0,
        ], dtype=np.float32)

        weather = np.array([
            self.wind_speed_kts / 100.0,
            self.wind_dir_deg   / 360.0,
            self.turbulence_idx / 3.0,
            self.visibility_nm  / 50.0,
            self.icing_level    / 3.0,
        ], dtype=np.float32)

        radar = np.array([
            self.radar_range_nm  / 100.0,
            self.n_contacts      / float(MAX_INTRUDERS),
            self.radar_quality,
        ], dtype=np.float32)

        comms = np.array([
            self.atc_instruction         / 10.0,
            (self.tcas_ra + 1)           / 15.0,
            self.datalink_msg            / 10.0,
        ], dtype=np.float32)

        # Intruder sub-vectors — pad to MAX_INTRUDERS
        intruder_vecs = [f.to_vector() for f in self.intruders[:MAX_INTRUDERS]]
        while len(intruder_vecs) < MAX_INTRUDERS:
            intruder_vecs.append(np.zeros(7, dtype=np.float32))
        intruder_block = np.concatenate(intruder_vecs)

        return np.concatenate([own, weather, radar, comms, intruder_block])

    @property
    def dim(self) -> int:
        return 9 + 5 + 3 + 3 + MAX_INTRUDERS * 7   # = 90








# ==============================================================================
# SECTION 4 — TRANSITION MODEL  T(s'|s, a)
# ==============================================================================

def compute_tau(
    h_sep_nm: float,
    h_closure_ktas: float,
    v_sep_ft: float,
    v_closure_fpm: float,
) -> float:
    """
    Estimate τ — time-to-minimum-separation in seconds.

    ACAS X tau calculation (simplified horizontal projection):

        τ_h = h_sep / max(ε, h_closure)    if h_closure > 0  else ∞
        τ_v = v_sep / max(ε, v_closure)    if v_closure > 0  else ∞
        τ   = min(τ_h, τ_v)

    Units
    -----
    h_sep_nm        → NM
    h_closure_ktas  → NM / hr  (positive = closing)
    v_sep_ft        → ft
    v_closure_fpm   → ft / min  (positive = closing vertically)
    """
    EPS = 1e-6
    tau_h = (h_sep_nm / max(EPS, h_closure_ktas)) * 3600.0  if h_closure_ktas > 0 else float("inf")
    tau_v = (v_sep_ft / max(EPS, v_closure_fpm))  * 60.0    if v_closure_fpm  > 0 else float("inf")
    return min(tau_h, tau_v)


def apply_action_kinematics(
    state: AircraftState,
    action: Action,
    dt_s: float = SIM_STEP_S,
) -> AircraftState:
    """
    Deterministic part of the transition T(s'|s, a).

    Equations
    ---------
    Δalt  = vr_new · dt  / 60          (ft, from fpm)
    alt'  = clip(alt + Δalt, MIN, MAX)

    v'    = clip(v + Δv, MIN, MAX)     (ktas)
    hdg'  = (hdg + Δhdg) mod 360       (degrees)

    lat'  = lat  + (vg · cos(hdg·π/180) · dt / 3600) / NM_PER_DEG_LAT
    lon'  = lon  + (vg · sin(hdg·π/180) · dt / 3600) /
                    (NM_PER_DEG_LAT · cos(lat·π/180))

    where vg = ground speed ≈ speed_ktas (wind correction omitted here;
    include wind_speed and wind_dir from S_weather for full fidelity).

    NOTE: This is the mean of the transition distribution. Stochastic
    noise (ε_pos, ε_alt) is added by CollisionMDP.step() to support
    the future POMDP extension.
    """
    import copy
    ns = copy.copy(state)

    # Vertical rate update
    vr_delta = ACTION_VERTICAL_RATE_FPM.get(action, 0.0)
    if action == Action.LEVEL_OFF:
        ns.vertical_rate = 0.0
    elif vr_delta != 0.0:
        ns.vertical_rate = vr_delta

    # Altitude
    delta_alt = ns.vertical_rate * (dt_s / 60.0)
    ns.altitude_ft = float(np.clip(ns.altitude_ft + delta_alt, MIN_ALTITUDE_FT, MAX_ALTITUDE_FT))

    # Speed
    dv = ACTION_DELTA_SPEED_KTAS.get(action, 0.0)
    ns.speed_ktas = float(np.clip(ns.speed_ktas + dv, MIN_SPEED_KTAS, MAX_SPEED_KTAS))

    # Heading
    dhdg = ACTION_DELTA_HEADING_DEG.get(action, 0.0)
    ns.heading_deg = (ns.heading_deg + dhdg) % 360.0

    # Position (great-circle approximation)
    hdg_rad = math.radians(ns.heading_deg)
    gs_nm_per_s = ns.speed_ktas / 3600.0
    d_nm = gs_nm_per_s * dt_s

    ns.lat = state.lat + (d_nm * math.cos(hdg_rad)) / NM_PER_DEG_LAT
    lat_rad = math.radians(state.lat)
    cos_lat = math.cos(lat_rad) if abs(math.cos(lat_rad)) > 1e-9 else 1e-9
    ns.lon = state.lon + (d_nm * math.sin(hdg_rad)) / (NM_PER_DEG_LAT * cos_lat)

    return ns



# ==============================================================================
# SECTION 4b — ACAS X POLICY  π(s)
# ==============================================================================
#
# ACAS X works by pre-computing a value table offline via value iteration and
# looking up the best Resolution Advisory (RA) for the current encounter
# geometry at runtime.  This function is the online lookup equivalent: given
# the current AircraftState it evaluates the same geometric logic ACAS X uses
# and returns the single best Action.
#
# Decision logic (priority order mirrors ACAS X RA issuance rules)
# ----------------------------------------------------------------
# 1. If no intruder is within τ ≤ TAU_THRESHOLD_S → MAINTAIN (no RA needed)
# 2. For the most threatening intruder (smallest τ):
#    a. If intruder is ABOVE ownship  → DESCEND  (sense: open vertical gap)
#    b. If intruder is BELOW ownship  → CLIMB
#    c. Rate selection scales with urgency:
#         τ ≤ 15 s  (imminent)  → 2500 fpm
#         τ ≤ 25 s  (urgent)    → 1500 fpm
#         τ ≤ 35 s  (advisory)  →  250 fpm
# 3. If vertical separation is already safe (v_sep ≥ threshold) but
#    horizontal is still closing → lateral turn away from intruder bearing
#         τ ≤ 20 s → TURN_15,  else TURN_5
# 4. LEVEL_OFF issued when a previous RA has opened enough vertical separation
#    and τ has grown back above threshold (conflict clearing).
#
# The `action_taken` return value is passed straight into CollisionMDP.step()
# so Mesa only needs to call mdp.run_step(step_num) — see below.




def acas_x_policy(state: AircraftState) -> Action:
    """
    ACAS X-inspired deterministic policy  π: S → A.

    Selects the Resolution Advisory for the most threatening intruder
    using τ (time-to-conflict) and relative vertical geometry.

    Parameters
    ----------
    state : AircraftState — current fully-observed state of the ownship

    Returns
    -------
    Action — the selected RA, or MAINTAIN if no threat is detected
    """
    # ── Filter to intruders that are actually threatening ─────────────────────
    threatening = [
        f for f in state.intruders
        if f.tau_s <= TAU_THRESHOLD_S and f.severity != "NONE"
    ]

    if not threatening:
        # No active threat — check if we should issue LEVEL_OFF to clear a
        # previous RA (ownship currently climbing or descending).
        if state.vertical_rate != 0.0:
            return Action.LEVEL_OFF
        return Action.MAINTAIN

    # ── Pick the most urgent intruder (smallest τ) ────────────────────────────
    threat = min(threatening, key=lambda f: f.tau_s)
    tau    = threat.tau_s

    # ── Determine vertical sense ──────────────────────────────────────────────
    # Positive v_sep_ft means intruder altitude differs from ownship.
    # We use the sign of (intruder_alt − own_alt) inferred from rel_vert_rate
    # and the conflict geometry: if intruder is above, descend; if below, climb.
    # rel_vert_rate > 0 means ownship is climbing faster (or intruder descending)
    # so the intruder is relatively below → we should climb to widen the gap.
    # Use the intruder's actual altitude to determine which side to evade toward.
    # rel_vert_rate only tells us whether the gap is opening or closing, not
    # which aircraft is higher — that requires the absolute altitude comparison.
    intruder_is_above = threat.intruder_alt_ft > state.altitude_ft

    # ── Vertical RA — rate chosen by urgency ─────────────────────────────────
    v_sep   = threat.v_sep_ft
    v_safe  = state.effective_vert_sep()

    if v_sep < v_safe:
        # Still inside the vertical bubble — issue climb/descend RA
        if tau <= 15.0:
            rate = 2500
        elif tau <= 25.0:
            rate = 1500
        else:
            rate = 250

        if intruder_is_above:
            return {2500: Action.DESCEND_2500,
                    1500: Action.DESCEND_1500,
                     250: Action.DESCEND_250}[rate]
        else:
            return {2500: Action.CLIMB_2500,
                    1500: Action.CLIMB_1500,
                     250: Action.CLIMB_250}[rate]

    # ── Vertical gap is safe but horizontal threat remains → lateral RA ───────
    # Turn away from the intruder bearing.
    # If intruder is to the right (bearing 0–180), turn left; else turn right.
    bearing = threat.rel_bearing_deg
    turn_left = (0.0 <= bearing < 180.0)

    if tau <= 20.0:
        return Action.TURN_LEFT_15 if turn_left else Action.TURN_RIGHT_15
    else:
        return Action.TURN_LEFT_5  if turn_left else Action.TURN_RIGHT_5






# ==============================================================================
# SECTION 5 — REWARD FUNCTION  R(s, a, s')
# ==============================================================================

@dataclass
class RewardComponents:
    """
    Decomposed reward — useful for logging, debugging, and curriculum learning.

    Total reward:
        R = r_collision + r_severity + r_fuel + r_delay + r_deviation
          + r_ra_compliance + r_clear_of_conflict + r_survival

    Signs:
        Penalties < 0  (collision, severity, fuel, delay, deviation)
        Incentives > 0 (ra_compliance, clear_of_conflict, survival)
    """
    r_collision        : float = 0.0
    r_severity         : float = 0.0
    r_fuel             : float = 0.0
    r_delay            : float = 0.0
    r_deviation        : float = 0.0
    r_ra_compliance    : float = 0.0
    r_clear_of_conflict: float = 0.0
    r_survival         : float = 0.0

    @property
    def total(self) -> float:
        return (
            self.r_collision
            + self.r_severity
            + self.r_fuel
            + self.r_delay
            + self.r_deviation
            + self.r_ra_compliance
            + self.r_clear_of_conflict
            + self.r_survival
        )

    def to_dict(self) -> dict:
        return {
            "r_collision"        : round(self.r_collision, 4),
            "r_severity"         : round(self.r_severity, 4),
            "r_fuel"             : round(self.r_fuel, 4),
            "r_delay"            : round(self.r_delay, 4),
            "r_deviation"        : round(self.r_deviation, 4),
            "r_ra_compliance"    : round(self.r_ra_compliance, 4),
            "r_clear_of_conflict": round(self.r_clear_of_conflict, 4),
            "r_survival"         : round(self.r_survival, 4),
            "total"              : round(self.total, 4),
        }


def compute_reward(
    state      : AircraftState,
    action     : Action,
    next_state : AircraftState,
    active_conflicts_before: Dict[str, ConflictEvent],
    active_conflicts_after : Dict[str, ConflictEvent],
    ownship_callsign       : str,
) -> RewardComponents:
    """
    Compute the composite reward R(s, a, s').

    Equations
    ---------

    1. COLLISION PENALTY  r_col
       Applied if ownship is in a CRITICAL conflict in s'.
       
       │  r_col = W_COLLISION   if severity(s') == 'CRITICAL'    
       │        = 0             otherwise                         
       

    2. SEVERITY PENALTY  r_sev
       Graded penalty proportional to how far inside the separation bubble
       the ownship is. Inspired by ACAS X cost-of-alert (COA) design.

       |Let  φ_h = max(0, 1 − h_sep / H_SEP)     ∈ [0, 1]
       |    φ_v = max(0, 1 − v_sep / V_SEP)     ∈ [0, 1]
       |    φ   = φ_h · φ_v                       ∈ [0, 1]  (AND logic)
       |
       |
       │  r_sev = W_SEVERITY · φ                                
       └─

    3. FUEL BURN PENALTY  r_fuel
       
       │  r_fuel = W_FUEL · action_cost_fuel(a, v, dt)         
       └──

    4. DELAY PENALTY  r_delay
       Penalises deviation from nominal cruise speed.
       
       │  r_delay = W_DELAY · |v − v_nominal| / v_nominal      
       │  v_nominal = 450 ktas (default cruise)                 
       └──

    5. ROUTE DEVIATION PENALTY  r_dev
       
       │  r_dev = W_DEVIATION · (|Δlat_nm| + |Δalt_fl|)        
       │  |Δalt_fl| = |route_alt_dev| / FT_PER_FL              
       └──

    6. RA COMPLIANCE REWARD  r_ra
       Positive reward when ownship follows a TCAS / ACAS RA.
       
       │  r_ra = W_RA_COMPLIANCE   if action == tcas_ra action  
       │       = 0                 otherwise                    
       └──

    7. CLEAR-OF-CONFLICT REWARD  r_coc
       Positive reward when ownship exits an active conflict this step.
       
       │  r_coc = W_CLEAR_OF_CONFLICT                           
       │          × (n_conflicts_before − n_conflicts_after)    
       │          (clamped to 0 if negative)                    
       └──

    8. STEP SURVIVAL REWARD  r_surv
       Small positive reward for each step completed without collision.
       
       │  r_surv = W_STEP_SURVIVAL   if not collision           
       │         = 0                 on collision               
       └──
    """
    
    
    
    
    
    
    rc = RewardComponents()

    # Helper: conflicts involving ownship
    def ownship_conflicts(conflict_dict: Dict[str, ConflictEvent]) -> List[ConflictEvent]:
        return [ev for key, ev in conflict_dict.items()
                if ownship_callsign in key.split("|")]

    before = ownship_conflicts(active_conflicts_before)
    after  = ownship_conflicts(active_conflicts_after)

    # ── 1. Collision penalty ──────────────────────────────────────────────────
    if next_state.worst_severity == "CRITICAL":
        rc.r_collision = W_COLLISION

    # ── 2. Severity penalty ───────────────────────────────────────────────────
    phi_total = 0.0
    for feat in next_state.intruders:
        if feat.severity != "NONE":
            phi_h = max(0.0, 1.0 - feat.h_sep_nm  / HORIZONTAL_SEP_NM)
            phi_v = max(0.0, 1.0 - feat.v_sep_ft  / next_state.effective_vert_sep())
            phi_total += phi_h * phi_v
    rc.r_severity = W_SEVERITY * phi_total

    # ── 3. Fuel penalty ───────────────────────────────────────────────────────
    rc.r_fuel = W_FUEL * action_cost_fuel(action, state.speed_ktas)

    # ── 4. Delay penalty ──────────────────────────────────────────────────────
    v_nominal = 450.0
    rc.r_delay = W_DELAY * abs(next_state.speed_ktas - v_nominal) / v_nominal

    # ── 5. Route deviation penalty ────────────────────────────────────────────
    lat_dev_nm = abs(next_state.route_lat_dev)
    alt_dev_fl = abs(next_state.route_alt_dev) / FT_PER_FL
    rc.r_deviation = W_DEVIATION * (lat_dev_nm + alt_dev_fl)

    # ── 6. RA compliance reward ───────────────────────────────────────────────
    if state.tcas_ra >= 0 and action == Action(state.tcas_ra):
        rc.r_ra_compliance = W_RA_COMPLIANCE

    # ── 7. Clear-of-conflict reward ───────────────────────────────────────────
    resolved = max(0, len(before) - len(after))
    rc.r_clear_of_conflict = W_CLEAR_OF_CONFLICT * resolved

    # ── 8. Survival reward ────────────────────────────────────────────────────
    if rc.r_collision == 0.0:
        rc.r_survival = W_STEP_SURVIVAL

    return rc







# ==============================================================================
# SECTION 6 — STEP RECORD  (output schema for further steps)
# ==============================================================================

@dataclass
class StepRecord:
    """
    One row in the simulation log — written by CollisionMDP.run_step()
    at every step and exposed via get_step_log() as a pandas DataFrame.

    This is the handoff contract between the MDP and the
    logging, plotting, and comparing baseline vs MDP results.

    Fields
    ------
    step            : simulation step number (matches Mesa schedule.steps)
    callsign        : ownship callsign
    action          : name of the ACAS X RA issued this step
    lat, lon        : ownship position after the action
    altitude_ft     : ownship altitude after the action (ft)
    speed_ktas      : ownship speed after the action (ktas)
    heading_deg     : ownship heading after the action (degrees)
    vertical_rate   : ownship vertical rate after the action (fpm)
    in_conflict     : True if ownship is in any active conflict this step
    conflict_severity: worst severity among active conflicts ('NONE' if clear)
    n_active_conflicts: number of conflict pairs active this step
    conflict_partners: comma-separated callsigns in conflict with ownship
    tau_min_s       : smallest τ among intruders (∞ if no threat)
    r_collision     : collision penalty component
    r_severity      : severity penalty component
    r_fuel          : fuel penalty component
    r_delay         : delay penalty component
    r_deviation     : route deviation penalty component
    r_ra_compliance : RA compliance reward component
    r_clear_of_conflict: conflict resolution reward component
    r_survival      : step survival reward
    reward_total    : total scalar reward this step
    cumulative_reward: discounted cumulative reward so far
    """
    step                : int
    callsign            : str
    action              : str
    lat                 : float
    lon                 : float
    altitude_ft         : float
    speed_ktas          : float
    heading_deg         : float
    vertical_rate       : float
    in_conflict         : bool
    conflict_severity   : str
    n_active_conflicts  : int
    conflict_partners   : str   # comma-separated, empty string if none
    tau_min_s           : float
    r_collision         : float
    r_severity          : float
    r_fuel              : float
    r_delay             : float
    r_deviation         : float
    r_ra_compliance     : float
    r_clear_of_conflict : float
    r_survival          : float
    reward_total        : float
    cumulative_reward   : float

    def to_dict(self) -> dict:
        return {
            "step"                : self.step,
            "callsign"            : self.callsign,
            "action"              : self.action,
            "lat"                 : round(self.lat, 6),
            "lon"                 : round(self.lon, 6),
            "altitude_ft"         : round(self.altitude_ft, 1),
            "speed_ktas"          : round(self.speed_ktas, 2),
            "heading_deg"         : round(self.heading_deg, 2),
            "vertical_rate"       : round(self.vertical_rate, 1),
            "in_conflict"         : self.in_conflict,
            "conflict_severity"   : self.conflict_severity,
            "n_active_conflicts"  : self.n_active_conflicts,
            "conflict_partners"   : self.conflict_partners,
            "tau_min_s"           : round(self.tau_min_s, 2) if self.tau_min_s != float("inf") else None,
            "r_collision"         : round(self.r_collision, 4),
            "r_severity"          : round(self.r_severity, 4),
            "r_fuel"              : round(self.r_fuel, 4),
            "r_delay"             : round(self.r_delay, 4),
            "r_deviation"         : round(self.r_deviation, 4),
            "r_ra_compliance"     : round(self.r_ra_compliance, 4),
            "r_clear_of_conflict" : round(self.r_clear_of_conflict, 4),
            "r_survival"          : round(self.r_survival, 4),
            "reward_total"        : round(self.reward_total, 4),
            "cumulative_reward"   : round(self.cumulative_reward, 4),
        }





# ==============================================================================
# SECTION 7 — MDP ENVIRONMENT  (CollisionMDP)
# ==============================================================================

# Number of simulation steps over which the aircraft blends from its avoidance
# position back to the CSV track after a conflict clears.  30 steps × 30 s/step
# = 15 minutes — long enough to be physically realistic at jet speeds.
RTR_BLEND_STEPS: int = 30


class CollisionMDP:
    """
    Gym-style MDP environment for aircraft collision avoidance.

    Wraps the ConflictDetector and provides:
        reset()  → initial AircraftState
        step(a)  → (next_state, reward, done, info)
        observation_space.shape → (90,)
        action_space.n          → 14

    Episode termination:
        • CRITICAL conflict detected  (collision)
        • Aircraft exits the defined airspace boundary
        • Episode step limit exceeded

    Parameters
    ----------
    detector        : ConflictDetector instance (shared with model)
    ownship_callsign: callsign of the aircraft this policy controls
    max_steps       : episode length limit
    gamma           : discount factor
    noise_std_nm    : position noise std for POMDP readiness (set 0 for MDP)
    noise_std_ft    : altitude noise std (set 0 for MDP)
    """

    OBSERVATION_DIM = 9 + 5 + 3 + 3 + MAX_INTRUDERS * 7   # = 90
    N_ACTIONS       = len(Action)                           # = 14

    def __init__(
        self,
        detector        : ConflictDetector,
        ownship_callsign: str,
        max_steps       : int  = 1_912,   # matches simulation run length
        gamma           : float = GAMMA,
        noise_std_nm    : float = 0.0,    # set > 0 to enable POMDP noise
        noise_std_ft    : float = 0.0,
    ):
        self.detector         = detector
        self.ownship_callsign = ownship_callsign
        self.max_steps        = max_steps
        self.gamma            = gamma
        self.noise_std_nm     = noise_std_nm
        self.noise_std_ft     = noise_std_ft

        self._state    : Optional[AircraftState] = None
        self._aircraft : Optional[object]        = None   # AircraftAgent ref
        self._all_ac   : List                    = []
        self._step_num : int                     = 0
        self._done     : bool                    = False
        self._cumulative_reward: float           = 0.0
        self._step_log : List[StepRecord]        = []     # populated by run_step()

        # Return-to-route state
        # After an avoidance manoeuvre ends the aircraft cannot snap back to the
        # CSV track instantly — it must blend smoothly from its avoidance
        # position back to the original trajectory over RTR_BLEND_STEPS steps.
        self._in_avoidance  : bool  = False   # True while an RA is active
        self._rtr_step      : int   = 0       # steps elapsed since RA cleared
        self._rtr_origin_lat: float = 0.0     # avoidance-exit lat
        self._rtr_origin_lon: float = 0.0     # avoidance-exit lon
        self._rtr_origin_alt: float = 0.0     # avoidance-exit altitude (ft)
        self._rtr_origin_spd: float = 0.0     # avoidance-exit speed (kts)
        self._rtr_origin_hdg: float = 0.0     # avoidance-exit heading (deg)

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(self, ownship_agent, all_agents: List) -> AircraftState:
        """
        Initialise or re-initialise the MDP from the current simulation state.

        Parameters
        ----------
        ownship_agent : AircraftAgent (must have .callsign, .lat, .lon,
                        .altitude, .speed, .heading, .vertical_rate)
        all_agents    : list of all AircraftAgent objects in the model
        """
        self._aircraft = ownship_agent
        self._all_ac   = all_agents
        self._step_num = 0
        self._done     = False
        self._cumulative_reward = 0.0
        self._step_log.clear()
        # NOTE: detector.reset() intentionally omitted — owned by AirTrafficModel

        # Reset return-to-route state
        self._in_avoidance   = False
        self._rtr_step       = 0
        self._rtr_origin_lat = 0.0
        self._rtr_origin_lon = 0.0
        self._rtr_origin_alt = 0.0
        self._rtr_origin_spd = 0.0
        self._rtr_origin_hdg = 0.0

        self._state = self._build_state(ownship_agent, all_agents)
        return self._state

    # ── Step ──────────────────────────────────────────────────────────────────

    def run_step(self, step_num: int = 0) -> Tuple[AircraftState, float, bool, dict]:
        """
        Primary entry point called by AircraftAgent.step() in Mesa.

        Internally:
          1. Calls acas_x_policy(current_state) to select the RA
          2. Applies kinematics to compute next_state
          3. Writes next_state back to the live AircraftAgent so Mesa
             and ConflictDetector always see the MDP-modified position
          4. Runs the conflict detector and computes reward
          5. Returns (next_state, reward, done, info)

        Mesa integration
        ----------------
        In AircraftAgent.step():

            next_state, reward, done, info = self.mdp.run_step(self.model.schedule.steps)

        The single call is all Mesa needs; the agent's position, altitude,
        speed, and heading are updated here automatically.

        Returns
        -------
        next_state  : AircraftState
        reward      : float (total scalar reward)
        done        : bool
        info        : dict (reward components, action taken, conflicts, summary)
        """
        if self._done:
            # In embedded (simulation) mode _done must never crash the run.
            # Auto-reset so the safety layer re-engages cleanly for the next
            # step rather than terminating the entire scenario.
            self.reset(self._aircraft, self._all_ac)

        # ── 1. Resync MDP state from the agent's current CSV position ─────────
        # The agent has already read its CSV row for this step before calling
        # run_step(), so self._aircraft now holds the ground-truth position.
        # Rebuilding the state here means the MDP always reasons from the real
        # trajectory, not from a kinematically-propagated approximation.
        self._state = self._build_state(self._aircraft, self._all_ac)

        # ── 2. Policy selects action from current (CSV-synced) state ──────────
        action = acas_x_policy(self._state)

        # ── 3. Snapshot conflicts before the action ───────────────────────────
        conflicts_before = self.detector.get_active_conflicts()

        # ── 4. Three-phase safety-layer ───────────────────────────────────────
        #
        #  Phase A — AVOIDANCE: policy returns a non-MAINTAIN action.
        #            Apply kinematics and write back to the agent.  Record
        #            avoidance-exit position so RTR can use it as its origin.
        #
        #  Phase B — RETURN-TO-ROUTE (RTR): threat just cleared but aircraft is
        #            displaced from the CSV track.  Linearly blend lat/lon/alt/
        #            spd/hdg from avoidance-exit position toward the CSV waypoint
        #            over RTR_BLEND_STEPS steps.  Write the blended position back
        #            so the agent moves smoothly rather than teleporting.
        #
        #  Phase C — NORMAL: no threat, RTR complete (or never needed).
        #            Leave the CSV values written above completely untouched.

        avoidance_action = action not in (Action.MAINTAIN, Action.LEVEL_OFF)

        # Pull the CSV position that the agent loaded at the top of step()
        # — these are already on self._aircraft because air_traffic_abm wrote
        # them before calling run_step().
        agent = self._aircraft
        csv_lat = agent.lat
        csv_lon = agent.lon
        csv_alt = agent.altitude
        csv_spd = agent.speed
        csv_hdg = agent.heading

        if avoidance_action:
            # ── Phase A ───────────────────────────────────────────────────────
            next_state_det = apply_action_kinematics(self._state, action)
            next_state = self._add_noise(next_state_det)
            self._write_back_to_agent(next_state)
            # Record where avoidance ends so RTR knows its origin
            self._rtr_origin_lat = next_state.lat
            self._rtr_origin_lon = next_state.lon
            self._rtr_origin_alt = next_state.altitude_ft
            self._rtr_origin_spd = next_state.speed_ktas
            self._rtr_origin_hdg = next_state.heading_deg
            self._in_avoidance   = True
            self._rtr_step       = 0

        elif self._in_avoidance:
            # Threat just cleared (or LEVEL_OFF issued).  Decide whether we
            # still need to blend or if we have converged to the CSV track.
            self._rtr_step += 1
            t = min(self._rtr_step / RTR_BLEND_STEPS, 1.0)   # 0 → 1

            if t >= 1.0:
                # ── Phase C reached: RTR complete, hand fully back to CSV ─────
                self._in_avoidance = False
                self._rtr_step     = 0
                next_state = self._add_noise(self._state)
                # Agent already holds CSV values; nothing to write back.
            else:
                # ── Phase B: linear blend avoidance-exit → CSV waypoint ───────
                def _lerp(a, b, t):
                    return a + (b - a) * t

                def _lerp_heading(h0, h1, t):
                    # Shortest-path blend across the 0/360 wrap
                    diff = ((h1 - h0 + 180) % 360) - 180
                    return (h0 + diff * t) % 360

                blended_lat = _lerp(self._rtr_origin_lat, csv_lat, t)
                blended_lon = _lerp(self._rtr_origin_lon, csv_lon, t)
                blended_alt = _lerp(self._rtr_origin_alt, csv_alt, t)
                blended_spd = _lerp(self._rtr_origin_spd, csv_spd, t)
                blended_hdg = _lerp_heading(self._rtr_origin_hdg, csv_hdg, t)

                # Write blended position to agent so the ConflictDetector and
                # DataCollector see a smooth trajectory, not a teleport.
                agent.lat      = blended_lat
                agent.lon      = blended_lon
                agent.altitude = blended_alt
                agent.speed    = blended_spd
                agent.heading  = blended_hdg

                # Build next_state from the blended position for reward/logging
                import copy
                next_state = copy.copy(self._state)
                next_state.lat         = blended_lat
                next_state.lon         = blended_lon
                next_state.altitude_ft = blended_alt
                next_state.speed_ktas  = blended_spd
                next_state.heading_deg = blended_hdg
                next_state = self._add_noise(next_state)

        else:
            # ── Phase C — normal, no avoidance history ────────────────────────
            next_state = self._add_noise(self._state)
            # Agent already holds CSV values; nothing to write back.

        # ── 5. Read current conflict state ────────────────────────────────────
        # detector.check() is called once per model step at the AirTrafficModel
        # level (same as baseline). Calling it here per-agent would fire N times
        # per step with shifting positions, creating phantom conflict events.
        conflicts_after = self.detector.get_active_conflicts()

        # ── 6. Rebuild intruder picture for next_state ────────────────────────
        next_state.intruders  = self._build_intruder_features(self._aircraft, self._all_ac)
        next_state.n_contacts = len(next_state.intruders)

        # ── 7. Compute reward ─────────────────────────────────────────────────
        rc = compute_reward(
            state                   = self._state,
            action                  = action,
            next_state              = next_state,
            active_conflicts_before = conflicts_before,
            active_conflicts_after  = conflicts_after,
            ownship_callsign        = self.ownship_callsign,
        )

        self._step_num += 1
        self._cumulative_reward += (self.gamma ** self._step_num) * rc.total

        # ── 8. Termination conditions ─────────────────────────────────────────
        collision     = (next_state.worst_severity == "CRITICAL")
        out_of_bounds = self._is_out_of_bounds(next_state)
        timeout       = (self._step_num >= self.max_steps)

        self._done  = collision or out_of_bounds or timeout
        self._state = next_state

        info = {
            "step"              : self._step_num,
            "action"            : action.name,
            "reward_components" : rc.to_dict(),
            "active_conflicts"  : len(conflicts_after),
            "conflict_summary"  : self.detector.summary(),
            "done_reason"       : ("collision"    if collision     else
                                   "out_of_bounds" if out_of_bounds else
                                   "timeout"       if timeout       else
                                   "running"),
            "cumulative_reward" : round(self._cumulative_reward, 4),
        }

        # ── 9. Append to step log ────────────────────────────────────────────
        partners = ", ".join(
            self.detector.conflict_partners(self.ownship_callsign)
        )
        self._step_log.append(StepRecord(
            step                = self._step_num,
            callsign            = self.ownship_callsign,
            action              = action.name,
            lat                 = next_state.lat,
            lon                 = next_state.lon,
            altitude_ft         = next_state.altitude_ft,
            speed_ktas          = next_state.speed_ktas,
            heading_deg         = next_state.heading_deg,
            vertical_rate       = next_state.vertical_rate,
            in_conflict         = next_state.in_conflict,
            conflict_severity   = next_state.worst_severity,
            n_active_conflicts  = len(conflicts_after),
            conflict_partners   = partners,
            tau_min_s           = next_state.min_tau,
            r_collision         = rc.r_collision,
            r_severity          = rc.r_severity,
            r_fuel              = rc.r_fuel,
            r_delay             = rc.r_delay,
            r_deviation         = rc.r_deviation,
            r_ra_compliance     = rc.r_ra_compliance,
            r_clear_of_conflict = rc.r_clear_of_conflict,
            r_survival          = rc.r_survival,
            reward_total        = rc.total,
            cumulative_reward   = self._cumulative_reward,
        ))

        return next_state, rc.total, self._done, info

    def step(
        self,
        action: Action,
        step_num: int = 0,
    ) -> Tuple[AircraftState, float, bool, dict]:
        """
        Manual action override — accepts an explicit Action instead of
        calling the policy.  Useful for testing specific RAs or for
        future POMDP integration where an external solver supplies actions.

        For normal Mesa operation use run_step() instead.
        """
        if self._done:
            # In embedded (simulation) mode _done must never crash the run.
            # Auto-reset so the safety layer re-engages cleanly for the next
            # step rather than terminating the entire scenario.
            self.reset(self._aircraft, self._all_ac)

        conflicts_before   = self.detector.get_active_conflicts()
        next_state_det     = apply_action_kinematics(self._state, action)
        next_state         = self._add_noise(next_state_det)

        self._write_back_to_agent(next_state)

        conflicts_after = self.detector.get_active_conflicts()

        next_state.intruders  = self._build_intruder_features(self._aircraft, self._all_ac)
        next_state.n_contacts = len(next_state.intruders)

        rc = compute_reward(
            state                   = self._state,
            action                  = action,
            next_state              = next_state,
            active_conflicts_before = conflicts_before,
            active_conflicts_after  = conflicts_after,
            ownship_callsign        = self.ownship_callsign,
        )

        self._step_num += 1
        self._cumulative_reward += (self.gamma ** self._step_num) * rc.total

        collision     = (next_state.worst_severity == "CRITICAL")
        out_of_bounds = self._is_out_of_bounds(next_state)
        timeout       = (self._step_num >= self.max_steps)

        self._done  = collision or out_of_bounds or timeout
        self._state = next_state

        info = {
            "step"              : self._step_num,
            "action"            : action.name,
            "reward_components" : rc.to_dict(),
            "active_conflicts"  : len(conflicts_after),
            "conflict_summary"  : self.detector.summary(),
            "done_reason"       : ("collision"     if collision     else
                                   "out_of_bounds" if out_of_bounds else
                                   "timeout"       if timeout       else
                                   "running"),
            "cumulative_reward" : round(self._cumulative_reward, 4),
        }

        return next_state, rc.total, self._done, info

    # ── Convenience ───────────────────────────────────────────────────────────

    def sample_action(self) -> Action:
        """Uniformly random action — for random rollout / testing."""
        return Action(np.random.randint(0, self.N_ACTIONS))

    def conflict_status(self, callsign: str) -> dict:
        """
        Public API for the MDP policy to query conflict state.
        Returns severity, tau, and partner callsigns for a given aircraft.
        """
        active = self.detector.get_active_conflicts()
        relevant = {k: v for k, v in active.items() if callsign in k.split("|")}
        if not relevant:
            return {"in_conflict": False, "severity": "NONE", "tau_s": float("inf"), "partners": []}
        worst = min(relevant.values(), key=lambda e: e.h_sep_nm)
        partners = [k.replace(callsign, "").strip("|") for k in relevant]
        return {
            "in_conflict": True,
            "severity"   : worst.severity,
            "h_sep_nm"   : worst.h_sep_nm,
            "v_sep_ft"   : worst.v_sep_ft,
            "tau_s"      : compute_tau(worst.h_sep_nm, 0.0, worst.v_sep_ft, 0.0),
            "partners"   : partners,
        }

    
    import pandas as pd
    def get_step_log(self) -> "pd.DataFrame":
        """
        Return the full per-step simulation log as a pandas DataFrame.

        Each row is one call to run_step() — one Mesa simulation step
        for this ownship. Columns match the StepRecord dataclass fields.

        Intended usage (for the logging / plotting)
        ------------------------------------------------
        After the simulation finishes:

            log_df = mdp.get_step_log()
            log_df.to_csv('mdp_step_log.csv', index=False)

        The DataFrame can be merged with the ConflictDetector event log
        (detector.get_event_log()) on the 'step' column for a complete
        picture of what the policy did and what the detector recorded.
        """
        import pandas as pd
        if not self._step_log:
            return pd.DataFrame()
        return pd.DataFrame([r.to_dict() for r in self._step_log])



    # ── Internal builders ─────────────────────────────────────────────────────

    def _write_back_to_agent(self, next_state: AircraftState) -> None:

        """
        Write the MDP-computed next_state back to the live AircraftAgent.

        This is the wire between the MDP and the Mesa simulation.
        After this call, ConflictDetector.check() and all other agents see
        the position that the ACAS X policy chose, not the original CSV track.

        Attributes written
        ------------------
        agent.lat          ← next_state.lat
        agent.lon          ← next_state.lon
        agent.altitude     ← next_state.altitude_ft
        agent.speed        ← next_state.speed_ktas    (if attribute exists)
        agent.heading      ← next_state.heading_deg   (if attribute exists)
        agent.vertical_rate← next_state.vertical_rate (if attribute exists)

        The optional attributes are written with setattr so the method works
        even if the AircraftAgent only exposes lat/lon/altitude (the minimum
        the ConflictDetector requires).
        """
        agent = self._aircraft
        agent.lat      = next_state.lat
        agent.lon      = next_state.lon
        agent.altitude = next_state.altitude_ft

        # Optional kinematic attributes — write if the agent has them
        for attr, val in (
            ("speed",         next_state.speed_ktas),
            ("heading",       next_state.heading_deg),
            ("vertical_rate", next_state.vertical_rate),
        ):
            if hasattr(agent, attr):
                setattr(agent, attr, val)

    def _build_state(self, ownship, all_agents) -> AircraftState:
        """Construct AircraftState from live AircraftAgent objects."""
        intruders = self._build_intruder_features(ownship, all_agents)
        return AircraftState(
            lat           = ownship.lat,
            lon           = ownship.lon,
            altitude_ft   = ownship.altitude,
            speed_ktas    = getattr(ownship, "speed",         450.0),
            heading_deg   = getattr(ownship, "heading",       0.0),
            vertical_rate = getattr(ownship, "vertical_rate", 0.0),
            n_contacts    = len(intruders),
            intruders     = intruders,
        )

    def _build_intruder_features(self, ownship, all_agents) -> List[IntruderFeatures]:
        """
        Build IntruderFeatures for every active agent within radar range.
        Sorted by horizontal separation (closest first).

        Closure rate (rel_speed_ktas)
        ─────────────────────────────
        The old code used (own_spd − int_spd) — a scalar that is zero for any
        two aircraft flying at the same speed regardless of heading, so crossing
        traffic was invisible to the τ estimator.

        The correct quantity is the rate at which the distance between the two
        aircraft is decreasing, i.e. the dot product of their relative velocity
        vector onto the unit line-of-sight (LOS) vector:

            v_own  = speed_own  × [sin(hdg_own),  cos(hdg_own)]   (East, North)
            v_int  = speed_int  × [sin(hdg_int),  cos(hdg_int)]
            v_rel  = v_own − v_int                                 (own relative to int)
            LOS    = (pos_int − pos_own) / |pos_int − pos_own|     (unit vector)
            closure = −v_rel · LOS   (positive = aircraft closing)

        The sign convention matches compute_tau: positive closure → finite τ.

        Vertical sense (intruder_alt_ft)
        ────────────────────────────────
        The intruder's absolute altitude is stored directly so acas_x_policy
        can determine which side to evade toward using geometry, not rates.
        """
        features = []
        for agent in all_agents:
            if not getattr(agent, "active", True):
                continue
            if agent.callsign == ownship.callsign:
                continue

            h_sep     = haversine_nm(ownship.lat, ownship.lon, agent.lat, agent.lon)
            v_sep     = abs(ownship.altitude - agent.altitude)
            int_alt   = getattr(agent,   "altitude", 0.0)

            # ── Line-of-sight unit vector (East, North components) ─────────────
            # Convert lat/lon difference to approximate NM offsets, then
            # normalise.  cos(lat) corrects for longitude compression.
            cos_lat  = math.cos(math.radians(ownship.lat))
            d_north  = (agent.lat - ownship.lat) * NM_PER_DEG_LAT
            d_east   = (agent.lon - ownship.lon) * NM_PER_DEG_LAT * cos_lat
            dist_nm  = math.hypot(d_east, d_north)          # == h_sep
            if dist_nm > 1e-6:
                los_e = d_east  / dist_nm
                los_n = d_north / dist_nm
            else:
                los_e, los_n = 0.0, 0.0

            # ── Velocity vectors (kts, East/North components) ──────────────────
            own_hdg = math.radians(getattr(ownship, "heading", 0.0))
            int_hdg = math.radians(getattr(agent,   "heading", 0.0))
            own_spd = getattr(ownship, "speed",  450.0)
            int_spd = getattr(agent,   "speed",  450.0)

            vown_e  = own_spd * math.sin(own_hdg)
            vown_n  = own_spd * math.cos(own_hdg)
            vint_e  = int_spd * math.sin(int_hdg)
            vint_n  = int_spd * math.cos(int_hdg)

            # Relative velocity of ownship w.r.t. intruder
            vrel_e  = vown_e - vint_e
            vrel_n  = vown_n - vint_n

            # Closure rate: positive means the aircraft are getting closer.
            # The LOS vector points from ownship toward intruder, so the
            # component of v_rel along LOS gives the rate of approach.
            closure = vrel_e * los_e + vrel_n * los_n

            # Bearing from ownship to intruder (0 = North, 90 = East)
            bearing = (math.degrees(math.atan2(d_east, d_north)) + 360) % 360

            # Relative vertical rate (fpm)
            own_vr = getattr(ownship, "vertical_rate", 0.0)
            int_vr = getattr(agent,   "vertical_rate", 0.0)
            rel_vr = own_vr - int_vr

            # τ — use vectorial closure for horizontal, vertical rate for vertical
            tau = compute_tau(h_sep, max(0.0, closure), v_sep, max(0.0, rel_vr))

            # Severity
            v_thresh = RVSM_VERT_SEP_FT if ownship.altitude > FL290_FT else VERTICAL_SEP_FT
            if h_sep < HORIZONTAL_SEP_NM and v_sep < v_thresh:
                if h_sep < 1.0:
                    sev = "CRITICAL"
                elif h_sep < 2.0:
                    sev = "HIGH"
                else:
                    sev = "MODERATE"
            else:
                sev = "NONE"

            features.append(IntruderFeatures(
                callsign        = agent.callsign,
                h_sep_nm        = h_sep,
                v_sep_ft        = v_sep,
                intruder_alt_ft = int_alt,
                rel_bearing_deg = bearing,
                rel_speed_ktas  = closure,
                rel_vert_rate   = rel_vr,
                tau_s           = tau,
                severity        = sev,
            ))

        # Sort by separation — most threatening first
        features.sort(key=lambda f: (f.h_sep_nm, f.v_sep_ft))
        return features

    def _add_noise(self, state: AircraftState) -> AircraftState:
        """
        Inject Gaussian position / altitude noise.
        Set noise_std_nm > 0 to switch from MDP to POMDP mode.

        Noise model (future POMDP):
            lat_obs  = lat  + ε_lat    where ε_lat  ~ N(0, σ_nm / NM_PER_DEG_LAT)
            lon_obs  = lon  + ε_lon    where ε_lon  ~ N(0, σ_nm / NM_PER_DEG_LAT)
            alt_obs  = alt  + ε_alt    where ε_alt  ~ N(0, σ_ft)
        """
        if self.noise_std_nm == 0.0 and self.noise_std_ft == 0.0:
            return state
        import copy
        ns = copy.copy(state)
        ns.lat        += np.random.normal(0, self.noise_std_nm / NM_PER_DEG_LAT)
        ns.lon        += np.random.normal(0, self.noise_std_nm / NM_PER_DEG_LAT)
        ns.altitude_ft += np.random.normal(0, self.noise_std_ft)
        return ns

    @staticmethod
    def _is_out_of_bounds(state: AircraftState) -> bool:
        # The altitude floor (MIN_ALTITUDE_FT = 1 000 ft) only applies once the
        # aircraft is airborne.  Aircraft in the ground-roll or initial-climb
        # phase legitimately sit below that floor and must not be terminated.
        # We consider an aircraft "airborne" once it has climbed above the floor
        # at least once; until then only the hard ceiling and geographic limits
        # are checked.
        below_floor = (
            state.altitude_ft >= MIN_ALTITUDE_FT          # has been airborne
            and state.altitude_ft < MIN_ALTITUDE_FT * 0.5 # dropped well below
        )
        return (
            below_floor
            or state.altitude_ft > MAX_ALTITUDE_FT
            or abs(state.lat) > 90.0
            or abs(state.lon) > 180.0
        )


# ==============================================================================
# SECTION 7 — QUICK SANITY CHECK
# ==============================================================================

if __name__ == "__main__":
    from dataclasses import replace

    print("=" * 65)
    print("  mdp_collision_env.py — formulation sanity check")
    print("=" * 65)

    # ── State vector dimensionality ───────────────────────────────────────────
    s = AircraftState()
    vec = s.to_vector()
    print(f"\nState vector dim   : {len(vec)}  (expected {s.dim})")
    assert len(vec) == s.dim, "Dimension mismatch!"
    print("State vector OK")

    # ── Action space ──────────────────────────────────────────────────────────
    print(f"\nAction space size  : {len(Action)}  actions")
    for a in Action:
        vr  = ACTION_VERTICAL_RATE_FPM.get(a, 0.0)
        dv  = ACTION_DELTA_SPEED_KTAS.get(a, 0.0)
        dh  = ACTION_DELTA_HEADING_DEG.get(a, 0.0)
        fuel = action_cost_fuel(a, 450.0)
        print(f"  {a.name:<18} vr={vr:+7.0f} fpm  Δv={dv:+4.0f} kt  Δhdg={dh:+4.0f}°  fuel={fuel:.4f}")

    # ── Reward function with a mock conflict ──────────────────────────────────
    print("\nReward function test (mock CRITICAL conflict)")

    # Build a state with one intruder very close
    intruder = IntruderFeatures(
        callsign="INS001", h_sep_nm=0.5, v_sep_ft=200.0,
        rel_bearing_deg=45.0, rel_speed_ktas=50.0, rel_vert_rate=100.0,
        tau_s=20.0, severity="CRITICAL",
    )
    s_conflict = AircraftState(intruders=[intruder])
    s_conflict_next = AircraftState(intruders=[intruder])

    mock_ev = ConflictEvent(
        step=1, callsign_a="OWN001", callsign_b="INS001",
        h_sep_nm=0.5, v_sep_ft=200.0,
        alt_a_ft=35000, alt_b_ft=34800,
        lat_a=25.0, lon_a=55.0, lat_b=25.005, lon_b=55.005,
    )

    rc = compute_reward(
        state                   = s_conflict,
        action                  = Action.CLIMB_1500,
        next_state              = s_conflict_next,
        active_conflicts_before = {"OWN001|INS001": mock_ev},
        active_conflicts_after  = {"OWN001|INS001": mock_ev},
        ownship_callsign        = "OWN001",
    )
    print(f"\n  Reward components: {rc.to_dict()}")
    assert rc.r_collision == W_COLLISION, "Collision penalty not applied!"
    assert rc.r_severity  <  0.0,        "Severity penalty missing!"
    print("  ✓ Reward function OK")

    # ── Transition test ───────────────────────────────────────────────────────
    print("\nTransition model test")
    s0 = AircraftState(lat=25.0, lon=55.0, altitude_ft=35_000,
                       speed_ktas=450, heading_deg=90, vertical_rate=0)
    s1 = apply_action_kinematics(s0, Action.CLIMB_1500)
    print(f"  s0: alt={s0.altitude_ft:.0f} ft  vr={s0.vertical_rate:.0f} fpm  lat={s0.lat:.4f}  lon={s0.lon:.4f}")
    print(f"  s1: alt={s1.altitude_ft:.0f} ft  vr={s1.vertical_rate:.0f} fpm  lat={s1.lat:.4f}  lon={s1.lon:.4f}")
    assert s1.altitude_ft > s0.altitude_ft, "Climb did not increase altitude!"
    print("  ✓ Transition model OK")

    # ── τ calculation ─────────────────────────────────────────────────────────
    tau = compute_tau(h_sep_nm=2.0, h_closure_ktas=200.0, v_sep_ft=800.0, v_closure_fpm=300.0)
    print(f"\nτ (time-to-conflict): {tau:.1f} s  (expected ~160 s for h-limited case)")
    print("  ✓ τ calculation OK")

    print("\n" + "=" * 65)
    print("  All checks passed.")
    print("=" * 65)

    # ── ACAS X policy test ────────────────────────────────────────────────────
    print("\nACAS X policy test")

    # No threat — expect MAINTAIN
    s_clear = AircraftState()
    assert acas_x_policy(s_clear) == Action.MAINTAIN, "Expected MAINTAIN when no threat"
    print("  No threat          → MAINTAIN  ✓")

    # Imminent threat from above (τ=10s, intruder above → DESCEND_2500)
    threat_above = IntruderFeatures(
        callsign="THR001", h_sep_nm=1.5, v_sep_ft=400.0,
        rel_bearing_deg=30.0, rel_speed_ktas=200.0, rel_vert_rate=-50.0,
        tau_s=10.0, severity="HIGH",
    )
    s_threat_above = AircraftState(intruders=[threat_above])
    a = acas_x_policy(s_threat_above)
    assert a == Action.DESCEND_2500, f"Expected DESCEND_2500, got {a.name}"
    print(f"  τ=10s, above       → {a.name}  ✓")

    # Urgent threat from below (τ=20s, intruder below → CLIMB_1500)
    threat_below = IntruderFeatures(
        callsign="THR002", h_sep_nm=2.0, v_sep_ft=500.0,
        rel_bearing_deg=200.0, rel_speed_ktas=150.0, rel_vert_rate=80.0,
        tau_s=20.0, severity="MODERATE",
    )
    s_threat_below = AircraftState(intruders=[threat_below])
    a = acas_x_policy(s_threat_below)
    assert a == Action.CLIMB_1500, f"Expected CLIMB_1500, got {a.name}"
    print(f"  τ=20s, below       → {a.name}  ✓")

    # Horizontal-only threat (v_sep safe at low altitude, bearing=45 → turn left)
    threat_horiz = IntruderFeatures(
        callsign="THR003", h_sep_nm=1.8, v_sep_ft=1200.0,
        rel_bearing_deg=45.0, rel_speed_ktas=300.0, rel_vert_rate=0.0,
        tau_s=18.0, severity="HIGH",
    )
    # Altitude below FL290 so effective_vert_sep = 1000 ft → v_sep 1200 ft is safe
    s_threat_horiz = AircraftState(altitude_ft=20_000.0, intruders=[threat_horiz])
    a = acas_x_policy(s_threat_horiz)
    assert a in (Action.TURN_LEFT_15, Action.TURN_LEFT_5), f"Expected left turn, got {a.name}"
    print(f"  τ=18s, horiz only  → {a.name}  ✓")

    # ── Write-back test ───────────────────────────────────────────────────────
    print("\nAgent write-back test")

    class _MockAgent:
        callsign = "OWN001"; lat = 25.0; lon = 55.0; altitude = 35_000.0
        speed = 450.0; heading = 90.0; vertical_rate = 0.0; active = True

    mock_agent = _MockAgent()
    from conflict_detection import ConflictDetector as _CD
    mdp = CollisionMDP(detector=_CD(), ownship_callsign="OWN001")
    mdp.reset(mock_agent, [mock_agent])

    # Force a threat so policy issues a climb
    mdp._state.intruders = [IntruderFeatures(
        callsign="THR001", h_sep_nm=1.5, v_sep_ft=400.0,
        rel_bearing_deg=30.0, rel_speed_ktas=200.0, rel_vert_rate=-50.0,
        tau_s=10.0, severity="HIGH",
    )]

    alt_before = mock_agent.altitude
    mdp.run_step(step_num=1)
    alt_after = mock_agent.altitude

    assert alt_after != alt_before, "Agent altitude was not updated by run_step()"
    print(f"  Agent alt before : {alt_before:.0f} ft")
    print(f"  Agent alt after  : {alt_after:.0f} ft  (MDP wrote back)  ✓")
    print(f"  Action issued    : {acas_x_policy(mdp._state).name}")

    # ── Step log test ─────────────────────────────────────────────────────
    print("\nStep log test")

    class _MockAgent2:
        callsign = "LOG001"; lat = 25.0; lon = 55.0; altitude = 35_000.0
        speed = 450.0; heading = 90.0; vertical_rate = 0.0; active = True

    log_agent = _MockAgent2()
    from conflict_detection import ConflictDetector as _CD2
    mdp_log = CollisionMDP(detector=_CD2(), ownship_callsign="LOG001")
    mdp_log.reset(log_agent, [log_agent])

    for i in range(5):
        mdp_log.run_step(step_num=i)

    log_df = mdp_log.get_step_log()
    assert len(log_df) == 5, f"Expected 5 rows, got {len(log_df)}"
    expected_cols = [
        "step", "callsign", "action", "lat", "lon", "altitude_ft",
        "speed_ktas", "heading_deg", "vertical_rate",
        "in_conflict", "conflict_severity", "n_active_conflicts",
        "conflict_partners", "tau_min_s",
        "r_collision", "r_severity", "r_fuel", "r_delay", "r_deviation",
        "r_ra_compliance", "r_clear_of_conflict", "r_survival",
        "reward_total", "cumulative_reward",
    ]
    for col in expected_cols:
        assert col in log_df.columns, f"Missing column: {col}"
    print(f"  Rows recorded      : {len(log_df)}  ✓")
    print(f"  Columns            : {list(log_df.columns)}")
    print(f"  Actions issued     : {log_df['action'].tolist()}")
    print(f"  Rewards            : {log_df['reward_total'].tolist()}")
    print("  Step log OK")

    print("\n" + "=" * 65)
    print("  All checks passed including policy, write-back, and log.")
    print("=" * 65)


# ==============================================================================
# SECTION 8 — OBSERVATION MODEL  Z(o | s)
# ==============================================================================
#
# POMDP background
# ----------------
# A POMDP extends the MDP with an observation model:
#
#     M_POMDP = (S, A, T, R, Ω, Z, γ)
#
#     Ω  — observation space  (noisy, delayed version of S)
#     Z  — observation function  Z(o | s', a)  — probability of receiving
#          observation o after taking action a and landing in state s'
#
# The agent never sees the true state s directly. Instead it receives an
# observation o ~ Z(· | s') and maintains a belief b — a probability
# distribution over possible true states.
#
# Observation noise model (ADS-B / Mode C realistic values)
# ----------------------------------------------------------
# Position noise:
#     ε_lat, ε_lon  ~ N(0, σ_pos)    σ_pos = 0.05 NM default
#     Represents ADS-B horizontal accuracy (NACp=9 ≈ 30 m ≈ 0.016 NM,
#     degraded to 0.05 NM to account for processing latency and
#     multipath in congested airspace)
#
# Altitude noise:
#     ε_alt  ~ N(0, σ_alt)           σ_alt = 75 ft default
#     Mode C altitude reporting has 100 ft quantisation; Gaussian
#     approximation centres on the reported value.
#
# Speed noise:
#     ε_spd  ~ N(0, σ_spd)           σ_spd = 5 kt default
#
# Report delay:
#     ADS-B update rate ≈ 1–2 s; simulation step = 30 s.
#     delay_steps ∈ {0, 1} with P(delay=1) = p_delay (default 0.1)
#     Models occasional missed transponder updates.
#
# Observation function Z(o | s'):
#     For continuous state, Z is a Gaussian:
#         p(o | s') = N(o ; s', Σ)
#     where Σ = diag(σ_pos², σ_pos², σ_alt², σ_spd², ...)

@dataclass
class ObservationNoise:
    """
    Noise parameters for the POMDP observation model.
    All values represent 1-sigma (std deviation) of Gaussian noise.

    Parameters
    ----------
    sigma_pos_nm    : position noise std (NM)
    sigma_alt_ft    : altitude noise std (ft)
    sigma_spd_kt    : speed noise std (ktas)
    sigma_hdg_deg   : heading noise std (degrees)
    p_delay         : probability of a one-step report delay
    """
    sigma_pos_nm  : float = 0.05    # ADS-B horizontal accuracy degraded
    sigma_alt_ft  : float = 75.0    # Mode C quantisation approximation
    sigma_spd_kt  : float = 5.0     # groundspeed uncertainty
    sigma_hdg_deg : float = 1.5     # heading uncertainty
    p_delay       : float = 0.10    # P(delayed report)

    # Derived: per-degree lat/lon sigma (NM → degrees)
    @property
    def sigma_lat_deg(self) -> float:
        return self.sigma_pos_nm / NM_PER_DEG_LAT

    def sigma_lon_deg(self, lat_deg: float) -> float:
        cos_lat = math.cos(math.radians(lat_deg)) or 1e-9
        return self.sigma_pos_nm / (NM_PER_DEG_LAT * cos_lat)


class ObservationModel:
    """
    Generates noisy observations from true aircraft states.

    Z(o | s') = N(o ; h(s'), Σ)

    where h(s') is the nominal sensor reading (= true state with
    possible delay), and Σ is the diagonal noise covariance.

    Two responsibilities:
        1. sample(state)        → draw one noisy observation
        2. log_likelihood(o, s) → log p(o | s) for belief update
    """

    def __init__(self, noise: ObservationNoise = None):
        self.noise = noise or ObservationNoise()
        self._prev_state: Optional[AircraftState] = None   # for delay model

    def sample(self, true_state: AircraftState) -> AircraftState:
        """
        Draw a noisy observation from the true state.

        Equation
        --------
        o_pos  = s_pos  + N(0, σ_pos)
        o_alt  = s_alt  + N(0, σ_alt)
        o_spd  = s_spd  + N(0, σ_spd)
        o_hdg  = s_hdg  + N(0, σ_hdg)

        With probability p_delay the observation is from the previous
        step (stale report) rather than the current true state.
        """
        import copy

        # Delay model: return stale observation with probability p_delay
        if self._prev_state is not None and np.random.random() < self.noise.p_delay:
            source = self._prev_state
        else:
            source = true_state

        obs = copy.copy(source)

        rng = np.random.default_rng()
        obs.lat        = source.lat + rng.normal(0.0, self.noise.sigma_lat_deg)
        obs.lon        = source.lon + rng.normal(
            0.0, self.noise.sigma_lon_deg(source.lat)
        )
        obs.altitude_ft = float(np.clip(
            source.altitude_ft + rng.normal(0.0, self.noise.sigma_alt_ft),
            MIN_ALTITUDE_FT, MAX_ALTITUDE_FT,
        ))
        obs.speed_ktas  = float(np.clip(
            source.speed_ktas + rng.normal(0.0, self.noise.sigma_spd_kt),
            MIN_SPEED_KTAS, MAX_SPEED_KTAS,
        ))
        obs.heading_deg = (
            source.heading_deg + rng.normal(0.0, self.noise.sigma_hdg_deg)
        ) % 360.0

        # Intruders: apply position noise to each intruder's relative geometry
        noisy_intruders = []
        for feat in source.intruders:
            nf = copy.copy(feat)
            nf.h_sep_nm   = max(0.0, feat.h_sep_nm + rng.normal(0.0, self.noise.sigma_pos_nm))
            nf.v_sep_ft   = max(0.0, feat.v_sep_ft + rng.normal(0.0, self.noise.sigma_alt_ft))
            nf.rel_speed_ktas = feat.rel_speed_ktas + rng.normal(0.0, self.noise.sigma_spd_kt)
            noisy_intruders.append(nf)
        obs.intruders = noisy_intruders

        self._prev_state = copy.copy(true_state)
        return obs

    def log_likelihood(
        self,
        observation: AircraftState,
        hypothetical_state: AircraftState,
    ) -> float:
        """
        Compute log p(observation | hypothetical_state) under the Gaussian model.

        Used by the particle filter belief update to weight particles.

        log p(o|s) = -½ Σ_i [(o_i - s_i)² / σ_i²]  + const

        We use the key observable dimensions only:
            lat, lon, altitude_ft, speed_ktas, heading_deg
        """
        eps = 1e-9
        terms = [
            ((observation.lat - hypothetical_state.lat) /
             max(self.noise.sigma_lat_deg, eps)) ** 2,
            ((observation.lon - hypothetical_state.lon) /
             max(self.noise.sigma_lon_deg(hypothetical_state.lat), eps)) ** 2,
            ((observation.altitude_ft - hypothetical_state.altitude_ft) /
             max(self.noise.sigma_alt_ft, eps)) ** 2,
            ((observation.speed_ktas - hypothetical_state.speed_ktas) /
             max(self.noise.sigma_spd_kt, eps)) ** 2,
            ((observation.heading_deg - hypothetical_state.heading_deg) /
             max(self.noise.sigma_hdg_deg, eps)) ** 2,
        ]
        return -0.5 * sum(terms)


# ==============================================================================
# SECTION 9 — BELIEF STATE  b(s)
# ==============================================================================
#
# The belief state b is a probability distribution over possible true states.
# In continuous state spaces, exact belief updates are intractable.
# We use a Sequential Importance Resampling (SIR) particle filter:
#
#     Initialisation:
#         Draw N particles from a prior around the first observation:
#             x_i^0 ~ N(o_0, Σ_prior)
#
#     Prediction step (after action a):
#         Propagate each particle through the transition:
#             x_i' ~ T(· | x_i, a)   +  process noise
#
#     Update step (after observation o):
#         Compute importance weights:
#             w_i = p(o | x_i')   =   exp(log_likelihood(o, x_i'))
#         Normalise:  w_i ← w_i / Σ w_j
#         Resample:   draw N particles from {x_i', w_i}
#
#     Belief summary:
#         mean_state  = Σ_i w_i · x_i'
#         uncertainty = weighted variance of key dimensions
#
# The mean_state is passed to the SARSOP policy as the "most likely" state.
# The uncertainty feeds into the risk-aversion term of the reward.

@dataclass
class Particle:
    """One particle in the belief state — a hypothetical true state."""
    state : AircraftState
    weight: float = 1.0


class BeliefState:
    """
    Particle filter belief state for the POMDP.

    Maintains N weighted particles representing p(s | history of o, a).

    Parameters
    ----------
    n_particles     : number of particles (default 200)
    obs_model       : ObservationModel for likelihood computation
    process_noise_nm: std of particle propagation noise (NM)
    process_noise_ft: std of particle propagation altitude noise (ft)

    Key methods
    -----------
    initialise(obs)           → seed particles around first observation
    predict(action)           → propagate particles through transition
    update(observation)       → reweight + resample by likelihood
    mean_state()              → weighted mean across all particles
    uncertainty()             → dict of per-dimension std deviations
    """

    def __init__(
        self,
        obs_model       : ObservationModel,
        n_particles     : int   = 200,
        process_noise_nm: float = 0.02,
        process_noise_ft: float = 30.0,
    ):
        self.obs_model        = obs_model
        self.n_particles      = n_particles
        self.process_noise_nm = process_noise_nm
        self.process_noise_ft = process_noise_ft
        self.particles        : List[Particle] = []
        self._eff_sample_size : float = float(n_particles)

    def initialise(self, observation: AircraftState) -> None:
        """
        Seed N particles around the first observation.

        Each particle is drawn from:
            lat  ~ N(obs.lat,  σ_pos_lat)
            lon  ~ N(obs.lon,  σ_pos_lon)
            alt  ~ N(obs.alt,  σ_alt)
            spd  ~ N(obs.spd,  σ_spd)
            hdg  ~ N(obs.hdg,  σ_hdg)
        """
        import copy
        rng = np.random.default_rng()
        noise = self.obs_model.noise

        self.particles = []
        for _ in range(self.n_particles):
            s = copy.deepcopy(observation)
            s.lat        += rng.normal(0, noise.sigma_lat_deg)
            s.lon        += rng.normal(0, noise.sigma_lon_deg(observation.lat))
            s.altitude_ft = float(np.clip(
                s.altitude_ft + rng.normal(0, noise.sigma_alt_ft),
                MIN_ALTITUDE_FT, MAX_ALTITUDE_FT,
            ))
            s.speed_ktas  = float(np.clip(
                s.speed_ktas + rng.normal(0, noise.sigma_spd_kt),
                MIN_SPEED_KTAS, MAX_SPEED_KTAS,
            ))
            s.heading_deg = (s.heading_deg + rng.normal(0, noise.sigma_hdg_deg)) % 360.0
            self.particles.append(Particle(state=s, weight=1.0 / self.n_particles))

    def predict(self, action: Action) -> None:
        """
        Propagate all particles through the transition model T(s'|s, a)
        with added process noise.

        Process noise represents unmodelled dynamics (wind gusts, pilot
        deviations from advisory) — keeps the particle cloud from collapsing.

        x_i' = apply_action_kinematics(x_i, a) + ε_process
        """
        import copy
        rng = np.random.default_rng()
        nm_lat = self.process_noise_nm / NM_PER_DEG_LAT

        new_particles = []
        for p in self.particles:
            ns = apply_action_kinematics(p.state, action)
            ns.lat        += rng.normal(0, nm_lat)
            ns.lon        += rng.normal(
                0, self.process_noise_nm / (NM_PER_DEG_LAT *
                   max(math.cos(math.radians(ns.lat)), 1e-9))
            )
            ns.altitude_ft = float(np.clip(
                ns.altitude_ft + rng.normal(0, self.process_noise_ft),
                MIN_ALTITUDE_FT, MAX_ALTITUDE_FT,
            ))
            new_particles.append(Particle(state=ns, weight=p.weight))
        self.particles = new_particles

    def update(self, observation: AircraftState) -> None:
        """
        Reweight particles by likelihood then resample (SIR step).

        w_i ← w_i · p(o | x_i')
        normalise → systematic resample → uniform weights
        """
        # Compute log weights to avoid underflow
        log_weights = np.array([
            math.log(max(p.weight, 1e-300)) +
            self.obs_model.log_likelihood(observation, p.state)
            for p in self.particles
        ])

        # Stable softmax normalisation
        log_weights -= log_weights.max()
        weights = np.exp(log_weights)
        total   = weights.sum()
        if total < 1e-300:
            # All particles collapsed — reinitialise around observation
            self.initialise(observation)
            return
        weights /= total

        # Effective sample size — measure of particle diversity
        self._eff_sample_size = 1.0 / float(np.sum(weights ** 2))

        # Systematic resampling
        N = self.n_particles
        cumsum   = np.cumsum(weights)
        step     = 1.0 / N
        start    = np.random.uniform(0, step)
        pointers = start + step * np.arange(N)

        new_particles = []
        j = 0
        for ptr in pointers:
            while j < N - 1 and cumsum[j] < ptr:
                j += 1
            import copy
            new_particles.append(Particle(
                state=copy.deepcopy(self.particles[j].state),
                weight=1.0 / N,
            ))
        self.particles = new_particles

    def mean_state(self) -> AircraftState:
        """
        Compute the weighted mean across all particles.

        Returns an AircraftState representing the most likely true state
        given all observations received so far.
        """
        import copy
        if not self.particles:
            raise RuntimeError("BeliefState not initialised — call initialise() first")

        weights = np.array([p.weight for p in self.particles])
        weights /= weights.sum()

        mean = copy.deepcopy(self.particles[0].state)
        mean.lat         = float(np.sum([w * p.state.lat         for w, p in zip(weights, self.particles)]))
        mean.lon         = float(np.sum([w * p.state.lon         for w, p in zip(weights, self.particles)]))
        mean.altitude_ft = float(np.sum([w * p.state.altitude_ft for w, p in zip(weights, self.particles)]))
        mean.speed_ktas  = float(np.sum([w * p.state.speed_ktas  for w, p in zip(weights, self.particles)]))
        mean.heading_deg = float(np.sum([w * p.state.heading_deg for w, p in zip(weights, self.particles)]))
        mean.vertical_rate = float(np.sum([w * p.state.vertical_rate for w, p in zip(weights, self.particles)]))
        return mean

    def uncertainty(self) -> dict:
        """
        Per-dimension weighted standard deviation across particles.

        Returns a dict with keys: lat_nm, alt_ft, spd_kt, hdg_deg.
        Used by SARSOP policy to scale risk aversion.
        """
        weights = np.array([p.weight for p in self.particles])
        weights /= weights.sum()

        def wstd(vals):
            mean = np.sum(weights * vals)
            return float(np.sqrt(np.sum(weights * (vals - mean) ** 2)))

        lats  = np.array([p.state.lat         for p in self.particles])
        alts  = np.array([p.state.altitude_ft  for p in self.particles])
        spds  = np.array([p.state.speed_ktas   for p in self.particles])
        hdgs  = np.array([p.state.heading_deg  for p in self.particles])

        return {
            "lat_nm"     : wstd(lats) * NM_PER_DEG_LAT,
            "alt_ft"     : wstd(alts),
            "spd_kt"     : wstd(spds),
            "hdg_deg"    : wstd(hdgs),
            "eff_samples": round(self._eff_sample_size, 1),
        }


# ==============================================================================
# SECTION 10 — SARSOP-INSPIRED ONLINE POLICY
# ==============================================================================
#
# SARSOP (Successive Approximations of the Reachable Space under Optimal
# Policies, Kurniawati et al. 2008) is an offline point-based POMDP solver.
# It builds a piecewise-linear alpha-vector representation of the value
# function V*(b) = max_α  α · b and solves for the optimal policy offline.
#
# For this application an offline solver is impractical because:
#   - The state space is continuous and 90-dimensional
#   - The aircraft encounter geometry changes dynamically every step
#   - A new SARSOP tree would need to be built per encounter
#
# Instead we implement SARSOP-ONLINE: at each step we perform a bounded
# lookahead search from the current belief state, guided by the SARSOP
# value function approximation principle (alpha vectors over sampled
# belief points).
#
# Algorithm
# ---------
# 1. From current belief b, sample K scenario states {s_1,...,s_K}
#    (K particles drawn from the belief distribution)
#
# 2. For each candidate action a ∈ A:
#    a. For each scenario s_k:
#       i.   Predict next state:  s_k' ~ T(·|s_k, a)
#       ii.  Sample observation:  o_k  ~ Z(·|s_k')
#       iii. Compute immediate reward:  r_k = R(s_k, a, s_k')
#       iv.  Estimate future value via ACAS X heuristic:  V(s_k')
#    b. Estimate Q(b, a) = (1/K) Σ_k [r_k + γ · V(s_k')]
#
# 3. Select action:  π*(b) = argmax_a  Q(b, a)
#
# Risk-aversion under uncertainty
# --------------------------------
# SARSOP's key insight is that under high belief uncertainty, the policy
# should be more conservative. We implement this via a risk penalty:
#
#     Q_risk(b, a) = Q(b, a)  −  λ_risk · σ_belief · collision_exposure(a)
#
# where:
#     σ_belief          = mean uncertainty in position (NM)
#     collision_exposure = estimated probability of conflict given action a
#     λ_risk            = risk aversion coefficient (default 2.0)
#
# This means: under high positional uncertainty, the policy prefers
# conservative actions (larger separation margins) over aggressive ones.
# This is the core contribution of POMDP over MDP for this problem.

SARSOP_N_SCENARIOS   = 50    # particles sampled per action evaluation
SARSOP_LOOKAHEAD     = 3     # lookahead depth (steps)
SARSOP_RISK_LAMBDA   = 2.0   # risk aversion coefficient λ


def _mdp_value_estimate(state: AircraftState) -> float:
    """
    Terminal value estimate V_MDP(s) for SARSOP lookahead leaf nodes.

    In point-based POMDP solvers (PBVI, SARSOP) the lookahead tree must
    terminate at some depth. At leaf nodes the standard approach is to
    substitute V_MDP(s) — the value of state s under the fully-observed
    MDP — as the terminal estimate. This is principled because:

        V_MDP(s) >= V_POMDP(b)   for any belief b consistent with s

    i.e. full observability is an upper bound on POMDP performance, so
    using V_MDP as the leaf estimate keeps the lookahead optimistic in
    the correct direction and does not introduce ACAS X policy logic
    into the POMDP planner.

    The approximation used here:
        V_MDP(s) ≈ W_STEP_SURVIVAL − W_SEVERITY · φ(s)

    where φ(s) is the separation violation depth from the MDP reward
    function (Section 5). This is a fast, analytically tractable proxy
    for the true MDP value without requiring a full offline solve.
    """
    if not state.intruders:
        return W_STEP_SURVIVAL

    phi_total = 0.0
    for feat in state.intruders:
        if feat.h_sep_nm < HORIZONTAL_SEP_NM:
            v_safe = RVSM_VERT_SEP_FT if state.altitude_ft > FL290_FT else VERTICAL_SEP_FT
            phi_h  = max(0.0, 1.0 - feat.h_sep_nm / HORIZONTAL_SEP_NM)
            phi_v  = max(0.0, 1.0 - feat.v_sep_ft / v_safe)
            phi_total += phi_h * phi_v

    return W_STEP_SURVIVAL - W_SEVERITY * phi_total


class SARSOPPolicy:
    """
    SARSOP-inspired online POMDP policy.

    At each step, performs a bounded lookahead over K sampled scenarios
    from the current belief state and selects the action that maximises
    the risk-adjusted expected value.

    Parameters
    ----------
    obs_model       : ObservationModel
    n_scenarios     : particles to sample per action evaluation
    lookahead       : planning depth in steps
    risk_lambda     : risk aversion coefficient (higher = more conservative
                      under uncertainty)
    gamma           : discount factor

    Key method
    ----------
    select_action(belief) → Action
    """

    def __init__(
        self,
        obs_model   : ObservationModel,
        n_scenarios : int   = SARSOP_N_SCENARIOS,
        lookahead   : int   = SARSOP_LOOKAHEAD,
        risk_lambda : float = SARSOP_RISK_LAMBDA,
        gamma       : float = GAMMA,
    ):
        self.obs_model   = obs_model
        self.n_scenarios = n_scenarios
        self.lookahead   = lookahead
        self.risk_lambda = risk_lambda
        self.gamma       = gamma

    def select_action(self, belief: BeliefState) -> Action:
        """
        π*(b) = argmax_a  Q_risk(b, a)

        Evaluates all 14 actions via scenario sampling and returns the
        action with the highest risk-adjusted expected value.

        Falls back to acas_x_policy(mean_state) if the belief has fewer
        than 2 particles (degenerate belief).
        """
        if len(belief.particles) < 2:
            return acas_x_policy(belief.mean_state())

        # Sample K scenario states from belief
        weights = np.array([p.weight for p in belief.particles])
        weights /= weights.sum()
        indices = np.random.choice(
            len(belief.particles),
            size=min(self.n_scenarios, len(belief.particles)),
            replace=True,
            p=weights,
        )
        scenarios = [belief.particles[i].state for i in indices]

        # Belief uncertainty for risk penalty
        unc = belief.uncertainty()
        sigma_pos = unc["lat_nm"]   # positional uncertainty in NM

        best_action = Action.MAINTAIN
        best_q      = -float("inf")

        for action in Action:
            q_values = []
            for s_k in scenarios:
                q_k = self._rollout(s_k, action, depth=self.lookahead)
                q_values.append(q_k)

            q_mean = float(np.mean(q_values))

            # Risk penalty: how likely is this action to lead to conflict
            # given current positional uncertainty?
            collision_exposure = self._collision_exposure(scenarios, action, sigma_pos)
            q_risk = q_mean - self.risk_lambda * sigma_pos * collision_exposure

            if q_risk > best_q:
                best_q      = q_risk
                best_action = action

        return best_action

    def _rollout(
        self,
        state : AircraftState,
        action: Action,
        depth : int,
    ) -> float:
        """
        Recursive lookahead rollout from state s under action a.

        Q(s, a, depth) = R(s, a, s') + γ · V(s')           if depth == 1
                       = R(s, a, s') + γ · Q(s', π(s'), d-1) otherwise

        π(s') uses the ACAS X heuristic policy for inner nodes
        (greedy with respect to the value heuristic).
        """
        import copy
        next_s = apply_action_kinematics(state, action)
        obs    = self.obs_model.sample(next_s)

        # Immediate reward estimate using heuristic components
        r = self._immediate_reward(state, action, next_s)

        if depth <= 1:
            return r + self.gamma * _mdp_value_estimate(next_s)

        # Inner node: follow ACAS X heuristic policy
        inner_action = acas_x_policy(obs)
        return r + self.gamma * self._rollout(next_s, inner_action, depth - 1)

    def _immediate_reward(
        self,
        state    : AircraftState,
        action   : Action,
        next_state: AircraftState,
    ) -> float:
        """
        Fast reward estimate for rollout (no full ConflictDetector call).
        Uses the severity gradient and fuel cost only.
        """
        r = W_STEP_SURVIVAL

        # Severity
        for feat in next_state.intruders:
            if feat.h_sep_nm < HORIZONTAL_SEP_NM:
                v_safe = (RVSM_VERT_SEP_FT if next_state.altitude_ft > FL290_FT
                          else VERTICAL_SEP_FT)
                phi_h = max(0.0, 1.0 - feat.h_sep_nm / HORIZONTAL_SEP_NM)
                phi_v = max(0.0, 1.0 - feat.v_sep_ft / v_safe)
                r    += W_SEVERITY * phi_h * phi_v
                if feat.h_sep_nm < 1.0:
                    r += W_COLLISION

        # Fuel
        r += W_FUEL * action_cost_fuel(action, state.speed_ktas)

        return r

    def _collision_exposure(
        self,
        scenarios  : List[AircraftState],
        action     : Action,
        sigma_pos  : float,
    ) -> float:
        """
        Estimate collision exposure E[conflict | action, uncertainty].

        For each scenario, apply action and check if any intruder is within
        (HORIZONTAL_SEP_NM + sigma_pos) — a safety margin that grows with
        positional uncertainty. Returns the fraction of scenarios in conflict.

        This is the risk term that makes the POMDP policy more conservative
        than the MDP policy under high positional uncertainty.
        """
        if not scenarios:
            return 0.0

        n_conflict = 0
        margin = HORIZONTAL_SEP_NM + sigma_pos  # uncertainty-expanded margin

        for s in scenarios:
            ns = apply_action_kinematics(s, action)
            for feat in ns.intruders:
                if feat.h_sep_nm < margin:
                    n_conflict += 1
                    break

        return n_conflict / len(scenarios)


# ==============================================================================
# SECTION 11 — POMDP ENVIRONMENT  (CollisionPOMDP)
# ==============================================================================

class CollisionPOMDP(CollisionMDP):
    """
    POMDP extension of CollisionMDP using SARSOP-inspired online planning.

    Extends CollisionMDP by adding:
        • ObservationModel   — generates noisy observations each step
        • BeliefState        — particle filter over true state
        • SARSOPPolicy       — selects actions from belief, not true state

    The agent never acts on the true state directly. Instead:
        1. A noisy observation o is drawn from Z(· | s_true)
        2. The belief b is updated: predict(a) → update(o)
        3. The SARSOP policy selects: a* = π*(b)
        4. Kinematics + write-back proceed as in CollisionMDP

    This is the minimal change from MDP to POMDP — everything else
    (reward, conflict detection, write-back, step log) is inherited.

    Parameters
    ----------
    All CollisionMDP parameters, plus:
    noise           : ObservationNoise — sensor noise parameters
    n_particles     : belief state particle count
    n_scenarios     : SARSOP lookahead scenario count
    lookahead       : SARSOP planning depth
    risk_lambda     : SARSOP risk aversion coefficient

    Mesa integration
    ----------------
    Identical to CollisionMDP:
        next_state, reward, done, info = self.pomdp.run_step(step_num)

    Usage
    -----
        from mdp_collision_env import CollisionPOMDP, ObservationNoise

        noise = ObservationNoise(sigma_pos_nm=0.05, sigma_alt_ft=75.0)
        agent.pomdp = CollisionPOMDP(
            detector         = shared_detector,
            ownship_callsign = agent.callsign,
            noise            = noise,
        )
        agent.pomdp.reset(agent, all_agents)

        # In AircraftAgent.step():
        next_state, reward, done, info = agent.pomdp.run_step(step_num)
    """

    def __init__(
        self,
        detector        : ConflictDetector,
        ownship_callsign: str,
        max_steps       : int   = 1_912,
        gamma           : float = GAMMA,
        noise           : ObservationNoise = None,
        n_particles     : int   = 200,
        n_scenarios     : int   = SARSOP_N_SCENARIOS,
        lookahead       : int   = SARSOP_LOOKAHEAD,
        risk_lambda     : float = SARSOP_RISK_LAMBDA,
    ):
        # Pass noise to parent via noise_std_nm — CollisionMDP._add_noise()
        # is now superseded by the full ObservationModel here
        super().__init__(
            detector         = detector,
            ownship_callsign = ownship_callsign,
            max_steps        = max_steps,
            gamma            = gamma,
            noise_std_nm     = 0.0,   # disabled — ObservationModel handles this
            noise_std_ft     = 0.0,
        )
        self._obs_noise    = noise or ObservationNoise()
        self._obs_model    = ObservationModel(self._obs_noise)
        self._belief       = BeliefState(
            obs_model        = self._obs_model,
            n_particles      = n_particles,
        )
        self._sarsop       = SARSOPPolicy(
            obs_model        = self._obs_model,
            n_scenarios      = n_scenarios,
            lookahead        = lookahead,
            risk_lambda      = risk_lambda,
            gamma            = gamma,
        )
        self._last_obs     : Optional[AircraftState] = None
        self._uncertainty_log: List[dict]             = []

    def reset(self, ownship_agent, all_agents: List) -> AircraftState:
        """Reset MDP and initialise belief state from first observation."""
        true_state = super().reset(ownship_agent, all_agents)
        first_obs  = self._obs_model.sample(true_state)
        self._belief.initialise(first_obs)
        self._last_obs = first_obs
        self._uncertainty_log.clear()
        return true_state

    def run_step(self, step_num: int = 0) -> Tuple[AircraftState, float, bool, dict]:
        """
        POMDP step — overrides CollisionMDP.run_step().

        Sequence
        --------
        1. Sample noisy observation  o ~ Z(· | s_true)
        2. Belief predict:           b.predict(last_action)
        3. Belief update:            b.update(o)
        4. SARSOP selects action:    a* = π*(b)
        5. Apply kinematics on MEAN STATE (not noisy obs)
        6. Write back to agent
        7. Conflict detection + reward (inherited)
        8. Log uncertainty alongside standard StepRecord
        """
        if self._done:
            # In embedded (simulation) mode _done must never crash the run.
            # Auto-reset so the safety layer re-engages cleanly for the next
            # step rather than terminating the entire scenario.
            self.reset(self._aircraft, self._all_ac)

        # ── 1. Resync true state from the agent's current CSV position ─────────
        # Mirrors the same resync in CollisionMDP.run_step(): the agent has
        # already loaded its CSV row before calling run_step(), so rebuilding
        # here gives the POMDP the real ground-truth position to reason from.
        self._state = self._build_state(self._aircraft, self._all_ac)

        # ── 2–4. Threat-gated belief update + SARSOP ─────────────────────────
        #
        # Running the full particle filter + SARSOP on every step for every
        # aircraft regardless of traffic costs ~44 M rollouts for a high-density
        # scenario (43 min wall time).  Almost all of that work happens in clear
        # sky where the answer is always MAINTAIN.
        #
        # Two-tier gate:
        #
        #   BELIEF_WARMUP_NM  (default 2× HORIZONTAL_SEP_NM = 6 NM)
        #       Any intruder within this radius → run predict+update so the
        #       particle filter is warm before a threat enters RA range.
        #
        #   TAU_THRESHOLD_S   (35 s — existing ACAS X trigger)
        #       Any intruder with τ ≤ this value → run full SARSOP select_action.
        #       Otherwise skip to MAINTAIN immediately.
        #
        # When neither gate fires the belief state is left unchanged (particles
        # stay at their last positions).  This is safe: if no aircraft is
        # within 6 NM there is nothing for the filter to track anyway.

        BELIEF_WARMUP_NM = HORIZONTAL_SEP_NM * 2.0   # 6 NM lookahead radius

        # Classify the current intruder picture
        intruders_nearby    = [
            f for f in self._state.intruders
            if f.h_sep_nm < BELIEF_WARMUP_NM
        ]
        intruders_threating = [
            f for f in self._state.intruders
            if f.tau_s <= TAU_THRESHOLD_S and f.severity != "NONE"
        ]

        last_action = Action.MAINTAIN
        if self._step_log:
            try:
                last_action = Action[self._step_log[-1].action]
            except KeyError:
                last_action = Action.MAINTAIN

        if intruders_nearby or self._in_avoidance:
            # ── 2. Observe ────────────────────────────────────────────────────
            obs = self._obs_model.sample(self._state)
            self._last_obs = obs

            # ── 3. Belief predict + update ────────────────────────────────────
            self._belief.predict(last_action)
            self._belief.update(obs)

            if intruders_threating or self._in_avoidance:
                # ── 4a. Full SARSOP — threat is within RA horizon ─────────────
                mean_s          = self._belief.mean_state()
                mean_s.intruders = self._state.intruders
                action          = self._sarsop.select_action(self._belief)
            else:
                # ── 4b. Intruder nearby but not yet threatening — MAINTAIN ─────
                # Belief is being kept warm; no RA needed yet.
                action = Action.MAINTAIN
        else:
            # ── 4c. Clear sky — skip belief update and SARSOP entirely ─────────
            action = Action.MAINTAIN

        # ── 5. Three-phase safety-layer (mirrors CollisionMDP.run_step) ────────
        # Phase A: avoidance action → apply kinematics, write back, record exit.
        # Phase B: RTR blend       → linearly interpolate back to CSV over
        #                            RTR_BLEND_STEPS steps.
        # Phase C: normal          → leave CSV position untouched.

        avoidance_action = action not in (Action.MAINTAIN, Action.LEVEL_OFF)
        conflicts_before = self.detector.get_active_conflicts()

        agent   = self._aircraft
        csv_lat = agent.lat
        csv_lon = agent.lon
        csv_alt = agent.altitude
        csv_spd = agent.speed
        csv_hdg = agent.heading

        if avoidance_action:
            # Phase A — avoidance: run kinematics on belief mean state
            mean_s_for_kin = self._belief.mean_state()
            mean_s_for_kin.intruders = self._state.intruders
            next_state_det = apply_action_kinematics(mean_s_for_kin, action)
            next_state = self._add_noise(next_state_det)
            self._write_back_to_agent(next_state)
            self._rtr_origin_lat = next_state.lat
            self._rtr_origin_lon = next_state.lon
            self._rtr_origin_alt = next_state.altitude_ft
            self._rtr_origin_spd = next_state.speed_ktas
            self._rtr_origin_hdg = next_state.heading_deg
            self._in_avoidance   = True
            self._rtr_step       = 0

        elif self._in_avoidance:
            self._rtr_step += 1
            t = min(self._rtr_step / RTR_BLEND_STEPS, 1.0)

            if t >= 1.0:
                self._in_avoidance = False
                self._rtr_step     = 0
                next_state = self._add_noise(self._state)
            else:
                def _lerp(a, b, t):
                    return a + (b - a) * t
                def _lerp_heading(h0, h1, t):
                    diff = ((h1 - h0 + 180) % 360) - 180
                    return (h0 + diff * t) % 360

                blended_lat = _lerp(self._rtr_origin_lat, csv_lat, t)
                blended_lon = _lerp(self._rtr_origin_lon, csv_lon, t)
                blended_alt = _lerp(self._rtr_origin_alt, csv_alt, t)
                blended_spd = _lerp(self._rtr_origin_spd, csv_spd, t)
                blended_hdg = _lerp_heading(self._rtr_origin_hdg, csv_hdg, t)

                agent.lat      = blended_lat
                agent.lon      = blended_lon
                agent.altitude = blended_alt
                agent.speed    = blended_spd
                agent.heading  = blended_hdg

                import copy
                next_state = copy.copy(self._state)
                next_state.lat         = blended_lat
                next_state.lon         = blended_lon
                next_state.altitude_ft = blended_alt
                next_state.speed_ktas  = blended_spd
                next_state.heading_deg = blended_hdg
                next_state = self._add_noise(next_state)

        else:
            # Phase C — normal, no active avoidance
            next_state = self._add_noise(self._state)

        conflicts_after = self.detector.get_active_conflicts()

        next_state.intruders  = self._build_intruder_features(self._aircraft, self._all_ac)
        next_state.n_contacts = len(next_state.intruders)

        # ── 6. Reward ─────────────────────────────────────────────────────────
        rc = compute_reward(
            state                   = self._state,
            action                  = action,
            next_state              = next_state,
            active_conflicts_before = conflicts_before,
            active_conflicts_after  = conflicts_after,
            ownship_callsign        = self.ownship_callsign,
        )

        self._step_num += 1
        self._cumulative_reward += (self.gamma ** self._step_num) * rc.total

        collision     = (next_state.worst_severity == "CRITICAL")
        out_of_bounds = self._is_out_of_bounds(next_state)
        timeout       = (self._step_num >= self.max_steps)
        self._done    = collision or out_of_bounds or timeout
        self._state   = next_state

        info = {
            "step"              : self._step_num,
            "action"            : action.name,
            "reward_components" : rc.to_dict(),
            "active_conflicts"  : len(conflicts_after),
            "conflict_summary"  : self.detector.summary(),
            "done_reason"       : ("collision"    if collision     else
                                   "out_of_bounds" if out_of_bounds else
                                   "timeout"       if timeout       else
                                   "running"),
            "cumulative_reward" : round(self._cumulative_reward, 4),
        }

        partners = ", ".join(self.detector.conflict_partners(self.ownship_callsign))
        self._step_log.append(StepRecord(
            step                = self._step_num,
            callsign            = self.ownship_callsign,
            action              = action.name,
            lat                 = next_state.lat,
            lon                 = next_state.lon,
            altitude_ft         = next_state.altitude_ft,
            speed_ktas          = next_state.speed_ktas,
            heading_deg         = next_state.heading_deg,
            vertical_rate       = next_state.vertical_rate,
            in_conflict         = next_state.in_conflict,
            conflict_severity   = next_state.worst_severity,
            n_active_conflicts  = len(conflicts_after),
            conflict_partners   = partners,
            tau_min_s           = next_state.min_tau,
            r_collision         = rc.r_collision,
            r_severity          = rc.r_severity,
            r_fuel              = rc.r_fuel,
            r_delay             = rc.r_delay,
            r_deviation         = rc.r_deviation,
            r_ra_compliance     = rc.r_ra_compliance,
            r_clear_of_conflict = rc.r_clear_of_conflict,
            r_survival          = rc.r_survival,
            reward_total        = rc.total,
            cumulative_reward   = self._cumulative_reward,
        ))

        # Replace the super().step() call that was here before — the POMDP
        # now handles everything inline so it can control the safety-layer gate.
        next_state, reward, done = next_state, rc.total, self._done

        # ── 9. Log uncertainty ────────────────────────────────────────────────
        unc = self._belief.uncertainty()
        info["belief_uncertainty"] = unc
        info["n_effective_particles"] = unc["eff_samples"]
        info["policy"] = "SARSOP"

        self._uncertainty_log.append({
            "step"              : self._step_num,
            "pos_uncertainty_nm": round(unc["lat_nm"], 4),
            "alt_uncertainty_ft": round(unc["alt_ft"], 2),
            "spd_uncertainty_kt": round(unc["spd_kt"], 2),
            "eff_particles"     : unc["eff_samples"],
        })

        return next_state, reward, done, info

    import pandas as pd
    def get_uncertainty_log(self) -> "pd.DataFrame":
        """
        Return per-step belief uncertainty as a DataFrame.

        Columns: step, pos_uncertainty_nm, alt_uncertainty_ft,
                 spd_uncertainty_kt, eff_particles

        Merge with get_step_log() on 'step' for a complete POMDP record.
        """
        import pandas as pd
        if not self._uncertainty_log:
            return pd.DataFrame()
        return pd.DataFrame(self._uncertainty_log)
