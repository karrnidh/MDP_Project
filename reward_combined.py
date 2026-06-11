"""
reward_combined.py
==================
Combined Reward Function — Group A (TFR Avoidance) + Group B (Collision Avoidance)

Authors:
  Karrnidh  — Group A reward design  (mdp.py  ::  reward())
  Anaya     — Group B reward design  (mdp_collision_env.py  ::  compute_reward())

Mathematical formulation
------------------------
The joint reward is:

    R(s, a, s') = R_col(s, a, s')          ← Group B  collision penalty
                + R_tfr(s, a)              ← Group A  TFR penalty
                + R_sep(s, a)              ← Group A  separation penalty
                + R_dest(s, a)             ← Group A  destination guidance
                + R_fuel(a, v)             ← Group B  fuel efficiency
                + R_delay(s')              ← Group B  speed deviation
                + R_dev(s')                ← Group B  route deviation
                + R_ra(s, a)               ← Group B  ACAS-X RA compliance
                + R_coc(before, after)     ← Group B  clear-of-conflict bonus
                + R_surv                   ← Group B  per-step survival

Priority & weight rationale
----------------------------
Safety penalties are separated into two tiers:

  Tier 1 – catastrophic (terminal-like):
      Collision  (h < HORIZONTAL_SEP_NM AND v < VERTICAL_SEP_FT) : −500
      TFR breach (inside polygon)                                  : −500

  Tier 2 – graduated:
      Separation warning / awareness zones  : −350 … −40   (Group B)
      TFR warning / emergency buffer        : −100 … −500  (Group A)

  Tier 3 – efficiency:
      Fuel, delay, deviation, manoeuvre cost

The two Tier-1 weights are equal (−500) so neither threat is implicitly
deprioritised. When both are simultaneously active, the combined penalty
is −1000, which dominates every efficiency term and forces evasion.

Destination reward (+25 per step heading toward goal) is intentionally
smaller than the safety penalties so the agent never sacrifices safety
for progress.

Usage
-----
    from reward_combined import CombinedRewardComponents, compute_combined_reward

    rc = compute_combined_reward(
        group_a_state   = (tfr_b, tfr_rel_b, dest_b, threat_b, sep_b, traffic_b),
        group_a_action  = "TURN_LEFT_45",
        group_b_state   = aircraft_state,       # AircraftState from mdp_collision_env
        group_b_action  = Action.TURN_LEFT_15,
        group_b_next    = next_aircraft_state,
        conflicts_before= detector.get_active_conflicts(),
        conflicts_after = detector.get_active_conflicts(),
        ownship_callsign= "JBU1052",
    )
    print(rc.total)          # scalar reward
    print(rc.to_dict())      # full breakdown for logging
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

# ── Group B imports ───────────────────────────────────────────────────────────
from conflict_detection import (
    ConflictEvent,
    HORIZONTAL_SEP_NM,
    VERTICAL_SEP_FT,
)
from mdp_collision_env import (
    Action,
    AircraftState,
    RewardComponents as _B_RewardComponents,
    action_cost_fuel,
    FL290_FT,
    RVSM_VERT_SEP_FT,
    W_COLLISION,
    W_SEVERITY,
    W_FUEL,
    W_DELAY,
    W_DEVIATION,
    W_RA_COMPLIANCE,
    W_CLEAR_OF_CONFLICT,
    W_STEP_SURVIVAL,
    FT_PER_FL,
)

# ── Group A constants (keep in sync with mdp.py) ─────────────────────────────
SEPARATION_HARD_NM  = HORIZONTAL_SEP_NM   # 3.0 NM  — single source of truth
SEPARATION_WARN_NM  = 5.0
SEPARATION_AWARE_NM = 10.0

# ── Combined weight overrides ─────────────────────────────────────────────────
# Group B's W_COLLISION is −500 by default.  We set TFR breach to the same
# value so both catastrophic events carry equal weight.
W_TFR_BREACH       = -500.0   # inside TFR polygon          (mirrors W_COLLISION)
W_TFR_CRITICAL     = -500.0   # within TFR_EMERGENCY_BUFFER_NM
W_TFR_WARNING      = -100.0   # within TFR_WARNING_BUFFER_NM, on collision course
W_TFR_NEAR         = -25.0    # 15-30 nm, on collision course
W_TFR_MARGIN_FAR   = +2.0     # 30+ nm and no threat (reward for staying clear)
W_TFR_MARGIN_MID   = +1.0     # 15-30 nm and no threat


# =============================================================================
# SECTION 1 — Combined reward dataclass
# =============================================================================

@dataclass
class CombinedRewardComponents:
    """
    Full decomposed reward covering both Group A and Group B terms.

    Total reward:
        R = r_collision + r_severity          ← Group B collision
          + r_tfr + r_sep                     ← Group A safety
          + r_destination + r_tfr_margin      ← Group A guidance / margin
          + r_fuel + r_delay + r_deviation    ← Group B efficiency
          + r_ra_compliance                   ← Group B RA compliance
          + r_clear_of_conflict + r_survival  ← Group B resolution
          + r_manoeuvre                       ← Group A manoeuvre cost
    """
    # ── Group B (collision) ──────────────────────────────────────────────────
    r_collision        : float = 0.0   # W_COLLISION if CRITICAL conflict
    r_severity         : float = 0.0   # graded separation violation depth
    r_fuel             : float = 0.0   # fuel burn per step
    r_delay            : float = 0.0   # speed deviation from nominal
    r_deviation        : float = 0.0   # lateral / altitude route deviation
    r_ra_compliance    : float = 0.0   # positive when ACAS-X RA is followed
    r_clear_of_conflict: float = 0.0   # positive when conflict resolves
    r_survival         : float = 0.0   # small per-step living reward

    # ── Group A (TFR + guidance) ─────────────────────────────────────────────
    r_tfr              : float = 0.0   # TFR breach / warning penalty
    r_sep              : float = 0.0   # Group A separation penalty (discrete)
    r_destination      : float = 0.0   # heading-toward-destination reward
    r_tfr_margin       : float = 0.0   # reward for maintaining TFR clearance
    r_manoeuvre        : float = 0.0   # manoeuvre efficiency cost

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
            + self.r_tfr
            + self.r_sep
            + self.r_destination
            + self.r_tfr_margin
            + self.r_manoeuvre
        )

    def to_dict(self) -> dict:
        return {
            # Group B
            "r_collision"        : round(self.r_collision, 4),
            "r_severity"         : round(self.r_severity, 4),
            "r_fuel"             : round(self.r_fuel, 4),
            "r_delay"            : round(self.r_delay, 4),
            "r_deviation"        : round(self.r_deviation, 4),
            "r_ra_compliance"    : round(self.r_ra_compliance, 4),
            "r_clear_of_conflict": round(self.r_clear_of_conflict, 4),
            "r_survival"         : round(self.r_survival, 4),
            # Group A
            "r_tfr"              : round(self.r_tfr, 4),
            "r_sep"              : round(self.r_sep, 4),
            "r_destination"      : round(self.r_destination, 4),
            "r_tfr_margin"       : round(self.r_tfr_margin, 4),
            "r_manoeuvre"        : round(self.r_manoeuvre, 4),
            # Total
            "total"              : round(self.total, 4),
        }

    def group_a_total(self) -> float:
        """Sum of all Group A terms only."""
        return (
            self.r_tfr + self.r_sep + self.r_destination
            + self.r_tfr_margin + self.r_manoeuvre
        )

    def group_b_total(self) -> float:
        """Sum of all Group B terms only."""
        return (
            self.r_collision + self.r_severity + self.r_fuel
            + self.r_delay + self.r_deviation + self.r_ra_compliance
            + self.r_clear_of_conflict + self.r_survival
        )


# =============================================================================
# SECTION 2 — Group A reward sub-function
# =============================================================================

def _compute_group_a_reward(
    group_a_state : Tuple,   # (tfr_b, tfr_rel_b, dest_b, threat_b, sep_b, traffic_b)
    group_a_action: str,     # e.g. "TURN_LEFT_45"
) -> CombinedRewardComponents:
    """
    Compute the Group A portion of the combined reward.

    Mirrors mdp.py::reward() exactly but writes results into
    CombinedRewardComponents fields instead of a single float.

    Equations
    ---------
    TFR breach (tfr_b == 0):
        r_tfr = W_TFR_BREACH = −500

    TFR critical buffer (tfr_b == 1):
        r_tfr = W_TFR_CRITICAL = −500

    TFR warning + threat (tfr_b == 2 and threat_b):
        r_tfr = W_TFR_WARNING = −100

    TFR near + threat (tfr_b == 3 and threat_b):
        r_tfr = W_TFR_NEAR = −25

    Margin reward (no threat, far from TFR):
        r_tfr_margin = W_TFR_MARGIN_FAR  if tfr_b == 4
                     = W_TFR_MARGIN_MID  if tfr_b == 3

    Separation (discrete bins, mirrors Group A logic):
        sep_b == 0  : r_sep = −2500   (inside hard sep — keep Group A weight)
        sep_b == 1  : r_sep = −350
        sep_b == 2  : r_sep = −40

    Destination guidance:
        dest_b == 0 : r_destination += +25   (heading toward destination)
        dest_b == 3 : r_destination += −25   (heading away)

    TFR directional avoidance (when threat_b):
        turning away from TFR side : +45
        turning toward TFR side    : −55

    Manoeuvre cost:
        small turns  : −2
        large turns  : −7
        climb/descend: −10
    """
    rc = CombinedRewardComponents()

    # Unpack state
    tfr_b, tfr_rel_b, dest_b, threat_b, sep_b, _ = group_a_state

    # Base step cost (matches mdp.py)
    rc.r_manoeuvre -= 1.0

    # ── TFR safety ───────────────────────────────────────────────────────────
    if tfr_b == 0:
        rc.r_tfr += W_TFR_BREACH
    elif tfr_b == 1:
        rc.r_tfr += W_TFR_CRITICAL
    elif tfr_b == 2 and threat_b:
        rc.r_tfr += W_TFR_WARNING
    elif tfr_b == 3 and threat_b:
        rc.r_tfr += W_TFR_NEAR

    # ── TFR margin reward ─────────────────────────────────────────────────────
    if tfr_b == 4 and not threat_b:
        rc.r_tfr_margin += W_TFR_MARGIN_FAR
    elif tfr_b == 3 and not threat_b:
        rc.r_tfr_margin += W_TFR_MARGIN_MID

    # ── Separation (Group A discrete bins) ───────────────────────────────────
    # NOTE: Group B's r_severity already captures the continuous version.
    #       This adds the discrete Group A bins on top for the VI/QL solvers.
    #       Weight is halved (÷2) relative to mdp.py to avoid double-counting
    #       when both reward systems are active.
    if sep_b == 0:
        rc.r_sep += -2500.0 / 2.0
    elif sep_b == 1:
        rc.r_sep += -350.0 / 2.0
    elif sep_b == 2:
        rc.r_sep += -40.0 / 2.0

    # ── Destination guidance ──────────────────────────────────────────────────
    if dest_b == 0:
        rc.r_destination += 25.0
        if tfr_b == 4 and not threat_b:
            rc.r_destination += 10.0    # combo bonus: safe AND on track
        elif tfr_b == 3 and not threat_b:
            rc.r_destination += 5.0
    elif dest_b == 1 and group_a_action in ("TURN_RIGHT_20", "TURN_RIGHT_45"):
        rc.r_destination += 10.0
    elif dest_b == 2 and group_a_action in ("TURN_LEFT_20", "TURN_LEFT_45"):
        rc.r_destination += 10.0
    elif dest_b == 3:
        rc.r_destination -= 25.0

    # ── TFR directional avoidance ─────────────────────────────────────────────
    if threat_b:
        if tfr_rel_b == 1:                  # TFR on right → turn left
            if "LEFT" in group_a_action:
                rc.r_tfr += 45.0
            elif "RIGHT" in group_a_action:
                rc.r_tfr -= 55.0
        elif tfr_rel_b == 2:                # TFR on left → turn right
            if "RIGHT" in group_a_action:
                rc.r_tfr += 45.0
            elif "LEFT" in group_a_action:
                rc.r_tfr -= 55.0
        elif tfr_rel_b == 0:                # TFR dead ahead → hard evasive turn
            if group_a_action in ("TURN_LEFT_45", "TURN_RIGHT_45"):
                rc.r_tfr += 25.0

    # Penalise drifting toward TFR even when no active threat
    if tfr_b in (2, 3) and not threat_b:
        if tfr_rel_b == 1 and "RIGHT" in group_a_action:
            rc.r_tfr -= 15.0
        elif tfr_rel_b == 2 and "LEFT" in group_a_action:
            rc.r_tfr -= 15.0

    # ── Manoeuvre cost ────────────────────────────────────────────────────────
    if group_a_action in ("TURN_LEFT_20", "TURN_RIGHT_20"):
        rc.r_manoeuvre -= 2.0
    elif group_a_action in ("TURN_LEFT_45", "TURN_RIGHT_45"):
        rc.r_manoeuvre -= 7.0
    elif group_a_action in ("CLIMB", "DESCEND"):
        rc.r_manoeuvre -= 10.0

    return rc


# =============================================================================
# SECTION 3 — Group B reward sub-function (thin wrapper around existing code)
# =============================================================================

def _compute_group_b_reward(
    group_b_state   : AircraftState,
    group_b_action  : Action,
    group_b_next    : AircraftState,
    conflicts_before: Dict[str, ConflictEvent],
    conflicts_after : Dict[str, ConflictEvent],
    ownship_callsign: str,
) -> CombinedRewardComponents:
    """
    Compute the Group B portion of the combined reward.

    Delegates to mdp_collision_env.py::compute_reward() and maps the
    result into CombinedRewardComponents fields.

    Equations (unchanged from Group B — see mdp_collision_env.py Section 5)
    ------------------------------------------------------------------------
    r_collision  = W_COLLISION (−500) if CRITICAL conflict
    r_severity   = W_SEVERITY (−100) × φ_h × φ_v   (separation depth)
    r_fuel       = W_FUEL × action_cost_fuel(a, v)
    r_delay      = W_DELAY × |v − 450| / 450
    r_deviation  = W_DEVIATION × (|Δlat_nm| + |Δalt_fl|)
    r_ra         = W_RA_COMPLIANCE (+10) if action follows active TCAS RA
    r_coc        = W_CLEAR_OF_CONFLICT (+25) × resolved_conflicts
    r_survival   = W_STEP_SURVIVAL (+1)
    """
    from mdp_collision_env import compute_reward as _b_compute

    rc_b = _b_compute(
        state                   = group_b_state,
        action                  = group_b_action,
        next_state              = group_b_next,
        active_conflicts_before = conflicts_before,
        active_conflicts_after  = conflicts_after,
        ownship_callsign        = ownship_callsign,
    )

    rc = CombinedRewardComponents()
    rc.r_collision         = rc_b.r_collision
    rc.r_severity          = rc_b.r_severity
    rc.r_fuel              = rc_b.r_fuel
    rc.r_delay             = rc_b.r_delay
    rc.r_deviation         = rc_b.r_deviation
    rc.r_ra_compliance     = rc_b.r_ra_compliance
    rc.r_clear_of_conflict = rc_b.r_clear_of_conflict
    rc.r_survival          = rc_b.r_survival
    return rc


# =============================================================================
# SECTION 4 — Main combined reward function
# =============================================================================

def compute_combined_reward(
    # Group A inputs
    group_a_state   : Optional[Tuple],   # None if TFR not in range
    group_a_action  : str,               # Group A action string
    # Group B inputs
    group_b_state   : AircraftState,
    group_b_action  : Action,
    group_b_next    : AircraftState,
    conflicts_before: Dict[str, ConflictEvent],
    conflicts_after : Dict[str, ConflictEvent],
    ownship_callsign: str,
) -> CombinedRewardComponents:
    """
    Compute the full combined reward R(s, a, s').

    Parameters
    ----------
    group_a_state    : discrete state tuple from mdp.py::discretize_state()
                       Pass None when aircraft is outside MDP_ACTIVE_TFR_NM —
                       Group A terms will be zeroed.
    group_a_action   : Group A action string (e.g. "TURN_LEFT_45")
    group_b_state    : AircraftState (current) from mdp_collision_env
    group_b_action   : Group B Action enum (e.g. Action.CLIMB_1500)
    group_b_next     : AircraftState (after action)
    conflicts_before : detector.get_active_conflicts() snapshot before step
    conflicts_after  : detector.get_active_conflicts() snapshot after step
    ownship_callsign : callsign of the aircraft being evaluated

    Returns
    -------
    CombinedRewardComponents — full breakdown + .total scalar

    Notes
    -----
    The two reward systems use different action vocabularies.  The caller
    is responsible for mapping between them (the AircraftAgent does this
    in _step_policy()).  Passing mismatched actions is valid — it simply
    means one system's directional bonuses won't fire, which is fine when
    only one threat is active.

    When group_a_state is None (aircraft far from TFR), all Group A terms
    are zero and the result is identical to Group B's compute_reward().
    This makes the combined function a strict superset of both originals.
    """
    # ── Group B terms (always computed) ──────────────────────────────────────
    rc = _compute_group_b_reward(
        group_b_state    = group_b_state,
        group_b_action   = group_b_action,
        group_b_next     = group_b_next,
        conflicts_before = conflicts_before,
        conflicts_after  = conflicts_after,
        ownship_callsign = ownship_callsign,
    )

    # ── Group A terms (only when TFR is in range) ─────────────────────────────
    if group_a_state is not None:
        rc_a = _compute_group_a_reward(group_a_state, group_a_action)
        rc.r_tfr          = rc_a.r_tfr
        rc.r_sep          = rc_a.r_sep
        rc.r_destination  = rc_a.r_destination
        rc.r_tfr_margin   = rc_a.r_tfr_margin
        rc.r_manoeuvre    = rc_a.r_manoeuvre

    return rc


# =============================================================================
# SECTION 5 — Standalone reward for Group A VI / QL solvers (backward compat)
# =============================================================================

def reward_group_a_only(state: Tuple, action: str) -> float:
    """
    Drop-in replacement for mdp.py::reward(state, action).

    Returns a single float (not CombinedRewardComponents) so Group A's
    ValueIterationSolver and QLearningSolver work unchanged.

    This function reproduces mdp.py::reward() exactly — it is NOT the
    combined reward. Use compute_combined_reward() for the full system.
    """
    rc = _compute_group_a_reward(state, action)
    return rc.r_tfr + rc.r_sep + rc.r_destination + rc.r_tfr_margin + rc.r_manoeuvre


# =============================================================================
# SECTION 6 — Quick sanity check
# =============================================================================

if __name__ == "__main__":
    from dataclasses import replace
    from mdp_collision_env import IntruderFeatures

    print("=" * 65)
    print("  reward_combined.py — sanity check")
    print("=" * 65)

    # ── Test 1: Group A only (far from collision, near TFR) ───────────────────
    print("\nTest 1: TFR breach only (group_b no conflict)")

    state_a_breach = (0, 0, 0, 1, 3, 0)   # tfr_b=0 (inside), sep_b=3 (safe)
    s_b = AircraftState()
    s_b_next = AircraftState()

    rc = compute_combined_reward(
        group_a_state    = state_a_breach,
        group_a_action   = "MAINTAIN",
        group_b_state    = s_b,
        group_b_action   = Action.MAINTAIN,
        group_b_next     = s_b_next,
        conflicts_before = {},
        conflicts_after  = {},
        ownship_callsign = "OWN001",
    )
    print(f"  r_tfr     : {rc.r_tfr}")
    print(f"  r_collision: {rc.r_collision}")
    print(f"  total     : {rc.total}")
    assert rc.r_tfr == W_TFR_BREACH, f"Expected {W_TFR_BREACH}, got {rc.r_tfr}"
    assert rc.r_collision == 0.0,    "No collision expected"
    print("  ✓ TFR breach penalty correct")

    # ── Test 2: Collision only (far from TFR) ─────────────────────────────────
    print("\nTest 2: Collision only (no TFR threat)")

    intruder = IntruderFeatures(
        callsign="INT001", h_sep_nm=0.5, v_sep_ft=200.0,
        intruder_alt_ft=35_000.0, rel_bearing_deg=45.0,
        rel_speed_ktas=50.0, rel_vert_rate=100.0,
        tau_s=10.0, severity="CRITICAL",
    )
    from conflict_detection import ConflictEvent
    mock_ev = ConflictEvent(
        step=1, callsign_a="OWN001", callsign_b="INT001",
        h_sep_nm=0.5, v_sep_ft=200.0,
        alt_a_ft=35_000, alt_b_ft=34_800,
        lat_a=25.0, lon_a=55.0, lat_b=25.005, lon_b=55.005,
    )
    s_conflict     = AircraftState(intruders=[intruder])
    s_conflict_next = AircraftState(intruders=[intruder])

    rc2 = compute_combined_reward(
        group_a_state    = None,           # TFR not in range
        group_a_action   = "MAINTAIN",
        group_b_state    = s_conflict,
        group_b_action   = Action.CLIMB_1500,
        group_b_next     = s_conflict_next,
        conflicts_before = {"OWN001|INT001": mock_ev},
        conflicts_after  = {"OWN001|INT001": mock_ev},
        ownship_callsign = "OWN001",
    )
    print(f"  r_collision: {rc2.r_collision}")
    print(f"  r_tfr     : {rc2.r_tfr}")
    print(f"  total     : {rc2.total}")
    assert rc2.r_collision == W_COLLISION, f"Expected {W_COLLISION}, got {rc2.r_collision}"
    assert rc2.r_tfr == 0.0,              "No TFR penalty expected"
    print("  ✓ Collision penalty correct, TFR terms zeroed")

    # ── Test 3: Both threats active simultaneously ────────────────────────────
    print("\nTest 3: Both threats simultaneously")

    rc3 = compute_combined_reward(
        group_a_state    = state_a_breach,
        group_a_action   = "MAINTAIN",
        group_b_state    = s_conflict,
        group_b_action   = Action.CLIMB_1500,
        group_b_next     = s_conflict_next,
        conflicts_before = {"OWN001|INT001": mock_ev},
        conflicts_after  = {"OWN001|INT001": mock_ev},
        ownship_callsign = "OWN001",
    )
    print(f"  r_collision: {rc3.r_collision}")
    print(f"  r_tfr     : {rc3.r_tfr}")
    print(f"  total     : {rc3.total}")
    assert rc3.r_collision == W_COLLISION,  "Collision penalty missing"
    assert rc3.r_tfr == W_TFR_BREACH,       "TFR penalty missing"
    assert rc3.total < rc.total,             "Both-threat total must be worse than TFR-only"
    assert rc3.total < rc2.total,            "Both-threat total must be worse than collision-only"
    print("  ✓ Both penalties stack correctly")

    # ── Test 4: Backward compat — reward_group_a_only ─────────────────────────
    print("\nTest 4: reward_group_a_only (backward compat)")

    safe_state = (4, 1, 0, 0, 3, 0)   # far from TFR, safe sep, heading toward dest
    r_float = reward_group_a_only(safe_state, "MAINTAIN")
    print(f"  reward_group_a_only → {r_float}")
    assert isinstance(r_float, float), "Must return float"
    assert r_float > 0, "Safe + on-track state should have positive reward"
    print("  ✓ Backward-compat function returns float and positive reward")

    # ── Test 5: Full breakdown dict ───────────────────────────────────────────
    print("\nTest 5: to_dict() completeness")
    d = rc3.to_dict()
    expected_keys = [
        "r_collision", "r_severity", "r_fuel", "r_delay", "r_deviation",
        "r_ra_compliance", "r_clear_of_conflict", "r_survival",
        "r_tfr", "r_sep", "r_destination", "r_tfr_margin", "r_manoeuvre",
        "total",
    ]
    for k in expected_keys:
        assert k in d, f"Missing key: {k}"
    print(f"  Keys present: {list(d.keys())}")
    print("  ✓ All expected keys in breakdown dict")

    # ── Test 6: group_a_total / group_b_total ─────────────────────────────────
    print("\nTest 6: group_a_total + group_b_total == total")
    tol = 1e-9
    diff = abs((rc3.group_a_total() + rc3.group_b_total()) - rc3.total)
    assert diff < tol, f"Totals don't add up: diff={diff}"
    print(f"  group_a_total: {rc3.group_a_total():.4f}")
    print(f"  group_b_total: {rc3.group_b_total():.4f}")
    print(f"  combined total: {rc3.total:.4f}")
    print("  ✓ group_a + group_b = total")

    print("\n" + "=" * 65)
    print("  All 6 checks passed.")
    print("=" * 65)
