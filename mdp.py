"""
Air Traffic MDP Simulation
==========================
Group A - Navigation (TFR Avoidance)

Authors:
  Dev       - TFR & Geometry Lead   (tfr_model.py)
  Karrnidh  - MDP/POMDP Lead        (mdp_navigation.py)

What this file covers:
  - TFR polygon loading and geometry (Dev)
  - ADS-B data loading (shared/Dev)
  - Route planning around TFR (Dev)
  - MDP state space, actions, reward function (Karrnidh)
  - Value Iteration solver (Karrnidh)
  - Q-Learning solver (Karrnidh)
  - POMDP belief state + solver (Karrnidh)
  - Aircraft agent + simulation loop (Dev + Karrnidh)
  - Basic metrics output (no visualization - that's Rom's job)

NOT in this file:
  - Visualization / HTML plots (Rom)
  - Collision avoidance (Group B)
"""

# =============================================================================
# IMPORTS
# =============================================================================

import math
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================

_HERE = os.path.dirname(os.path.abspath(__file__))

# Both known spellings of the data folder are tried at runtime.
# Override with environment variable if needed.
_DATA_CANDIDATES = [
    os.path.join(_HERE, "dats"),
]
DATA_FOLDER = os.environ.get(
    "AIR_TRAFFIC_DATA_FOLDER",
    next((p for p in _DATA_CANDIDATES if os.path.isdir(p)), _DATA_CANDIDATES[0]),
)
TFR_VERTEX_FILE = os.environ.get(
    "AIR_TRAFFIC_TFR_FILE",
    os.path.join(_HERE, "TFR_Lat_Lon.xlsx"),
)

# TFR polygon mode: auto | raw | angle | hull
TFR_POLYGON_MODE = os.environ.get("AIR_TRAFFIC_TFR_MODE", "auto").lower()

# Simulation timing
GAP_THRESHOLD_SECONDS  = 60
STEP_DURATION_SECONDS  = 30
CONFLICT_LOOKAHEAD_STEPS = 8
EXTRA_ROUTE_STEPS      = 1000   # increased to give long-haul detour flights enough steps

# Separation thresholds (nautical miles)
SEPARATION_HARD_NM  = 3.0
SEPARATION_WARN_NM  = 5.0
SEPARATION_AWARE_NM = 10.0

# TFR buffer distances (nautical miles)
TFR_WARNING_BUFFER_NM  = 10.0
TFR_EMERGENCY_BUFFER_NM = 3.0
TFR_REJOIN_BUFFER_NM   = 18.0
TFR_DETOUR_BUFFER_NM   = 28.0

# Altitude limits
MIN_AIRBORNE_ALT_FT   = 5000.0
ALT_FLOOR_FT          = MIN_AIRBORNE_ALT_FT
ALT_CEILING_FT        = 45000.0
ALT_CLIMB_RATE_FT     = 1000.0
ALT_DRIFT_TO_NOMINAL_FT = 100.0

# Heading / turn limits
GUIDANCE_TURN_DEG = 20.0
EVASIVE_TURN_DEG  = 45.0
WAYPOINT_CAPTURE_NM = 20.0

# MDP hyperparameters
MDP_ACTIVE_TFR_NM  = 45.0
MDP_GAMMA          = 0.97     # increased from 0.95 - agent values future rewards more
MDP_VI_TOL         = 1e-3     # back to original tolerance - tighter was too slow
MDP_VI_MAX_ITERS   = 500      # back to original
MDP_T_SAMPLES      = 4        # back to 8 - 12 was the main cause of slow runtime

# Q-Learning hyperparameters (Karrnidh)
MDP_QL_EPISODES    = 25000    # increased from 15000 — ensures full state coverage
MDP_QL_STEPS       = 150      # increased from 130 for longer rollouts
MDP_QL_ALPHA_START = 0.4
MDP_QL_ALPHA_END   = 0.03
MDP_QL_EPS_START   = 1.0
MDP_QL_EPS_END     = 0.02
MDP_RNG_SEED       = 42

MAX_PLOT_POINTS = 700


# =============================================================================
# DEV'S SECTION: GEOMETRY HELPERS
# =============================================================================

EARTH_RADIUS_NM = 3440.0


def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance in nautical miles."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_NM * np.arcsin(np.sqrt(a))


def bearing_between(lat1, lon1, lat2, lon2):
    """Compass bearing from point 1 to point 2, in degrees."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0


def move_by_heading(lat, lon, heading_deg, speed_kts, time_seconds):
    """Dead-reckon one step."""
    distance_nm   = (speed_kts / 3600.0) * time_seconds
    angular_dist  = distance_nm / EARTH_RADIUS_NM
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    hdg_rad = math.radians(heading_deg)

    new_lat = math.asin(
        math.sin(lat_rad) * math.cos(angular_dist)
        + math.cos(lat_rad) * math.sin(angular_dist) * math.cos(hdg_rad)
    )
    new_lon = lon_rad + math.atan2(
        math.sin(hdg_rad) * math.sin(angular_dist) * math.cos(lat_rad),
        math.cos(angular_dist) - math.sin(lat_rad) * math.sin(new_lat),
    )
    return math.degrees(new_lat), math.degrees(new_lon)


def relative_bearing(my_heading_deg, target_bearing_deg):
    """Target bearing relative to nose. Positive = right, negative = left."""
    return (target_bearing_deg - my_heading_deg + 540.0) % 360.0 - 180.0


def turn_toward(current_heading, target_heading, max_turn_deg):
    diff = relative_bearing(current_heading, target_heading)
    turn = max(-max_turn_deg, min(max_turn_deg, diff))
    return (current_heading + turn) % 360.0, turn


def bearing_quadrant(rel_bearing_deg):
    """0=ahead, 1=right, 2=left, 3=behind."""
    rb = rel_bearing_deg
    if -30.0 <= rb <= 30.0:
        return 0
    if 30.0 < rb <= 150.0:
        return 1
    if -150.0 <= rb < -30.0:
        return 2
    return 3


def path_length_nm(points: Sequence[Tuple[float, float]]):
    if len(points) < 2:
        return 0.0
    total = 0.0
    for a, b in zip(points[:-1], points[1:]):
        total += haversine(a[0], a[1], b[0], b[1])
    return float(total)


# =============================================================================
# DEV'S SECTION: TFR POLYGON (tfr_model.py)
# =============================================================================

@dataclass
class TFRDiagnostics:
    raw_lats:           np.ndarray
    raw_lons:           np.ndarray
    raw_order:          np.ndarray
    angle_order:        np.ndarray
    hull_order:         np.ndarray
    selected_order:     np.ndarray
    selected_mode:      str
    raw_intersections:  int
    angle_intersections: int
    hull_intersections: int


class TFRZone:
    """
    DEV - tfr_model.py
    Polygonal TFR with signed distance and planning helpers.
    Responsibilities:
      - Load and clean polygon vertices from Excel
      - Auto-select best vertex ordering (raw / angle-sorted / convex hull)
      - Answer: is point inside TFR? (contains)
      - Answer: signed distance to TFR edge (distance_to_edge)
      - Answer: is a path segment clear of the TFR buffer? (segment_clear)
      - Generate expanded vertices for detour planning (expanded_vertices)
    """

    def __init__(self, vertices_lat, vertices_lon, mode=TFR_POLYGON_MODE):
        lat = np.asarray(vertices_lat, dtype=float)
        lon = np.asarray(vertices_lon, dtype=float)
        lat, lon = self._clean_vertices(lat, lon)
        if len(lat) < 3:
            raise ValueError("TFR polygon needs at least three valid vertices.")

        raw_order   = np.arange(len(lat))
        angle_order = self._angle_sort_order(lat, lon)
        hull_order  = self._convex_hull_order(lat, lon)

        raw_i   = self._count_self_intersections(lat, lon, raw_order)
        angle_i = self._count_self_intersections(lat, lon, angle_order)
        hull_i  = self._count_self_intersections(lat, lon, hull_order)

        selected_mode = mode
        if selected_mode == "raw":
            selected_order = raw_order
        elif selected_mode == "angle":
            selected_order = angle_order
        elif selected_mode == "hull":
            selected_order = hull_order
        elif selected_mode == "auto":
            if raw_i == 0:
                selected_mode, selected_order = "raw", raw_order
            elif angle_i == 0:
                selected_mode, selected_order = "angle", angle_order
            else:
                selected_mode, selected_order = "hull", hull_order
        else:
            raise ValueError(f"Unknown TFR polygon mode: {mode!r}")

        self.raw_lats     = lat
        self.raw_lons     = lon
        self.lats         = lat[selected_order]
        self.lons         = lon[selected_order]
        self.n_vertices   = len(self.lats)
        self.mode         = selected_mode
        self.centroid_lat, self.centroid_lon = self._polygon_centroid(self.lats, self.lons)
        self.diagnostics  = TFRDiagnostics(
            raw_lats=lat, raw_lons=lon,
            raw_order=raw_order, angle_order=angle_order, hull_order=hull_order,
            selected_order=selected_order, selected_mode=selected_mode,
            raw_intersections=raw_i, angle_intersections=angle_i, hull_intersections=hull_i,
        )

    def contains(self, lat, lon):
        if self._min_edge_distance(lat, lon) < 1e-6:
            return True
        return self._ray_cast(lat, lon)

    def distance_to_edge(self, lat, lon):
        """Negative = inside TFR, positive = outside."""
        d = self._min_edge_distance(lat, lon)
        return -d if self.contains(lat, lon) else d

    def segment_clear(self, lat1, lon1, lat2, lon2, buffer_nm):
        dist = float(haversine(lat1, lon1, lat2, lon2))
        n = max(3, min(90, int(dist / 12.0) + 2))
        for f in np.linspace(0.0, 1.0, n):
            lat = lat1 + f * (lat2 - lat1)
            lon = lon1 + f * (lon2 - lon1)
            if self.distance_to_edge(lat, lon) < buffer_nm:
                return False
        return True

    def expanded_vertices(self, buffer_nm):
        pts = []
        for lat, lon in zip(self.lats, self.lons):
            brg = bearing_between(self.centroid_lat, self.centroid_lon, lat, lon)
            d   = float(haversine(self.centroid_lat, self.centroid_lon, lat, lon)) + buffer_nm
            pts.append(move_by_heading(self.centroid_lat, self.centroid_lon, brg, d, 3600.0))
        return pts

    # ------------------------------------------------------------------
    # Internal helpers (Dev)
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_vertices(lat, lon):
        keep = np.isfinite(lat) & np.isfinite(lon)
        lat, lon = lat[keep], lon[keep]
        seen = set()
        out_lat, out_lon = [], []
        for la, lo in zip(lat, lon):
            key = (round(float(la), 8), round(float(lo), 8))
            if key in seen:
                continue
            seen.add(key)
            out_lat.append(float(la))
            out_lon.append(float(lo))
        return np.asarray(out_lat), np.asarray(out_lon)

    @staticmethod
    def _angle_sort_order(lat, lon):
        cy, cx = float(np.mean(lat)), float(np.mean(lon))
        return np.argsort(np.arctan2(lat - cy, lon - cx))

    @staticmethod
    def _convex_hull_order(lat, lon):
        pts = sorted(range(len(lat)), key=lambda i: (lon[i], lat[i]))

        def cross(o, a, b):
            return (lon[a] - lon[o]) * (lat[b] - lat[o]) - (lat[a] - lat[o]) * (lon[b] - lon[o])

        lower = []
        for i in pts:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], i) <= 0:
                lower.pop()
            lower.append(i)
        upper = []
        for i in reversed(pts):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], i) <= 0:
                upper.pop()
            upper.append(i)
        return np.asarray(lower[:-1] + upper[:-1], dtype=int)

    @staticmethod
    def _polygon_centroid(lat, lon):
        x = np.asarray(lon, dtype=float)
        y = np.asarray(lat, dtype=float)
        cross = x * np.roll(y, -1) - np.roll(x, -1) * y
        area2 = float(np.sum(cross))
        if abs(area2) < 1e-12:
            return float(np.mean(y)), float(np.mean(x))
        cx = float(np.sum((x + np.roll(x, -1)) * cross) / (3.0 * area2))
        cy = float(np.sum((y + np.roll(y, -1)) * cross) / (3.0 * area2))
        return cy, cx

    def _ray_cast(self, lat, lon):
        inside = False
        j = self.n_vertices - 1
        for i in range(self.n_vertices):
            yi, xi = self.lats[i], self.lons[i]
            yj, xj = self.lats[j], self.lons[j]
            if (yi > lat) != (yj > lat):
                x_at_lat = (xj - xi) * (lat - yi) / ((yj - yi) + 1e-12) + xi
                if lon < x_at_lat:
                    inside = not inside
            j = i
        return inside

    def _min_edge_distance(self, lat, lon):
        ref_lat = self.centroid_lat
        px = self._lon_to_x(lon, ref_lat)
        py = self._lat_to_y(lat)
        xs = self._lon_to_x(self.lons, ref_lat)
        ys = self._lat_to_y(self.lats)
        best = float("inf")
        for i in range(self.n_vertices):
            j = (i + 1) % self.n_vertices
            best = min(best, self._point_to_segment(px, py, xs[i], ys[i], xs[j], ys[j]))
        return best

    @staticmethod
    def _lon_to_x(lon_deg, ref_lat_deg):
        return np.radians(lon_deg) * EARTH_RADIUS_NM * math.cos(math.radians(ref_lat_deg))

    @staticmethod
    def _lat_to_y(lat_deg):
        return np.radians(lat_deg) * EARTH_RADIUS_NM

    @staticmethod
    def _point_to_segment(px, py, ax, ay, bx, by):
        abx, aby = bx - ax, by - ay
        apx, apy = px - ax, py - ay
        denom = abx * abx + aby * aby
        t = 0.0 if denom == 0 else max(0.0, min(1.0, (apx * abx + apy * aby) / denom))
        cx = ax + t * abx
        cy = ay + t * aby
        return math.hypot(px - cx, py - cy)

    @classmethod
    def _count_self_intersections(cls, lat, lon, order):
        count = 0
        n = len(order)
        for a in range(n):
            a1, a2 = order[a], order[(a + 1) % n]
            for b in range(a + 1, n):
                if abs(a - b) <= 1 or {a, b} == {0, n - 1}:
                    continue
                b1, b2 = order[b], order[(b + 1) % n]
                if cls._segments_intersect(
                    lon[a1], lat[a1], lon[a2], lat[a2],
                    lon[b1], lat[b1], lon[b2], lat[b2],
                ):
                    count += 1
        return count

    @staticmethod
    def _segments_intersect(ax, ay, bx, by, cx, cy, dx, dy):
        def orient(px, py, qx, qy, rx, ry):
            return (qy - py) * (rx - qx) - (qx - px) * (ry - qy)
        o1 = orient(ax, ay, bx, by, cx, cy)
        o2 = orient(ax, ay, bx, by, dx, dy)
        o3 = orient(cx, cy, dx, dy, ax, ay)
        o4 = orient(cx, cy, dx, dy, bx, by)
        return o1 * o2 < 0 and o3 * o4 < 0


def load_tfr_from_excel(path):
    """DEV - Load TFR polygon from Excel file."""
    try:
        df = pd.read_excel(path)
        return TFRZone(df["Lat_dd"].tolist(), df["Lon_dd"].tolist())
    except Exception as exc:
        print(f"  ! Could not load TFR file ({exc}); using fallback circle.")
        cx_lat, cx_lon, r_nm = 25.997, -97.156, 30.0
        ang  = np.linspace(0, 2 * np.pi, 36, endpoint=False)
        lats = cx_lat + (r_nm / 60.0) * np.sin(ang)
        lons = cx_lon + (r_nm / 60.0) * np.cos(ang) / math.cos(math.radians(cx_lat))
        return TFRZone(lats, lons, mode="raw")


# =============================================================================
# DEV'S SECTION: ADS-B DATA LOADING (data_loader.py)
# =============================================================================

def fill_gaps(df, gap_threshold_seconds=GAP_THRESHOLD_SECONDS):
    """DEV - Interpolate missing ADS-B pings and flag them as interpolated."""
    rows = []
    for i in range(len(df) - 1):
        rows.append(df.iloc[i].to_dict())
        gap = (df["UTC"].iloc[i + 1] - df["UTC"].iloc[i]).total_seconds()
        if gap > gap_threshold_seconds:
            n_fill = int(gap / STEP_DURATION_SECONDS) - 1
            for step in range(1, n_fill + 1):
                frac = (step * STEP_DURATION_SECONDS) / gap
                f = df.iloc[i].to_dict()
                f["UTC"] = df["UTC"].iloc[i] + pd.Timedelta(seconds=step * STEP_DURATION_SECONDS)
                for col in ("lat", "lon", "Altitude", "Speed"):
                    f[col] = df[col].iloc[i] + frac * (df[col].iloc[i + 1] - df[col].iloc[i])
                f["interpolated"] = True
                rows.append(f)
    rows.append(df.iloc[-1].to_dict())
    out = pd.DataFrame(rows).reset_index(drop=True)
    out["interpolated"] = out.get("interpolated", False).fillna(False).astype(bool)
    return out


def load_aircraft_data(folder_path):
    """
    DEV - Load all CSV files from dats folder.

    Task 6 (unified data loading): delegates to Group B's load_flight_data()
    so both groups use an identical loading pipeline.  The returned dict
    {callsign: DataFrame} retains the same contract as before, so all
    existing callers (AircraftAgent, run_simulation, main) are unchanged.

    Falls back to the original inline loader if air_traffic_abm is not on
    the path (e.g. when running mdp.py in isolation during development).
    """
    if not os.path.isdir(folder_path):
        print(f"  ! Folder not found: {folder_path}")
        return {}

    try:
        from air_traffic_abm import load_flight_data as _load_b
        df_all = _load_b(folder_path)
        if df_all.empty:
            return {}
        aircraft = {}
        for cs, grp in df_all.groupby("Callsign"):
            grp = grp.reset_index(drop=True)
            if len(grp) >= 2:
                aircraft[str(cs)] = grp
        print(f"  Loaded {len(aircraft)} aircraft via unified loader.")
        return aircraft

    except ImportError:
        # ── Fallback: original standalone loader ──────────────────────────
        print("  air_traffic_abm not found; using standalone loader.")
        aircraft = {}
        csvs = [f for f in os.listdir(folder_path) if f.lower().endswith(".csv")]
        print(f"  Found {len(csvs)} CSV file(s).")
        total_dropped = 0
        for filename in csvs:
            path = os.path.join(folder_path, filename)
            df   = pd.read_csv(path)
            df["UTC"] = pd.to_datetime(df["UTC"])
            df[["lat", "lon"]] = df["Position"].str.split(",", expand=True).astype(float)
            df["interpolated"] = False
            before = len(df)
            df = df[df["Altitude"] >= MIN_AIRBORNE_ALT_FT].reset_index(drop=True)
            total_dropped += before - len(df)
            if df.empty:
                continue
            if "Callsign" in df.columns:
                for callsign, group in df.groupby("Callsign"):
                    if len(group) < 2:
                        continue
                    group = group.sort_values("UTC").reset_index(drop=True)
                    aircraft[str(callsign)] = fill_gaps(group)
            else:
                callsign = filename.replace(".csv", "")
                df = df.sort_values("UTC").reset_index(drop=True)
                aircraft[callsign] = fill_gaps(df)
        if total_dropped:
            print(f"  Dropped {total_dropped} rows below {MIN_AIRBORNE_ALT_FT:.0f} ft.")
        return aircraft


# =============================================================================
# DEV'S SECTION: ROUTE PLANNING AROUND TFR
# =============================================================================

def _index_path(i, j, n, clockwise=True):
    """DEV - Walk polygon vertex indices clockwise or counter-clockwise."""
    out = []
    k = i
    direction = 1 if clockwise else -1
    while True:
        out.append(k)
        if k == j:
            return out
        k = (k + direction) % n


def plan_tfr_detour(tfr: TFRZone, start, dest, buffer_nm):
    """
    DEV - Return waypoint list around the buffered TFR.
    Empty list means direct path is clear.
    """
    if tfr.segment_clear(start[0], start[1], dest[0], dest[1], buffer_nm):
        return []

    expanded = tfr.expanded_vertices(max(TFR_DETOUR_BUFFER_NM, buffer_nm + 15.0))
    n = len(expanded)
    best_route, best_cost = None, float("inf")

    for i in range(n):
        for j in range(n):
            for clockwise in (True, False):
                seq_idx = _index_path(i, j, n, clockwise)
                route   = [expanded[k] for k in seq_idx]
                points  = [start] + route + [dest]
                if not tfr.segment_clear(start[0], start[1], route[0][0], route[0][1], buffer_nm * 0.75):
                    continue
                if not tfr.segment_clear(route[-1][0], route[-1][1], dest[0], dest[1], buffer_nm * 0.75):
                    continue
                cost = path_length_nm(points) + 8.0 * len(route)
                if cost < best_cost:
                    best_cost, best_route = cost, route

    if best_route is not None:
        return best_route

    # Fallback: single best expanded vertex
    best_route, best_cost = [], float("inf")
    for p in expanded:
        cost = path_length_nm([start, p, dest])
        if cost < best_cost:
            best_cost, best_route = cost, [p]
    return best_route


# =============================================================================
# KARRNIDH'S SECTION: MDP DEFINITION (mdp_navigation.py)
# =============================================================================

# --- Actions -----------------------------------------------------------------
# 7 discrete actions the agent can take at each 30-second step

ACTIONS = (
    "MAINTAIN",       # hold current heading and altitude
    "TURN_LEFT_20",   # turn left 20 degrees
    "TURN_RIGHT_20",  # turn right 20 degrees
    "TURN_LEFT_45",   # turn left 45 degrees (evasive)
    "TURN_RIGHT_45",  # turn right 45 degrees (evasive)
    "CLIMB",          # climb 1000 ft
    "DESCEND",        # descend 1000 ft
)


# --- State Space -------------------------------------------------------------
# State = (tfr_dist_bin, tfr_direction, dest_direction, tfr_threat, sep_bin, traffic_direction)
# Total states = 5 x 4 x 4 x 2 x 4 x 5 = 3,200

def discretize_state(tfr_dist_nm, tfr_rel_bearing, dest_rel_bearing, tfr_threat, sep_min_nm, threat_rel_bearing):
    """
    KARRNIDH - Convert continuous aircraft situation into a discrete MDP state.

    State dimensions:
      tfr_b   : distance to TFR edge (0=inside, 1=<5nm, 2=5-15nm, 3=15-30nm, 4=30+nm)
      tfr_rel_b: TFR direction relative to heading (0=ahead, 1=right, 2=left, 3=behind)
      dest_b  : destination direction relative to heading (same 4 quadrants)
      threat_b: is the aircraft on a collision course with the TFR? (0=no, 1=yes)
      sep_b   : nearest aircraft separation (0=<3nm danger, 1=3-5nm warn, 2=5-10nm aware, 3=safe)
      traffic_b: direction of nearest traffic (0=none, 1=ahead, 2=right, 3=left, 4=behind)
    """
    # TFR distance bin
    if tfr_dist_nm < 0:
        tfr_b = 0                        # inside TFR
    elif tfr_dist_nm < 5:
        tfr_b = 1
    elif tfr_dist_nm < 15:
        tfr_b = 2
    elif tfr_dist_nm < 30:
        tfr_b = 3
    else:
        tfr_b = 4

    tfr_rel_b = bearing_quadrant(tfr_rel_bearing)
    dest_b    = bearing_quadrant(dest_rel_bearing)
    threat_b  = 1 if tfr_threat else 0

    # Separation bin
    if sep_min_nm < SEPARATION_HARD_NM:
        sep_b = 0
    elif sep_min_nm < SEPARATION_WARN_NM:
        sep_b = 1
    elif sep_min_nm < SEPARATION_AWARE_NM:
        sep_b = 2
    else:
        sep_b = 3

    # Traffic direction bin
    if threat_rel_bearing is None or sep_b == 3:
        traffic_b = 0
    else:
        traffic_b = {0: 1, 1: 2, 2: 3, 3: 4}[bearing_quadrant(threat_rel_bearing)]

    return (tfr_b, tfr_rel_b, dest_b, threat_b, sep_b, traffic_b)


def all_states():
    """KARRNIDH - Enumerate all 3,200 possible states."""
    return [
        (t, tr, d, h, s, tb)
        for t  in range(5)
        for tr in range(4)
        for d  in range(4)
        for h  in range(2)
        for s  in range(4)
        for tb in range(5)
    ]


# --- Reward Function ---------------------------------------------------------

def reward(state, action):
    """
    KARRNIDH - Reward function R(s, a).

    Design philosophy:
      - Safety first: massive penalties for TFR breach and separation loss
      - Graduated penalties: smaller penalties for being close to TFR
      - Directional guidance: reward turning toward destination
      - Directional avoidance: strongly reward turning AWAY from TFR side
      - Efficiency: small penalties for unnecessary maneuvers
      - Margin bonus: reward maintaining comfortable distance from TFR
        (this is the key improvement - agent should WANT to stay clear, not just avoid breach)
    """
    tfr_b, tfr_rel_b, dest_b, threat_b, sep_b, _ = state
    r = -1.0  # base step cost

    # ---- TFR safety penalties ----
    if tfr_b == 0:
        r -= 2500.0                         # inside TFR: catastrophic
    elif tfr_b == 1:
        r -= 500.0                          # within 5nm: critical
    elif tfr_b == 2 and threat_b:
        r -= 100.0                          # 5-15nm on collision course
    elif tfr_b == 3 and threat_b:
        r -= 25.0                           # 15-30nm on collision course

    # ---- Margin reward ----
    if tfr_b == 4 and not threat_b:
        r += 2.0
    elif tfr_b == 3 and not threat_b:
        r += 1.0

    # ---- Separation safety penalties ----
    if sep_b == 0:
        r -= 2500.0                         # inside hard separation: catastrophic
    elif sep_b == 1:
        r -= 350.0                          # within warning zone
    elif sep_b == 2:
        r -= 40.0                           # within awareness zone

    # ---- Destination guidance rewards ----
    if dest_b == 0:
        r += 25.0                           # heading toward destination
        if tfr_b == 4 and not threat_b:
            r += 10.0                       # combo bonus: safe AND on track
        elif tfr_b == 3 and not threat_b:
            r += 5.0                        # partial combo for bin 3
    elif dest_b == 1 and action in ("TURN_RIGHT_20", "TURN_RIGHT_45"):
        r += 10.0
    elif dest_b == 2 and action in ("TURN_LEFT_20", "TURN_LEFT_45"):
        r += 10.0
    elif dest_b == 3:
        r -= 25.0                           # heading away from destination

    # ---- TFR avoidance directional rewards ----
    # If TFR is to the right, left turns are better (and vice versa)
    if threat_b:
        if tfr_rel_b == 1:                  # TFR on right side
            if "LEFT" in action:
                r += 45.0
            elif "RIGHT" in action:
                r -= 55.0
        elif tfr_rel_b == 2:                # TFR on left side
            if "RIGHT" in action:
                r += 45.0
            elif "LEFT" in action:
                r -= 55.0
        elif tfr_rel_b == 0:                # TFR dead ahead
            if action in ("TURN_LEFT_45", "TURN_RIGHT_45"):
                r += 25.0                   # hard evasive turn needed

    # ---- NEW: penalize turns toward TFR even without active threat ----
    # Prevents drifting toward TFR when not yet flagged as threat
    if tfr_b in (2, 3) and not threat_b:
        if tfr_rel_b == 1 and "RIGHT" in action:
            r -= 15.0                       # turning toward TFR unnecessarily
        elif tfr_rel_b == 2 and "LEFT" in action:
            r -= 15.0

    # ---- Maneuver cost penalties (efficiency) ----
    if action in ("TURN_LEFT_20", "TURN_RIGHT_20"):
        r -= 2.0
    elif action in ("TURN_LEFT_45", "TURN_RIGHT_45"):
        r -= 7.0
    elif action in ("CLIMB", "DESCEND"):
        # NEW: reduced from -20 to -10 so altitude escape is more viable
        r -= 10.0

    return r


# =============================================================================
# KARRNIDH'S SECTION: TRANSITION MODEL SAMPLING
# =============================================================================

@dataclass
class _Sample:
    lat:       float
    lon:       float
    heading:   float
    speed:     float
    altitude:  float
    dest_lat:  float
    dest_lon:  float
    other_lat: Optional[float]
    other_lon: Optional[float]


def _rel_range(bin_id):
    return {
        0: (-25.0,  25.0),
        1: ( 35.0, 140.0),
        2: (-140.0, -35.0),
        3: (155.0, 205.0),
    }[bin_id]


def _draw_sample_for_bin(state, tfr: TFRZone, rng):
    """KARRNIDH - Sample a concrete aircraft situation consistent with the given state bin."""
    tfr_b, tfr_rel_b, dest_b, _, sep_b, traffic_b = state
    cx_lat, cx_lon = tfr.centroid_lat, tfr.centroid_lon

    if tfr_b == 0:
        lat, lon = cx_lat, cx_lon
    else:
        d = {
            1: rng.uniform(2.0, 5.0),
            2: rng.uniform(7.0, 14.0),
            3: rng.uniform(18.0, 28.0),
            4: rng.uniform(35.0, 80.0),
        }[tfr_b]
        lat, lon = move_by_heading(cx_lat, cx_lon, rng.uniform(0.0, 360.0), d, 3600.0)

    tfr_brg = bearing_between(lat, lon, cx_lat, cx_lon)
    lo, hi  = _rel_range(tfr_rel_b)
    heading = (tfr_brg - rng.uniform(lo, hi)) % 360.0
    speed    = rng.uniform(360.0, 480.0)
    altitude = rng.uniform(28000.0, 38000.0)

    lo, hi    = _rel_range(dest_b)
    dest_brg  = (heading + rng.uniform(lo, hi)) % 360.0
    dest_lat, dest_lon = move_by_heading(lat, lon, dest_brg, 550.0, 1800.0)

    if sep_b == 3:
        return _Sample(lat, lon, heading, speed, altitude, dest_lat, dest_lon, None, None)

    other_dist = {0: rng.uniform(0.5, 2.5), 1: rng.uniform(3.2, 4.8), 2: rng.uniform(5.5, 9.5)}[sep_b]
    rel = {
        1: rng.uniform(-25.0,  25.0),
        2: rng.uniform(35.0,  140.0),
        3: rng.uniform(-140.0, -35.0),
        4: rng.uniform(155.0, 205.0),
    }.get(traffic_b, rng.uniform(0.0, 360.0))
    other_brg = (heading + rel) % 360.0
    other_lat, other_lon = move_by_heading(lat, lon, other_brg, other_dist, 3600.0)
    return _Sample(lat, lon, heading, speed, altitude, dest_lat, dest_lon, other_lat, other_lon)


def _apply_action_to_sample(sample: _Sample, action, rng=None):
    """KARRNIDH - Apply an action to a sampled state and return next sample."""
    h   = sample.heading
    alt = sample.altitude
    if action == "TURN_LEFT_20":
        h = (h - 20.0) % 360.0
    elif action == "TURN_RIGHT_20":
        h = (h + 20.0) % 360.0
    elif action == "TURN_LEFT_45":
        h = (h - 45.0) % 360.0
    elif action == "TURN_RIGHT_45":
        h = (h + 45.0) % 360.0
    elif action == "CLIMB":
        alt += ALT_CLIMB_RATE_FT
    elif action == "DESCEND":
        alt -= ALT_CLIMB_RATE_FT
    alt = max(ALT_FLOOR_FT, min(ALT_CEILING_FT, alt))

    lat, lon = move_by_heading(sample.lat, sample.lon, h, sample.speed, STEP_DURATION_SECONDS)
    if sample.other_lat is None:
        other_lat, other_lon = None, None
    else:
        rr = rng or random
        other_lat = sample.other_lat + rr.uniform(-0.01, 0.01)
        other_lon = sample.other_lon + rr.uniform(-0.01, 0.01)
    return _Sample(lat, lon, h, sample.speed, alt, sample.dest_lat, sample.dest_lon, other_lat, other_lon)


def _classify_sample(sample: _Sample, tfr: TFRZone):
    """KARRNIDH - Classify a sample back into a discrete state."""
    tfr_dist = tfr.distance_to_edge(sample.lat, sample.lon)
    tfr_brg  = bearing_between(sample.lat, sample.lon, tfr.centroid_lat, tfr.centroid_lon)
    tfr_rel  = relative_bearing(sample.heading, tfr_brg)
    dest_brg = bearing_between(sample.lat, sample.lon, sample.dest_lat, sample.dest_lon)
    dest_rel = relative_bearing(sample.heading, dest_brg)

    # Look ahead to check if on collision course with TFR
    threat = False
    lat, lon = sample.lat, sample.lon
    for _ in range(10):
        lat, lon = move_by_heading(lat, lon, sample.heading, sample.speed, STEP_DURATION_SECONDS)
        if tfr.distance_to_edge(lat, lon) < TFR_WARNING_BUFFER_NM:
            threat = True
            break

    if sample.other_lat is None:
        sep, traffic_rel = 999.0, None
    else:
        sep         = haversine(sample.lat, sample.lon, sample.other_lat, sample.other_lon)
        traffic_brg = bearing_between(sample.lat, sample.lon, sample.other_lat, sample.other_lon)
        traffic_rel = relative_bearing(sample.heading, traffic_brg)

    return discretize_state(tfr_dist, tfr_rel, dest_rel, threat, sep, traffic_rel)


# =============================================================================
# KARRNIDH'S SECTION: VALUE ITERATION SOLVER
# =============================================================================

class ValueIterationSolver:
    """
    KARRNIDH - MDP solver using Value Iteration.

    Solves: V*(s) = max_a [ R(s,a) + gamma * sum_s' T(s,a,s') * V*(s') ]

    The transition model T(s,a,s') is estimated by sampling MDP_T_SAMPLES
    concrete aircraft situations for each (state, action) pair and simulating
    the outcome. This gives us an empirical transition distribution.

    After convergence, the optimal policy is: pi*(s) = argmax_a Q(s,a)
    """

    def __init__(self, tfr: TFRZone, gamma=MDP_GAMMA, n_samples=MDP_T_SAMPLES, seed=MDP_RNG_SEED):
        self.tfr      = tfr
        self.gamma    = gamma
        self.n_samples = n_samples
        self.rng      = random.Random(seed)
        self.states   = all_states()
        self.V        = {s: 0.0 for s in self.states}
        self.policy   = {s: "MAINTAIN" for s in self.states}
        self.T        = {}
        self.R        = {}

    def build_model(self):
        print(f"  Building MDP model: |S|={len(self.states)}, |A|={len(ACTIONS)}, K={self.n_samples}")
        for s in self.states:
            self.T[s] = {}
            self.R[s] = {}
            for a in ACTIONS:
                counts = defaultdict(int)
                r_sum  = 0.0
                for _ in range(self.n_samples):
                    sample      = _draw_sample_for_bin(s, self.tfr, self.rng)
                    next_sample = _apply_action_to_sample(sample, a, self.rng)
                    sn          = _classify_sample(next_sample, self.tfr)
                    counts[sn] += 1
                    r_sum      += reward(sn, a)
                total         = float(sum(counts.values()))
                self.T[s][a] = {sn: c / total for sn, c in counts.items()}
                self.R[s][a] = r_sum / self.n_samples

    def solve(self):
        """Run Bellman value iteration until convergence."""
        for it in range(MDP_VI_MAX_ITERS):
            delta = 0.0
            new_v = {}
            for s in self.states:
                best_q, best_a = -float("inf"), "MAINTAIN"
                for a in ACTIONS:
                    q  = self.R[s][a]
                    q += sum(self.gamma * p * self.V[sn] for sn, p in self.T[s][a].items())
                    if q > best_q:
                        best_q, best_a = q, a
                new_v[s]      = best_q
                self.policy[s] = best_a
                delta = max(delta, abs(best_q - self.V[s]))
            self.V = new_v
            if delta < MDP_VI_TOL:
                print(f"  Value iteration converged in {it + 1} iterations (delta={delta:.2e}).")
                return
        print(f"  Value iteration hit max iters ({MDP_VI_MAX_ITERS}).")

    def act(self, state):
        return self.policy.get(state, "MAINTAIN")


# =============================================================================
# KARRNIDH'S SECTION: Q-LEARNING SOLVER
# =============================================================================

class QLearningSolver:
    """
    KARRNIDH - MDP solver using Q-Learning.

    Key improvements over original:
      1. More episodes (5000 vs 3000) and steps (100 vs 80)
      2. Learning rate decay: alpha starts at 0.3 and decays to 0.05
         (aggressive early learning, stable convergence late)
      3. Exponential epsilon decay (smoother exploration schedule)
      4. Terminal state penalty: when agent enters TFR or hard separation,
         apply terminal reward and reset episode immediately rather than
         just breaking — ensures the Q-values correctly propagate the danger
      5. State visitation tracking: skip update if state never visited
         (avoids stale Q-values from unvisited states)
    """

    def __init__(self, tfr: TFRZone, seed=MDP_RNG_SEED + 1):
        self.tfr    = tfr
        self.rng    = random.Random(seed)
        self.states = all_states()
        self.Q      = {s: {a: 0.0 for a in ACTIONS} for s in self.states}
        self._visits = defaultdict(int)  # track how many times each state was visited

        # Biased start-state distribution — weight near-TFR bins heavily so
        # the critical avoidance states (tfr_b 0-2) are well-covered.
        # Without this, ~74% of random starts land in safe tfr_b=3/4 states,
        # leaving tfr_b=0,1,2 largely unexplored (root cause of 26% coverage).
        #   tfr_b 0 (inside TFR) — weight 8
        #   tfr_b 1 (<5 nm)      — weight 10
        #   tfr_b 2 (5-15 nm)    — weight 8
        #   tfr_b 3 (15-30 nm)   — weight 4
        #   tfr_b 4 (30+ nm)     — weight 2
        _TFRD_WEIGHTS = {0: 8, 1: 10, 2: 8, 3: 4, 4: 2}
        total_w = sum(_TFRD_WEIGHTS[s[0]] for s in self.states)
        self._start_weights = [_TFRD_WEIGHTS[s[0]] / total_w for s in self.states]

    def _epsilon(self, ep):
        """Exponential epsilon decay for smoother exploration schedule."""
        decay = math.exp(-5.0 * ep / MDP_QL_EPISODES)
        return MDP_QL_EPS_END + (MDP_QL_EPS_START - MDP_QL_EPS_END) * decay

    def _alpha(self, ep):
        """Learning rate decay: aggressive early, stable late."""
        decay = math.exp(-3.0 * ep / MDP_QL_EPISODES)
        return MDP_QL_ALPHA_END + (MDP_QL_ALPHA_START - MDP_QL_ALPHA_END) * decay

    def _greedy(self, s):
        return max(self.Q[s], key=self.Q[s].get)

    def _is_terminal(self, s):
        """State is terminal if aircraft is inside TFR or in hard separation loss."""
        tfr_b, _, _, _, sep_b, _ = s
        return tfr_b == 0 or sep_b == 0

    def train(self):
        print(f"  Q-learning training: {MDP_QL_EPISODES} episodes x {MDP_QL_STEPS} steps "
              f"(biased start-state sampling)")
        for ep in range(MDP_QL_EPISODES):
            eps   = self._epsilon(ep)
            alpha = self._alpha(ep)

            # Biased start: weight near-TFR bins more heavily so critical
            # avoidance states are well-covered (fixes 26% → ~90%+ coverage).
            start_state = self.rng.choices(self.states, weights=self._start_weights, k=1)[0]
            sample = _draw_sample_for_bin(start_state, self.tfr, self.rng)
            s      = _classify_sample(sample, self.tfr)

            for _ in range(MDP_QL_STEPS):
                # Epsilon-greedy action selection
                if self.rng.random() < eps:
                    a = self.rng.choice(ACTIONS)
                else:
                    a = self._greedy(s)

                next_sample = _apply_action_to_sample(sample, a, self.rng)
                sn          = _classify_sample(next_sample, self.tfr)
                r           = reward(sn, a)

                # Terminal state handling: apply large penalty and end episode
                if self._is_terminal(sn):
                    r -= 1000.0   # additional terminal penalty
                    td = r        # no future value from terminal state
                    self.Q[s][a] += alpha * (td - self.Q[s][a])
                    self._visits[s] += 1
                    break

                # Standard Q-learning update
                td = r + MDP_GAMMA * max(self.Q[sn].values())
                self.Q[s][a] += alpha * (td - self.Q[s][a])
                self._visits[s] += 1
                sample, s = next_sample, sn

        visited = sum(1 for v in self._visits.values() if v > 0)
        print(f"  Q-learning complete. States visited: {visited}/{len(self.states)}")

    def act(self, state):
        if state not in self.Q or self._visits[state] == 0:
            return "MAINTAIN"   # safe default for unvisited states
        return self._greedy(state)


# =============================================================================
# KARRNIDH'S SECTION: POMDP BELIEF STATE + SOLVER (mdp_navigation.py)
# =============================================================================

# Observation noise parameters
# These control how uncertain the agent is about its true state.
# Higher noise = more uncertainty = more conservative behavior.
# Lower noise = closer to MDP behavior.
POMDP_TFR_NOISE_NM  = 3.0    # GPS/sensor uncertainty in TFR distance (nautical miles)
POMDP_SEP_NOISE_NM  = 1.5    # uncertainty in separation distance
POMDP_THREAT_FLIP   = 0.10   # probability that threat flag observation is wrong


class BeliefState:
    """
    KARRNIDH - POMDP belief state over the MDP state space.

    Instead of knowing the exact state, the agent maintains a probability
    distribution (belief) over all 3,200 possible states.

    At each step:
      1. Agent takes an action
      2. Agent receives a noisy observation of the new state
      3. Belief is updated using Bayes' rule:
         b'(s') ∝ O(o | s', a) * Σ_s T(s, a, s') * b(s)

    For tractability we use a simplified observation model:
      - TFR distance observation is the true bin ± noise (adjacent bins get weight)
      - Threat flag has a small probability of being flipped
      - Separation bin has similar noise model

    Action selection uses point estimation (most likely state) passed to VI policy.
    This is practical and standard for large discrete POMDPs.
    """

    def __init__(self):
        self.states  = all_states()
        self.n       = len(self.states)
        # Start with uniform belief over all states
        self.belief  = {s: 1.0 / self.n for s in self.states}

    def _obs_prob(self, obs_state, true_state):
        """
        KARRNIDH - P(observation | true state).
        Models sensor noise for TFR distance, separation, and threat flag.
        """
        t_tfr, t_tfr_rel, t_dest, t_threat, t_sep, t_traffic = true_state
        o_tfr, o_tfr_rel, o_dest, o_threat, o_sep, o_traffic  = obs_state

        prob = 1.0

        # TFR distance noise: true bin observed correctly with high prob,
        # adjacent bins with lower prob (simulates GPS error of ~3nm)
        tfr_diff = abs(o_tfr - t_tfr)
        if tfr_diff == 0:
            prob *= 0.75
        elif tfr_diff == 1:
            prob *= 0.20
        elif tfr_diff == 2:
            prob *= 0.04
        else:
            prob *= 0.01

        # Direction bins observed perfectly (bearing is reliable)
        if o_tfr_rel != t_tfr_rel:
            prob *= 0.05
        if o_dest != t_dest:
            prob *= 0.05

        # Threat flag: small chance of being wrong
        if o_threat == t_threat:
            prob *= (1.0 - POMDP_THREAT_FLIP)
        else:
            prob *= POMDP_THREAT_FLIP

        # Separation noise: similar to TFR distance
        sep_diff = abs(o_sep - t_sep)
        if sep_diff == 0:
            prob *= 0.75
        elif sep_diff == 1:
            prob *= 0.20
        else:
            prob *= 0.05

        # Traffic direction: reliable if separation is close, noisy if far
        if t_sep == 3:   # no traffic nearby - direction doesn't matter
            prob *= 1.0
        elif o_traffic == t_traffic:
            prob *= 0.80
        else:
            prob *= 0.05

        return prob

    def update(self, action, obs_state, transition_model=None):
        """
        KARRNIDH - Bayesian belief update after taking action and observing obs_state.

        b'(s') ∝ O(obs | s') * Σ_s T(s, a, s') * b(s)

        For efficiency, if no transition model is provided we use a simplified
        version that just applies observation noise to the current belief without
        the full transition step. This is the 'observation-only' update.
        """
        new_belief = {}

        if transition_model and action in transition_model.get(list(self.belief.keys())[0], {}):
            # Full Bayesian update with transition model
            # Predict step: b_pred(s') = Σ_s T(s,a,s') * b(s)
            b_pred = defaultdict(float)
            for s, bs in self.belief.items():
                if bs < 1e-10:
                    continue
                if s in transition_model and action in transition_model[s]:
                    for sn, prob in transition_model[s][action].items():
                        b_pred[sn] += prob * bs
                else:
                    b_pred[s] += bs

            # Update step: b'(s') ∝ O(obs|s') * b_pred(s')
            for s in self.states:
                o_prob = self._obs_prob(obs_state, s)
                new_belief[s] = o_prob * b_pred.get(s, 0.0)
        else:
            # Observation-only update: b'(s) ∝ O(obs|s) * b(s)
            for s, bs in self.belief.items():
                o_prob = self._obs_prob(obs_state, s)
                new_belief[s] = o_prob * bs

        # Normalize
        total = sum(new_belief.values())
        if total > 1e-10:
            self.belief = {s: v / total for s, v in new_belief.items()}
        else:
            # Belief collapsed - reset to uniform (rare edge case)
            self.belief = {s: 1.0 / self.n for s in self.states}

    def most_likely_state(self):
        """KARRNIDH - Point estimation: return the state with highest belief."""
        return max(self.belief, key=self.belief.get)

    def expected_tfr_bin(self):
        """KARRNIDH - Expected TFR distance bin under current belief (for conservative action)."""
        return sum(s[0] * b for s, b in self.belief.items())

    def reset(self):
        self.belief = {s: 1.0 / self.n for s in self.states}


class POMDPSolver:
    """
    KARRNIDH - POMDP solver using belief-state point estimation.

    At each step:
      1. Get noisy observation of current state
      2. Update belief using Bayes rule
      3. Pick action using most likely state → VI policy

    This is practical for our state space size. Full POMDP solvers
    (PBVI, SARSOP) are intractable for 3,200 states in real-time.

    Key difference from MDP:
      - The agent doesn't trust its observations perfectly
      - It maintains uncertainty and acts more conservatively near TFR
      - This better models real aircraft with imperfect sensors
    """

    def __init__(self, vi_solver: ValueIterationSolver, tfr: TFRZone, seed=MDP_RNG_SEED + 2):
        self.vi      = vi_solver
        self.tfr     = tfr
        self.rng     = random.Random(seed)
        # Each aircraft gets its own belief state (keyed by unique_id)
        self._beliefs: Dict[str, BeliefState] = {}

    def _add_observation_noise(self, true_state):
        """
        KARRNIDH - Simulate noisy sensor reading of true state.
        Returns an observed state that may differ from true state.
        """
        tfr_b, tfr_rel, dest_b, threat_b, sep_b, traffic_b = true_state

        # TFR distance noise: shift bin by ±1 with some probability
        r = self.rng.random()
        if r < 0.12 and tfr_b > 0:
            tfr_b_obs = tfr_b - 1      # sensor thinks closer than reality
        elif r < 0.24 and tfr_b < 4:
            tfr_b_obs = tfr_b + 1      # sensor thinks further than reality
        else:
            tfr_b_obs = tfr_b

        # Threat flag noise
        if self.rng.random() < POMDP_THREAT_FLIP:
            threat_obs = 1 - threat_b
        else:
            threat_obs = threat_b

        # Separation noise
        r = self.rng.random()
        if r < 0.10 and sep_b > 0:
            sep_obs = sep_b - 1        # sensor thinks closer
        elif r < 0.20 and sep_b < 3:
            sep_obs = sep_b + 1        # sensor thinks further
        else:
            sep_obs = sep_b

        return (tfr_b_obs, tfr_rel, dest_b, threat_obs, sep_obs, traffic_b)

    def get_belief(self, aircraft_id: str) -> BeliefState:
        if aircraft_id not in self._beliefs:
            self._beliefs[aircraft_id] = BeliefState()
        return self._beliefs[aircraft_id]

    def act(self, aircraft_id: str, true_state, last_action: str = "MAINTAIN",
            transition_model=None):
        """
        KARRNIDH - POMDP action selection.

        1. Generate noisy observation from true state
        2. Update belief
        3. Use most likely state for VI policy lookup
        4. If belief is very uncertain near TFR, be more conservative
        """
        belief = self.get_belief(aircraft_id)

        # Get noisy observation and update belief
        obs = self._add_observation_noise(true_state)
        belief.update(last_action, obs, transition_model)

        # Point estimation: act on most likely state
        most_likely = belief.most_likely_state()

        # Conservative override: if expected TFR bin suggests we might be
        # closer than most likely state indicates, use more cautious state
        expected_tfr = belief.expected_tfr_bin()
        if expected_tfr < most_likely[0] - 0.5:
            # Belief is skewed toward being closer to TFR than point estimate
            # Use a more conservative state (one bin closer)
            conservative = (
                max(0, most_likely[0] - 1),
                most_likely[1], most_likely[2],
                1,   # assume threat active when uncertain
                most_likely[4], most_likely[5]
            )
            return self.vi.act(conservative)

        return self.vi.act(most_likely)

    def reset_aircraft(self, aircraft_id: str):
        if aircraft_id in self._beliefs:
            self._beliefs[aircraft_id].reset()


# =============================================================================
# DEV + KARRNIDH: AIRCRAFT AGENT (model.py)
# =============================================================================

class AircraftAgent:
    """
    DEV (movement, routing, data replay) + KARRNIDH (MDP state, action execution)
    Modes: replay | rule | mdp_vi | mdp_ql | pomdp
    """

    def __init__(self, unique_id, model, trajectory, mode="replay"):
        self.unique_id  = unique_id
        self.model      = model
        self.trajectory = trajectory
        self.mode       = mode
        self.data_index = 0

        row0 = trajectory.iloc[0]
        self.lat      = float(row0["lat"])
        self.lon      = float(row0["lon"])
        self.altitude = self._clamp_altitude(float(row0["Altitude"]))
        self.speed    = float(row0["Speed"])

        if len(trajectory) > 1:
            r1 = trajectory.iloc[1]
            self.heading = bearing_between(row0["lat"], row0["lon"], r1["lat"], r1["lon"])
        else:
            self.heading = float(row0.get("Direction", 0.0))

        last = trajectory.iloc[-1]
        self.destination_lat = float(last["lat"])
        self.destination_lon = float(last["lon"])

        cruise = trajectory[trajectory["Altitude"] >= MIN_AIRBORNE_ALT_FT]
        nominal_alt = float(cruise["Altitude"].mean()) if len(cruise) else 35000.0
        self.nominal_altitude = self._clamp_altitude(nominal_alt)
        self.nominal_speed    = float(trajectory["Speed"].mean())

        self.route_waypoints: List[Tuple[float, float]] = []
        self.finished            = False
        self.reached_destination = False
        self.in_tfr              = self.model.tfr.contains(self.lat, self.lon)

        # History for metrics and visualization
        self.history_lat          = [self.lat]
        self.history_lon          = [self.lon]
        self.history_altitude     = [self.altitude]
        self.history_time         = [0]
        self.history_in_tfr       = [self.in_tfr]
        self.history_tfr_distance = [self.model.tfr.distance_to_edge(self.lat, self.lon)]
        self.history_action       = ["INIT"]
        self.history_target       = ["DEST"]
        self._last_mdp_action     = "MAINTAIN"  # track for POMDP belief update

    @staticmethod
    def _clamp_altitude(alt):
        return max(ALT_FLOOR_FT, min(ALT_CEILING_FT, alt))

    def _dest_distance(self):
        return haversine(self.lat, self.lon, self.destination_lat, self.destination_lon)

    def _predict_position(self, heading, steps_ahead):
        lat, lon = self.lat, self.lon
        for _ in range(steps_ahead):
            lat, lon = move_by_heading(lat, lon, heading, self.speed, STEP_DURATION_SECONDS)
        return lat, lon

    def _predict_current(self, steps_ahead):
        return self._predict_position(self.heading, steps_ahead)

    def tfr_conflict_ahead(self, buffer_nm=TFR_WARNING_BUFFER_NM):
        if self.model.tfr.distance_to_edge(self.lat, self.lon) < buffer_nm:
            return True
        for k in range(1, CONFLICT_LOOKAHEAD_STEPS + 1):
            lat, lon = self._predict_current(k)
            if self.model.tfr.distance_to_edge(lat, lon) < buffer_nm:
                return True
        return False

    def separation_threats(self):
        threats = []
        for other in self.model.aircraft_agents:
            if other.unique_id == self.unique_id or other.finished:
                continue
            for k in range(1, CONFLICT_LOOKAHEAD_STEPS + 1):
                a_lat, a_lon = self._predict_current(k)
                b_lat, b_lon = other._predict_current(k)
                if haversine(a_lat, a_lon, b_lat, b_lon) < SEPARATION_WARN_NM:
                    threats.append(other)
                    break
        return threats

    def _refresh_route(self):
        """DEV - Update detour waypoints, drop captured ones."""
        while self.route_waypoints:
            wp = self.route_waypoints[0]
            if haversine(self.lat, self.lon, wp[0], wp[1]) <= WAYPOINT_CAPTURE_NM:
                self.route_waypoints.pop(0)
            else:
                break

        start  = (self.lat, self.lon)
        dest   = (self.destination_lat, self.destination_lon)
        buffer = TFR_WARNING_BUFFER_NM

        direct_clear  = self.model.tfr.segment_clear(self.lat, self.lon, self.destination_lat, self.destination_lon, buffer)
        far_from_tfr  = self.model.tfr.distance_to_edge(self.lat, self.lon) > TFR_REJOIN_BUFFER_NM
        if self.route_waypoints and direct_clear and far_from_tfr:
            self.route_waypoints = []
        if not self.route_waypoints and not direct_clear:
            self.route_waypoints = plan_tfr_detour(self.model.tfr, start, dest, buffer)

    def _guidance_target(self):
        self._refresh_route()
        if self.route_waypoints:
            return self.route_waypoints[0], "ROUTE"
        return (self.destination_lat, self.destination_lon), "DEST"

    def _guidance_action(self, max_turn=GUIDANCE_TURN_DEG):
        target, label = self._guidance_target()
        target_hdg    = bearing_between(self.lat, self.lon, target[0], target[1])
        _, turn       = turn_toward(self.heading, target_hdg, max_turn)
        if turn < -32.5:   return "TURN_LEFT_45",  label
        if turn < -8.0:    return "TURN_LEFT_20",   label
        if turn > 32.5:    return "TURN_RIGHT_45",  label
        if turn > 8.0:     return "TURN_RIGHT_20",  label
        return "MAINTAIN", label

    def _state_for_mdp(self):
        """KARRNIDH - Build MDP state from current aircraft situation."""
        tfr_dist = self.model.tfr.distance_to_edge(self.lat, self.lon)
        tfr_brg  = bearing_between(self.lat, self.lon, self.model.tfr.centroid_lat, self.model.tfr.centroid_lon)
        tfr_rel  = relative_bearing(self.heading, tfr_brg)
        target, _ = self._guidance_target()
        dest_brg = bearing_between(self.lat, self.lon, target[0], target[1])
        dest_rel = relative_bearing(self.heading, dest_brg)

        threats = self.separation_threats()
        if threats:
            t         = threats[0]
            sep       = haversine(self.lat, self.lon, t.lat, t.lon)
            threat_brg = bearing_between(self.lat, self.lon, t.lat, t.lon)
            threat_rel = relative_bearing(self.heading, threat_brg)
        else:
            sep, threat_rel = 999.0, None

        return discretize_state(tfr_dist, tfr_rel, dest_rel, self.tfr_conflict_ahead(), sep, threat_rel)

    def _candidate_score(self, action, solver_action=None):
        """DEV - Score an action by simulating 12 steps ahead."""
        h   = self.heading
        alt = self.altitude
        if action == "TURN_LEFT_20":    h = (h - 20.0) % 360.0
        elif action == "TURN_RIGHT_20": h = (h + 20.0) % 360.0
        elif action == "TURN_LEFT_45":  h = (h - 45.0) % 360.0
        elif action == "TURN_RIGHT_45": h = (h + 45.0) % 360.0
        elif action == "CLIMB":         alt += ALT_CLIMB_RATE_FT
        elif action == "DESCEND":       alt -= ALT_CLIMB_RATE_FT

        target, _    = self._guidance_target()
        start_dest_d = haversine(self.lat, self.lon, target[0], target[1])
        lat, lon     = move_by_heading(self.lat, self.lon, h, self.speed, STEP_DURATION_SECONDS)

        min_tfr = self.model.tfr.distance_to_edge(lat, lon)
        min_sep = 999.0
        look_h  = h
        sim_lat, sim_lon = lat, lon

        for step in range(1, 13):
            target_h = bearing_between(sim_lat, sim_lon, target[0], target[1])
            look_h, _ = turn_toward(look_h, target_h, GUIDANCE_TURN_DEG)
            sim_lat, sim_lon = move_by_heading(sim_lat, sim_lon, look_h, self.speed, STEP_DURATION_SECONDS)
            min_tfr = min(min_tfr, self.model.tfr.distance_to_edge(sim_lat, sim_lon))
            for other in self.model.aircraft_agents:
                if other.unique_id == self.unique_id or other.finished:
                    continue
                o_lat, o_lon = other._predict_current(step)
                min_sep = min(min_sep, float(haversine(sim_lat, sim_lon, o_lat, o_lon)))

        end_dest_d = haversine(sim_lat, sim_lon, target[0], target[1])
        progress   = start_dest_d - end_dest_d

        score  = 8.0 * progress
        score -= 0.02 * abs(relative_bearing(self.heading, h))
        score -= 0.003 * abs(alt - self.nominal_altitude)

        if min_tfr < 0:
            score -= 50000.0
        elif min_tfr < TFR_EMERGENCY_BUFFER_NM:
            score -= 8000.0
        elif min_tfr < TFR_WARNING_BUFFER_NM:
            score -= 1200.0 * (TFR_WARNING_BUFFER_NM - min_tfr + 1.0)

        if min_sep < SEPARATION_HARD_NM:
            score -= 50000.0
        elif min_sep < SEPARATION_WARN_NM:
            score -= 1800.0 * (SEPARATION_WARN_NM - min_sep + 1.0)

        if action == solver_action:
            score += 20.0
        if action in ("CLIMB", "DESCEND"):
            score -= 120.0
        return score

    def _best_local_action(self, solver_action=None):
        guidance_action, label = self._guidance_action(EVASIVE_TURN_DEG)
        candidates = set(ACTIONS)
        candidates.add(guidance_action)
        if solver_action:
            candidates.add(solver_action)
        best = max(candidates, key=lambda a: self._candidate_score(a, solver_action))
        return best, label

    def _apply_action(self, action):
        """KARRNIDH - Execute the chosen action."""
        if action == "TURN_LEFT_20":
            self.heading = (self.heading - 20.0) % 360.0
        elif action == "TURN_RIGHT_20":
            self.heading = (self.heading + 20.0) % 360.0
        elif action == "TURN_LEFT_45":
            self.heading = (self.heading - 45.0) % 360.0
        elif action == "TURN_RIGHT_45":
            self.heading = (self.heading + 45.0) % 360.0
        elif action == "CLIMB":
            self.altitude += ALT_CLIMB_RATE_FT
        elif action == "DESCEND":
            self.altitude -= ALT_CLIMB_RATE_FT
        else:
            # Gradually drift back to nominal altitude when maintaining
            if self.altitude > self.nominal_altitude + 100:
                self.altitude -= ALT_DRIFT_TO_NOMINAL_FT
            elif self.altitude < self.nominal_altitude - 100:
                self.altitude += ALT_DRIFT_TO_NOMINAL_FT
        self.altitude = self._clamp_altitude(self.altitude)
        self.lat, self.lon = move_by_heading(self.lat, self.lon, self.heading, self.speed, STEP_DURATION_SECONDS)

    def _step_replay(self):
        """DEV - Replay real ADS-B trajectory."""
        if self.data_index < len(self.trajectory) - 1:
            old_lat, old_lon = self.lat, self.lon
            self.data_index += 1
            row = self.trajectory.iloc[self.data_index]
            self.lat      = float(row["lat"])
            self.lon      = float(row["lon"])
            self.heading  = bearing_between(old_lat, old_lon, self.lat, self.lon)
            self.altitude = self._clamp_altitude(float(row["Altitude"]))
            self.speed    = float(row["Speed"])
            if self.data_index == len(self.trajectory) - 1:
                self.finished = self.reached_destination = True
            return "REPLAY", "CSV"
        self.finished = self.reached_destination = True
        return "DONE", "CSV"

    def _step_policy(self):
        """DEV + KARRNIDH - One step of policy-based flight."""
        if self._dest_distance() < 10.0:
            self.finished = self.reached_destination = True
            return "DONE", "DEST"

        guidance_action, label = self._guidance_action(GUIDANCE_TURN_DEG)
        active = (
            self.model.tfr.distance_to_edge(self.lat, self.lon) < MDP_ACTIVE_TFR_NM
            or self.tfr_conflict_ahead()
            or bool(self.separation_threats())
            or bool(self.route_waypoints)
        )

        if self.mode == "rule":
            action = self._best_local_action()[0] if active else guidance_action

        elif self.mode == "mdp_vi":
            if active:
                solver_action = self.model.vi_solver.act(self._state_for_mdp())
                action, label = self._best_local_action(solver_action)
                action = f"VI_{action}"
            else:
                action = guidance_action

        elif self.mode == "mdp_ql":
            if active:
                solver_action = self.model.ql_solver.act(self._state_for_mdp())
                action, label = self._best_local_action(solver_action)
                action = f"QL_{action}"
            else:
                action = guidance_action

        elif self.mode == "pomdp":
            if active:
                true_state    = self._state_for_mdp()
                solver_action = self.model.pomdp_solver.act(
                    self.unique_id, true_state, self._last_mdp_action,
                    self.model.vi_solver.T
                )
                action, label = self._best_local_action(solver_action)
                action = f"POMDP_{action}"
            else:
                action = guidance_action

        else:
            action = guidance_action

        base_action = action.replace("VI_", "").replace("QL_", "").replace("POMDP_", "")
        self._last_mdp_action = base_action
        self._apply_action(base_action)
        return action, label

    def step(self):
        if self.finished:
            return
        if self.mode == "replay":
            action, target_label = self._step_replay()
        else:
            action, target_label = self._step_policy()

        self.in_tfr = self.model.tfr.contains(self.lat, self.lon)
        self.history_lat.append(self.lat)
        self.history_lon.append(self.lon)
        self.history_altitude.append(self.altitude)
        self.history_time.append(self.model.time)
        self.history_in_tfr.append(self.in_tfr)
        self.history_tfr_distance.append(self.model.tfr.distance_to_edge(self.lat, self.lon))
        self.history_action.append(action)
        self.history_target.append(target_label)


# =============================================================================
# DEV + KARRNIDH: SIMULATION MODEL (model.py)
# =============================================================================

class AirTrafficModel:
    """DEV + KARRNIDH - Top-level simulation containing all aircraft."""

    def __init__(self, aircraft_data, tfr: TFRZone, mode="replay", vi_solver=None, ql_solver=None, pomdp_solver=None):
        self.time         = 0
        self.mode         = mode
        self.tfr          = tfr
        self.vi_solver    = vi_solver
        self.ql_solver    = ql_solver
        self.pomdp_solver = pomdp_solver
        self.aircraft_agents = [AircraftAgent(cs, self, df, mode) for cs, df in aircraft_data.items()]
        self.violations            = []
        self.tfr_warnings          = []
        self.separation_violations = []
        self._active_tfr  = set()
        self._active_warn = set()
        self._active_sep  = set()

    def all_finished(self):
        return all(ac.finished for ac in self.aircraft_agents)

    def _check_violations(self):
        """DEV - Detect and record TFR and separation violations."""
        for ac in self.aircraft_agents:
            if ac.finished:
                continue
            cs = ac.unique_id
            d  = self.tfr.distance_to_edge(ac.lat, ac.lon)

            if 0 <= d < TFR_WARNING_BUFFER_NM:
                if cs not in self._active_warn:
                    self.tfr_warnings.append((cs, self.time, round(d, 2)))
                    self._active_warn.add(cs)
            else:
                self._active_warn.discard(cs)

            if d < 0:
                if cs not in self._active_tfr:
                    self.violations.append((cs, self.time, round(-d, 2)))
                    self._active_tfr.add(cs)
            else:
                self._active_tfr.discard(cs)

        n = len(self.aircraft_agents)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = self.aircraft_agents[i], self.aircraft_agents[j]
                if a.finished or b.finished:
                    continue
                d    = float(haversine(a.lat, a.lon, b.lat, b.lon))
                pair = frozenset([a.unique_id, b.unique_id])
                if d < SEPARATION_HARD_NM:
                    if pair not in self._active_sep:
                        self.separation_violations.append((a.unique_id, b.unique_id, self.time, round(d, 2)))
                        self._active_sep.add(pair)
                else:
                    self._active_sep.discard(pair)

    def step(self):
        for ac in self.aircraft_agents:
            ac.step()
        self.time += 1
        self._check_violations()

    def metrics(self):
        return {
            "mode":                 self.mode,
            "aircraft_count":       len(self.aircraft_agents),
            "steps":                self.time,
            "tfr_warnings":         len(self.tfr_warnings),
            "tfr_violations":       len(self.violations),
            "separation_violations": len(self.separation_violations),
            "currently_in_tfr":     sum(1 for ac in self.aircraft_agents if ac.in_tfr),
            "reached_destination":  sum(1 for ac in self.aircraft_agents if ac.reached_destination),
            "alt_min_ft":           min((min(ac.history_altitude) for ac in self.aircraft_agents), default=0.0),
            "alt_max_ft":           max((max(ac.history_altitude) for ac in self.aircraft_agents), default=0.0),
            # NEW: average minimum TFR distance across all aircraft (higher = safer margin)
            "avg_min_tfr_dist_nm":  float(np.mean([
                min(ac.history_tfr_distance) for ac in self.aircraft_agents
            ])),
        }


# =============================================================================
# MAIN: TRAINING + SIMULATION RUNNER
# =============================================================================

def print_summary(model):
    m   = model.metrics()
    bar = "=" * 64
    print()
    print(bar)
    print(f"  SIMULATION COMPLETE - {m['mode'].upper()}")
    print(bar)
    print(f"  Aircraft               : {m['aircraft_count']}")
    print(f"  Steps run              : {m['steps']}")
    print(f"  Reached destination    : {m['reached_destination']} / {m['aircraft_count']}")
    print(f"  TFR buffer warnings    : {m['tfr_warnings']}")
    print(f"  TFR breaches           : {m['tfr_violations']}")
    print(f"  Separation losses      : {m['separation_violations']}")
    print(f"  In TFR right now       : {m['currently_in_tfr']}")
    print(f"  Altitude range         : {m['alt_min_ft']:.0f} - {m['alt_max_ft']:.0f} ft")
    print(f"  Avg min TFR distance   : {m['avg_min_tfr_dist_nm']:.2f} nm")
    print(bar)
    if model.violations:
        print("  Aircraft that breached the TFR:")
        for cs, t, d in model.violations:
            print(f"    {cs:>10s}   step {t:>4d}   penetration {d:.2f} nm")
    else:
        print("  No TFR breaches.")


def run_simulation(aircraft_data, tfr, mode, vi_solver=None, ql_solver=None, pomdp_solver=None, max_steps=None):
    if max_steps is None:
        max_steps = max(len(df) for df in aircraft_data.values()) + EXTRA_ROUTE_STEPS
    model = AirTrafficModel(aircraft_data, tfr, mode=mode, vi_solver=vi_solver, ql_solver=ql_solver, pomdp_solver=pomdp_solver)
    for _ in range(max_steps - 1):
        if model.all_finished():
            break
        model.step()
    return model


def main():
    print("=" * 64)
    print("  Air Traffic MDP Simulation - Group A (Dev + Karrnidh)")
    print("=" * 64)

    print("\n[1/4] Loading TFR polygon...")
    tfr  = load_tfr_from_excel(TFR_VERTEX_FILE)
    diag = tfr.diagnostics
    print(f"  TFR mode      : {tfr.mode}")
    print(f"  Vertices      : {tfr.n_vertices}")
    print(f"  Centroid      : ({tfr.centroid_lat:.3f}, {tfr.centroid_lon:.3f})")
    print(f"  Self-intersections (raw/angle/hull): {diag.raw_intersections}/{diag.angle_intersections}/{diag.hull_intersections}")

    print("\n[2/4] Loading aircraft trajectories...")
    aircraft_data = load_aircraft_data(DATA_FOLDER)
    if not aircraft_data:
        raise SystemExit("No aircraft data found.")
    print(f"  Loaded {len(aircraft_data)} aircraft.")
    for cs, df in aircraft_data.items():
        dist = haversine(df["lat"].iloc[0], df["lon"].iloc[0], df["lat"].iloc[-1], df["lon"].iloc[-1])
        print(f"    {cs:>10s}: {len(df):>4d} rows, {dist:6.0f} nm origin-destination")

    max_steps = max(len(df) for df in aircraft_data.values()) + EXTRA_ROUTE_STEPS

    print("\n[3/4] Training MDP solvers (Karrnidh)...")
    vi = ValueIterationSolver(tfr)
    vi.build_model()
    vi.solve()

    ql = QLearningSolver(tfr)
    ql.train()

    pomdp = POMDPSolver(vi, tfr)
    print(f"  POMDP solver ready (uses VI policy with belief-state point estimation).")

    print("\n[4/4] Running simulations...")
    models = []
    for mode in ("replay", "rule", "mdp_vi", "mdp_ql", "pomdp"):
        print(f"\n  --- mode: {mode} ---")
        model = run_simulation(aircraft_data, tfr, mode, vi_solver=vi, ql_solver=ql, pomdp_solver=pomdp, max_steps=max_steps)
        print_summary(model)
        models.append(model)

    # Comparison table
    print("\n" + "=" * 72)
    print(f"{'mode':<10}{'TFR warn':>10}{'TFR':>8}{'SEP':>8}{'reached':>12}{'avg TFR dist':>14}")
    print("=" * 72)
    for model in models:
        x = model.metrics()
        print(
            f"{x['mode']:<10}{x['tfr_warnings']:>10d}{x['tfr_violations']:>8d}"
            f"{x['separation_violations']:>8d}{x['reached_destination']:>7d}/{x['aircraft_count']:<4d}"
            f"{x['avg_min_tfr_dist_nm']:>13.2f} nm"
        )


if __name__ == "__main__":
    main()