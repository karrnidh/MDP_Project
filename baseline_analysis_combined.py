"""
baseline_analysis_combined.py
==============================
Owner  : Rom + Brishnee (Analysis & Visualisation)
         + Dev (TFR overlay)
Project: MDP Air Traffic Simulation  |  ITSEC 2026

Combined baseline analysis covering BOTH:
  - Group B: collision detection, separation timeline, heatmap
  - Group A: TFR polygon overlay, TFR proximity timeline, combined safety panel

Produces six outputs:
    baseline_map_combined.html      — Folium map: trajectories + conflict
                                      locations + TFR polygon overlay
    baseline_metrics_combined.png   — 6-panel figure (5 original + 1 new)
    baseline_conflicts.csv          — collision event log  (unchanged)
    baseline_summary.csv            — per-aircraft summary (extended)
    baseline_tfr_metrics.csv        — per-aircraft TFR metrics  (new)
    baseline_combined_safety.csv    — step-level combined safety timeline (new)

Run:
    python baseline_analysis_combined.py
    python baseline_analysis_combined.py --tfr TFR_Lat_Lon.xlsx
"""

import argparse
import os
import glob
import warnings
from math import radians, cos, sin, asin, sqrt

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import seaborn as sns
import folium

warnings.filterwarnings("ignore")


# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Both spelling variants of the data folder are tried
_DATA_CANDIDATES = [
    os.path.join(SCRIPT_DIR, "Validation data (FlightRadar24)"),
    os.path.join(SCRIPT_DIR, "Validation data (FlightRader24)"),
]
DATA_DIR = next((p for p in _DATA_CANDIDATES if os.path.isdir(p)), _DATA_CANDIDATES[0])

TFR_PATH_DEFAULT = os.path.join(SCRIPT_DIR, "TFR_Lat_Lon.xlsx")

H_SEP_NM           = 3.0
V_SEP_FT           = 1_000.0
TFR_WARNING_NM     = 10.0    # Group A warning buffer

PALETTE = {
    "conflict"  : "#e05c2e",
    "safe"      : "#3a7bd5",
    "warning"   : "#f0a500",
    "tfr"       : "#9b30ff",   # purple for TFR
    "tfr_fill"  : "#c77dff",
    "threshold" : "#e05c2e",
    "grid"      : "#e8e8e8",
}


# =============================================================================
# Utilities
# =============================================================================

def haversine_nm(lat1, lon1, lat2, lon2) -> float:
    R = 3440.065
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * R * asin(sqrt(a))


# =============================================================================
# Data loading
# =============================================================================

def load_trajectories(data_dir: str) -> dict:
    trajs = {}
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files found in: {data_dir}")
    for f in files:
        df = pd.read_csv(f)
        df[["lat", "lon"]] = (
            df["Position"].str.split(",", expand=True).astype(float)
        )
        df = df.sort_values("Timestamp").reset_index(drop=True)
        cs = df["Callsign"].iloc[0]
        trajs[cs] = df
    print(f"  Loaded {len(trajs)} aircraft: {list(trajs.keys())}")
    return trajs


def load_tfr(tfr_path: str):
    """Load TFRZone using Group A's loader. Returns None if unavailable."""
    if not tfr_path or not os.path.isfile(tfr_path):
        print(f"  TFR file not found ({tfr_path}); TFR overlay disabled.")
        return None
    try:
        from mdp import load_tfr_from_excel
        tfr = load_tfr_from_excel(tfr_path)
        print(f"  TFR loaded: {tfr.n_vertices} vertices, "
              f"centroid ({tfr.centroid_lat:.3f}, {tfr.centroid_lon:.3f})")
        return tfr
    except Exception as exc:
        print(f"  TFR load failed ({exc}); overlay disabled.")
        return None


# =============================================================================
# Simulation — collision + TFR proximity
# =============================================================================

def run_baseline(trajs: dict, tfr=None):
    """
    Step through all trajectories, recording:
      - Collision events  (h < H_SEP_NM AND v < V_SEP_FT)
      - TFR proximity     (distance to TFR edge per aircraft per step)

    Returns
    -------
    timeline_df    — per-step active/conflict/tfr_warning counts
    conflict_df    — collision event rows
    summary_df     — per-aircraft aggregate
    tfr_prox_df    — per-aircraft per-step TFR distance (empty if no TFR)
    """
    callsigns = list(trajs.keys())
    max_step  = max(len(df) for df in trajs.values())

    timeline_rows  = []
    conflict_rows  = []
    tfr_prox_rows  = []

    print(f"  Running {max_step} steps across {len(callsigns)} aircraft …")

    for step in range(max_step):
        active = {
            cs: trajs[cs].iloc[step]
            for cs in callsigns if step < len(trajs[cs])
        }
        ac_list = list(active.keys())
        n_conf  = 0
        n_tfr_warn = 0

        # ── Collision detection ───────────────────────────────────────────
        for i in range(len(ac_list)):
            for j in range(i + 1, len(ac_list)):
                cs_a, cs_b   = ac_list[i], ac_list[j]
                row_a, row_b = active[cs_a], active[cs_b]
                h = haversine_nm(float(row_a["lat"]), float(row_a["lon"]),
                                  float(row_b["lat"]), float(row_b["lon"]))
                v = abs(float(row_a["Altitude"]) - float(row_b["Altitude"]))
                if h < H_SEP_NM and v < V_SEP_FT:
                    n_conf += 1
                    conflict_rows.append({
                        "step": step, "cs_a": cs_a, "cs_b": cs_b,
                        "h_nm": round(h, 4), "v_ft": round(v, 1),
                        "lat_a": float(row_a["lat"]), "lon_a": float(row_a["lon"]),
                        "alt_a": float(row_a["Altitude"]),
                        "lat_b": float(row_b["lat"]), "lon_b": float(row_b["lon"]),
                        "alt_b": float(row_b["Altitude"]),
                        "conflict_lat": (float(row_a["lat"]) + float(row_b["lat"])) / 2,
                        "conflict_lon": (float(row_a["lon"]) + float(row_b["lon"])) / 2,
                    })

        # ── TFR proximity ─────────────────────────────────────────────────
        if tfr is not None:
            for cs, row in active.items():
                dist = tfr.distance_to_edge(float(row["lat"]), float(row["lon"]))
                in_tfr = tfr.contains(float(row["lat"]), float(row["lon"]))
                if dist < TFR_WARNING_NM:
                    n_tfr_warn += 1
                tfr_prox_rows.append({
                    "step": step, "callsign": cs,
                    "tfr_dist_nm": round(dist, 3),
                    "in_tfr": in_tfr,
                })

        timeline_rows.append({
            "step"       : step,
            "n_active"   : len(active),
            "n_conflicts": n_conf,
            "n_tfr_warn" : n_tfr_warn,
        })

    timeline_df = pd.DataFrame(timeline_rows)
    conflict_df = pd.DataFrame(conflict_rows) if conflict_rows else pd.DataFrame()
    tfr_prox_df = pd.DataFrame(tfr_prox_rows) if tfr_prox_rows else pd.DataFrame()

    # ── Per-aircraft summary ──────────────────────────────────────────────
    summary_rows = []
    for cs, df in trajs.items():
        coll_inv = 0
        if not conflict_df.empty:
            coll_inv = len(conflict_df[
                (conflict_df["cs_a"] == cs) | (conflict_df["cs_b"] == cs)
            ])

        tfr_entries  = 0
        min_tfr_dist = float("inf")
        steps_in_tfr = 0
        if not tfr_prox_df.empty:
            ag = tfr_prox_df[tfr_prox_df["callsign"] == cs]
            if not ag.empty:
                min_tfr_dist = float(ag["tfr_dist_nm"].min())
                steps_in_tfr = int(ag["in_tfr"].sum())
                in_vals = ag["in_tfr"].tolist()
                tfr_entries = sum(
                    1 for i in range(1, len(in_vals))
                    if in_vals[i] and not in_vals[i - 1]
                )

        summary_rows.append({
            "callsign"       : cs,
            "n_steps"        : len(df),
            "alt_mean_ft"    : round(df["Altitude"].mean(), 0),
            "alt_max_ft"     : round(df["Altitude"].max(), 0),
            "speed_mean_kt"  : round(df["Speed"].mean(), 0),
            "conflict_steps" : coll_inv,
            "in_conflict"    : coll_inv > 0,
            "steps_in_tfr"   : steps_in_tfr,
            "tfr_entries"    : tfr_entries,
            "min_tfr_dist_nm": round(min_tfr_dist, 3) if min_tfr_dist < 1e6 else None,
        })

    summary_df = pd.DataFrame(summary_rows)
    print(f"  Collision steps : {len(conflict_df)}")
    print(f"  TFR prox rows   : {len(tfr_prox_df)}")

    return timeline_df, conflict_df, summary_df, tfr_prox_df


# =============================================================================
# Pairwise separation matrix
# =============================================================================

def pairwise_min_sep(trajs: dict) -> pd.DataFrame:
    callsigns = list(trajs.keys())
    n  = len(callsigns)
    mat = pd.DataFrame(np.nan, index=callsigns, columns=callsigns)
    for i in range(n):
        for j in range(i + 1, n):
            cs_a, cs_b = callsigns[i], callsigns[j]
            min_steps  = min(len(trajs[cs_a]), len(trajs[cs_b]))
            min_h = min(
                haversine_nm(
                    float(trajs[cs_a]["lat"].iloc[s]),
                    float(trajs[cs_a]["lon"].iloc[s]),
                    float(trajs[cs_b]["lat"].iloc[s]),
                    float(trajs[cs_b]["lon"].iloc[s]),
                )
                for s in range(min_steps)
            )
            mat.loc[cs_a, cs_b] = round(min_h, 1)
            mat.loc[cs_b, cs_a] = round(min_h, 1)
    arr = mat.values.copy()
    np.fill_diagonal(arr, 0)
    return pd.DataFrame(arr, index=mat.index, columns=mat.columns)


# =============================================================================
# Visualisation 1 — Folium map  (extended with TFR overlay)
# =============================================================================

def make_map(trajs: dict, conflict_df: pd.DataFrame,
             tfr=None, out_path: str = ""):
    fmap = folium.Map(location=[22, -82], zoom_start=5,
                      tiles="CartoDB dark_matter")

    conflict_cs = set()
    if not conflict_df.empty:
        conflict_cs = set(conflict_df["cs_a"]) | set(conflict_df["cs_b"])

    colors = {
        cs: PALETTE["conflict"] if cs in conflict_cs else PALETTE["safe"]
        for cs in trajs
    }

    # ── Flight trajectories ───────────────────────────────────────────────
    for cs, df in trajs.items():
        coords = list(zip(df["lat"], df["lon"]))
        color  = colors[cs]
        folium.PolyLine(
            coords, color=color,
            weight=3 if cs in conflict_cs else 1.5,
            opacity=0.9 if cs in conflict_cs else 0.5,
            tooltip=cs,
        ).add_to(fmap)
        folium.CircleMarker(
            location=coords[0], radius=4,
            color=color, fill=True, fill_opacity=0.9,
            tooltip=f"{cs} — origin",
        ).add_to(fmap)
        folium.Marker(
            location=coords[-1],
            icon=folium.DivIcon(
                html=(f'<div style="font-size:10px;color:{color};'
                      f'font-weight:bold;white-space:nowrap">{cs}</div>'),
                icon_size=(80, 16),
            ),
        ).add_to(fmap)

    # ── Collision markers ─────────────────────────────────────────────────
    if not conflict_df.empty:
        for _, row in conflict_df.drop_duplicates(
                subset=["cs_a", "cs_b", "step"]).iterrows():
            folium.CircleMarker(
                location=[row["conflict_lat"], row["conflict_lon"]],
                radius=8, color="#ff2222", fill=True,
                fill_color="#ff2222", fill_opacity=0.7,
                tooltip=(f"CONFLICT: {row['cs_a']} ↔ {row['cs_b']}<br>"
                         f"Step {row['step']}<br>"
                         f"H: {row['h_nm']:.2f} NM  V: {row['v_ft']:.0f} ft"),
            ).add_to(fmap)

    # ── TFR polygon overlay (Group A) ────────────────────────────────────
    if tfr is not None:
        tfr_coords = list(zip(tfr.lats.tolist(), tfr.lons.tolist()))
        # Close the polygon
        tfr_coords.append(tfr_coords[0])

        # Filled polygon
        folium.Polygon(
            locations=tfr_coords,
            color=PALETTE["tfr"],
            weight=2.5,
            fill=True,
            fill_color=PALETTE["tfr_fill"],
            fill_opacity=0.25,
            tooltip=(f"TFR Zone — {tfr.n_vertices} vertices<br>"
                     f"Centroid: ({tfr.centroid_lat:.3f}, {tfr.centroid_lon:.3f})"),
        ).add_to(fmap)

        # Warning buffer ring (10 nm)
        folium.CircleMarker(
            location=[tfr.centroid_lat, tfr.centroid_lon],
            radius=3, color=PALETTE["tfr"], fill=True,
            fill_opacity=1.0, tooltip="TFR Centroid",
        ).add_to(fmap)

        # Vertex markers
        for i, (lat, lon) in enumerate(zip(tfr.lats, tfr.lons)):
            folium.CircleMarker(
                location=[lat, lon], radius=3,
                color=PALETTE["tfr"], fill=True, fill_opacity=0.8,
                tooltip=f"TFR vertex {i}",
            ).add_to(fmap)

    # ── Legend ────────────────────────────────────────────────────────────
    tfr_legend = (
        '<span style="color:#9b30ff">&#9644;</span> TFR boundary<br>'
        if tfr else ""
    )
    legend_html = f"""
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
                background:rgba(20,20,20,0.85);padding:12px 16px;
                border-radius:8px;color:#eee;font-size:12px;line-height:1.8">
      <b>Legend</b><br>
      <span style="color:#e05c2e">&#9644;</span> Conflict aircraft<br>
      <span style="color:#3a7bd5">&#9644;</span> Other aircraft<br>
      <span style="color:#ff2222">&#11044;</span> Conflict location<br>
      {tfr_legend}
      <span style="color:#888">&#11044;</span> Origin point
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(legend_html))
    fmap.save(out_path)
    print(f"  Map saved → {out_path}")


# =============================================================================
# Visualisation 2–7 — Static figure  (5 original + 1 new TFR panel)
# =============================================================================

def make_figures(trajs, timeline_df, conflict_df,
                 summary_df, sep_matrix, tfr_prox_df,
                 tfr=None, out_path: str = ""):

    plt.style.use("seaborn-v0_8-whitegrid")
    has_tfr = tfr is not None and not tfr_prox_df.empty

    # 3×2 grid — same as original; 6th panel is new TFR panel
    fig = plt.figure(figsize=(18, 24))
    fig.patch.set_facecolor("white")
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3)

    callsigns   = list(trajs.keys())
    conflict_cs = set(summary_df[summary_df["in_conflict"]]["callsign"])

    # ── Panel 1: conflict frequency + active aircraft ─────────────────────
    ax1   = fig.add_subplot(gs[0, 0])
    ax1_r = ax1.twinx()
    steps  = timeline_df["step"]
    n_conf = timeline_df["n_conflicts"]
    n_ac   = timeline_df["n_active"]
    ax1.fill_between(steps, n_conf, alpha=0.35, color=PALETTE["conflict"])
    ax1.plot(steps, n_conf, color=PALETTE["conflict"], lw=1.5,
             label="Active conflicts")
    ax1_r.plot(steps, n_ac, color=PALETTE["safe"], lw=1.2, alpha=0.6,
               linestyle="--", label="Active aircraft")
    ax1.set_xlabel("Simulation step", fontsize=10)
    ax1.set_ylabel("Active conflicts", color=PALETTE["conflict"], fontsize=10)
    ax1_r.set_ylabel("Active aircraft", color=PALETTE["safe"], fontsize=10)
    ax1.set_title("Conflict frequency vs active aircraft", fontsize=11, fontweight="bold")
    ax1.set_ylim(bottom=0)
    ax1_r.set_ylim(0, 15)
    l1, lb1 = ax1.get_legend_handles_labels()
    l2, lb2 = ax1_r.get_legend_handles_labels()
    ax1.legend(l1 + l2, lb1 + lb2, fontsize=8, loc="upper right")

    # ── Panel 2: horizontal separation for top conflict pair ──────────────
    ax2 = fig.add_subplot(gs[0, 1])
    if not conflict_df.empty:
        pair_a = conflict_df["cs_a"].iloc[0]
        pair_b = conflict_df["cs_b"].iloc[0]
        df_a   = trajs[pair_a]
        df_b   = trajs[pair_b]
        max_s  = min(len(df_a), len(df_b), 80)
        sep_steps = list(range(max_s))
        h_seps = [
            haversine_nm(float(df_a["lat"].iloc[s]), float(df_a["lon"].iloc[s]),
                         float(df_b["lat"].iloc[s]), float(df_b["lon"].iloc[s]))
            for s in sep_steps
        ]
        ax2.fill_between(sep_steps, h_seps, alpha=0.2, color=PALETTE["conflict"])
        ax2.plot(sep_steps, h_seps, color=PALETTE["conflict"], lw=2,
                 label=f"{pair_a} ↔ {pair_b}")
        ax2.axhline(H_SEP_NM, color=PALETTE["threshold"], lw=1.5,
                    linestyle="--", label=f"{H_SEP_NM} NM threshold")
        ax2.fill_between(sep_steps, 0, H_SEP_NM, alpha=0.06, color=PALETTE["conflict"])
        ax2.set_xlabel("Simulation step", fontsize=10)
        ax2.set_ylabel("Horizontal separation (NM)", fontsize=10)
        ax2.set_title(f"H-separation: {pair_a} ↔ {pair_b}", fontsize=11, fontweight="bold")
        ax2.legend(fontsize=8)
        ax2.set_ylim(bottom=0)

    # ── Panel 3: pairwise min-separation heatmap ──────────────────────────
    ax3  = fig.add_subplot(gs[1, :])
    mask = np.eye(len(callsigns), dtype=bool)
    sns.heatmap(
        sep_matrix.astype(float),
        ax=ax3, annot=True, fmt=".1f", cmap="RdYlGn",
        vmin=0, vmax=50, mask=mask,
        linewidths=0.5, linecolor="#ddd",
        cbar_kws={"label": "Min horizontal separation (NM)", "shrink": 0.6},
        annot_kws={"size": 8},
    )
    ax3.set_title(
        "Pairwise minimum horizontal separation matrix (NM)\n"
        "Red = within conflict threshold  |  Green = well separated",
        fontsize=11, fontweight="bold"
    )
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    ax3.set_yticklabels(ax3.get_yticklabels(), rotation=0, fontsize=8)

    # ── Panel 4: per-aircraft mean altitude ───────────────────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    bar_colors = [
        PALETTE["conflict"] if cs in conflict_cs else PALETTE["safe"]
        for cs in summary_df["callsign"]
    ]
    bars = ax4.barh(summary_df["callsign"], summary_df["alt_mean_ft"],
                    color=bar_colors, alpha=0.75,
                    edgecolor="white", linewidth=0.5)
    ax4.set_xlabel("Mean altitude (ft)", fontsize=10)
    ax4.set_title("Per-aircraft mean altitude\n(red = conflict-involved)",
                  fontsize=11, fontweight="bold")
    ax4.set_xlim(0, 42_000)
    ax4.invert_yaxis()
    for bar, row in zip(bars, summary_df.itertuples()):
        ax4.text(bar.get_width() + 200,
                 bar.get_y() + bar.get_height() / 2,
                 f"{int(row.alt_mean_ft):,} ft",
                 va="center", fontsize=8, color="#444")
    ax4.legend(handles=[
        mpatches.Patch(color=PALETTE["conflict"], label="In conflict"),
        mpatches.Patch(color=PALETTE["safe"],     label="No conflict"),
    ], fontsize=8, loc="lower right")

    # ── Panel 5 / 6: TFR proximity (new) OR altitude profile (fallback) ───
    ax5 = fig.add_subplot(gs[2, 1])

    if has_tfr:
        # Panel 5 — TFR proximity over time for each aircraft
        pivot = tfr_prox_df.pivot(index="step", columns="callsign",
                                   values="tfr_dist_nm")
        for cs in pivot.columns:
            color = PALETTE["conflict"] if cs in conflict_cs else PALETTE["safe"]
            ax5.plot(pivot.index, pivot[cs], lw=1.2, alpha=0.7,
                     color=color, label=cs)

        ax5.axhline(0, color=PALETTE["tfr"], lw=2,
                    linestyle="-", label="TFR boundary")
        ax5.axhline(TFR_WARNING_NM, color=PALETTE["tfr"], lw=1.2,
                    linestyle="--", alpha=0.6,
                    label=f"{TFR_WARNING_NM} NM warning")
        ax5.fill_between(pivot.index, 0, TFR_WARNING_NM,
                          alpha=0.07, color=PALETTE["tfr"])

        ax5.set_xlabel("Simulation step", fontsize=10)
        ax5.set_ylabel("Distance to TFR edge (NM)", fontsize=10)
        ax5.set_title(
            "TFR proximity — all aircraft\n"
            "(negative = inside TFR, purple band = warning zone)",
            fontsize=11, fontweight="bold"
        )
        # Keep legend readable — show at most 6 entries
        handles, labels = ax5.get_legend_handles_labels()
        ax5.legend(handles[:8], labels[:8], fontsize=7, loc="upper right")

    elif not conflict_df.empty:
        # Fallback: altitude profile of conflict pair
        pair_a = conflict_df["cs_a"].iloc[0]
        pair_b = conflict_df["cs_b"].iloc[0]
        conf_steps = set(conflict_df["step"])
        max_s = min(len(trajs[pair_a]), len(trajs[pair_b]), 80)
        sr = list(range(max_s))
        alt_a = [float(trajs[pair_a]["Altitude"].iloc[s]) for s in sr]
        alt_b = [float(trajs[pair_b]["Altitude"].iloc[s]) for s in sr]
        ax5.plot(sr, alt_a, color=PALETTE["conflict"], lw=2, label=pair_a)
        ax5.plot(sr, alt_b, color=PALETTE["warning"], lw=2,
                 label=pair_b, linestyle="--")
        for s in sr:
            if s in conf_steps:
                ax5.axvspan(s - 0.5, s + 0.5, alpha=0.15,
                             color=PALETTE["conflict"])
        ax5.set_xlabel("Simulation step", fontsize=10)
        ax5.set_ylabel("Altitude (ft)", fontsize=10)
        ax5.set_title(f"Altitude profiles: {pair_a} vs {pair_b}\n"
                       "(shaded = conflict steps)",
                       fontsize=11, fontweight="bold")
        ax5.legend(fontsize=8)
        ax5.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    # ── Super-title ───────────────────────────────────────────────────────
    tfr_tag = " + TFR Overlay" if has_tfr else ""
    fig.suptitle(
        f"Baseline Conflict Analysis{tfr_tag} — {len(trajs)} FlightRadar24 Aircraft\n"
        "Group A + B  |  MDP Air Traffic Simulation  |  ITSEC 2026",
        fontsize=13, fontweight="bold", y=0.98
    )

    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"  Figure saved → {out_path}")


# =============================================================================
# Combined safety timeline CSV
# =============================================================================

def build_combined_safety_timeline(timeline_df, tfr_prox_df) -> pd.DataFrame:
    """
    Merge collision timeline with per-step TFR warning counts.
    Adds a combined 'any_safety_event' column for easy filtering.
    """
    df = timeline_df.copy()
    if not tfr_prox_df.empty:
        tfr_warn = (
            tfr_prox_df[tfr_prox_df["tfr_dist_nm"] < TFR_WARNING_NM]
            .groupby("step")
            .size()
            .reset_index(name="n_tfr_warn_aircraft")
        )
        tfr_breach = (
            tfr_prox_df[tfr_prox_df["in_tfr"]]
            .groupby("step")
            .size()
            .reset_index(name="n_in_tfr")
        )
        df = df.merge(tfr_warn,   on="step", how="left")
        df = df.merge(tfr_breach, on="step", how="left")
        df["n_tfr_warn_aircraft"] = df["n_tfr_warn_aircraft"].fillna(0).astype(int)
        df["n_in_tfr"]            = df["n_in_tfr"].fillna(0).astype(int)
    else:
        df["n_tfr_warn_aircraft"] = 0
        df["n_in_tfr"]            = 0

    df["any_safety_event"] = (
        (df["n_conflicts"] > 0) | (df["n_in_tfr"] > 0)
    )
    return df


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Combined baseline analysis — TFR + Collision")
    parser.add_argument("--data", default=DATA_DIR,
                        help="FlightRadar24 CSV folder")
    parser.add_argument("--tfr", default=TFR_PATH_DEFAULT,
                        help="TFR_Lat_Lon.xlsx path")
    parser.add_argument("--out", default=SCRIPT_DIR,
                        help="Output directory (default: same as script)")
    args = parser.parse_args()

    print("\n" + "=" * 62)
    print("  Combined Baseline Analysis — FlightRadar24 + TFR")
    print("=" * 62)

    trajs = load_trajectories(args.data)
    tfr   = load_tfr(args.tfr)

    print("\n  Running simulation …")
    timeline_df, conflict_df, summary_df, tfr_prox_df = run_baseline(trajs, tfr)

    print("\n  Computing pairwise separation matrix …")
    sep_matrix = pairwise_min_sep(trajs)

    combined_timeline = build_combined_safety_timeline(timeline_df, tfr_prox_df)

    # ── Console summary ───────────────────────────────────────────────────
    print("\n── Summary ──────────────────────────────────────────────────")
    print(f"  Total aircraft     : {len(trajs)}")
    print(f"  Simulation steps   : {timeline_df['step'].max() + 1}")
    print(f"  Collision steps    : {len(conflict_df)}")
    print(f"  Peak simultaneous  : {timeline_df['n_conflicts'].max()}")
    if not conflict_df.empty:
        for pair, grp in conflict_df.groupby(["cs_a", "cs_b"]):
            print(f"  {pair[0]} ↔ {pair[1]}: {len(grp)} steps, "
                  f"min h-sep={grp['h_nm'].min():.3f} NM, "
                  f"steps {grp['step'].min()}–{grp['step'].max()}")
    if tfr is not None and not tfr_prox_df.empty:
        n_in = int(tfr_prox_df["in_tfr"].sum())
        n_warn = int((tfr_prox_df["tfr_dist_nm"] < TFR_WARNING_NM).sum())
        print(f"\n  TFR breach steps   : {n_in}")
        print(f"  TFR warning steps  : {n_warn}")
        closest = tfr_prox_df.loc[tfr_prox_df["tfr_dist_nm"].idxmin()]
        print(f"  Closest approach   : {closest['callsign']} "
              f"at {closest['tfr_dist_nm']:.3f} NM (step {closest['step']})")

    print("\n── Per-aircraft summary ─────────────────────────────────────")
    cols = ["callsign", "n_steps", "alt_mean_ft", "speed_mean_kt",
            "conflict_steps", "steps_in_tfr", "min_tfr_dist_nm"]
    print(summary_df[cols].to_string(index=False))

    # ── Save outputs ──────────────────────────────────────────────────────
    out = args.out
    os.makedirs(out, exist_ok=True)
    print("\n  Saving outputs …")

    map_path = os.path.join(out, "baseline_map_combined.html")
    fig_path = os.path.join(out, "baseline_metrics_combined.png")

    make_map(trajs, conflict_df, tfr=tfr, out_path=map_path)
    make_figures(trajs, timeline_df, conflict_df, summary_df,
                 sep_matrix, tfr_prox_df, tfr=tfr, out_path=fig_path)

    conflict_df.to_csv(os.path.join(out, "baseline_conflicts.csv"), index=False)
    summary_df.to_csv(os.path.join(out, "baseline_summary.csv"), index=False)
    combined_timeline.to_csv(
        os.path.join(out, "baseline_combined_safety.csv"), index=False)

    if not tfr_prox_df.empty:
        tfr_prox_df.to_csv(
            os.path.join(out, "baseline_tfr_metrics.csv"), index=False)

    print("\n── Files written ────────────────────────────────────────────")
    for fname in [
        "baseline_map_combined.html",
        "baseline_metrics_combined.png",
        "baseline_conflicts.csv",
        "baseline_summary.csv",
        "baseline_combined_safety.csv",
        "baseline_tfr_metrics.csv",
    ]:
        path = os.path.join(out, fname)
        if os.path.isfile(path):
            size = os.path.getsize(path) // 1024
            print(f"  {fname:<42} {size:>5} KB")

    print("\n✓ Done. Open baseline_map_combined.html for the interactive map.\n")


if __name__ == "__main__":
    main()
