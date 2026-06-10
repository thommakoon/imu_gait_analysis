"""
Interactive browser dashboard: foot trajectory, head orientation, and gaze
on linked time axes (zoom/pan one time plot → all time plots follow).

Opens an HTML file in your browser. The continuous side-view (distance vs
height) is shown above the time panels and has its own x-axis.
"""

import json
import os
import webbrowser

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from plot_trajectory_head_gaze import (
    DATASET,
    MARK_GAIT_EVENTS,
    RUN,
    SUBJECT,
    load_foot_data,
    load_head_gaze,
    stitch_continuous_sideview,
)


#### PARAMS ####
OPEN_IN_BROWSER = True
SAVE_HTML = True
#### PARAMS END ####

TIME_ROWS = (2, 3, 4, 5)  # subplot rows that share the time x-axis


def _foot_height(trajectory):
    pz = trajectory["position_z"].values
    return pz - pz.min()


def _add_gait_markers(fig, gait_params, row, y_anchor, ic_label, fo_label):
    """Lightweight IC/FO markers (line-ns) that follow linked time zoom."""
    if not MARK_GAIT_EVENTS:
        return
    ic_t = gait_params["ic_time"].values
    fo_t = gait_params["fo_time"].values
    fig.add_trace(
        go.Scatter(
            x=ic_t,
            y=[y_anchor] * len(ic_t),
            mode="markers",
            name=ic_label,
            legendgroup="ic",
            showlegend=(row == TIME_ROWS[0]),
            marker=dict(color="red", size=8, symbol="line-ns-open"),
            hovertemplate="IC t=%{x:.3f} s<extra></extra>",
        ),
        row=row,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=fo_t,
            y=[y_anchor] * len(fo_t),
            mode="markers",
            name=fo_label,
            legendgroup="fo",
            showlegend=(row == TIME_ROWS[0]),
            marker=dict(color="purple", size=8, symbol="line-ns-open"),
            hovertemplate="FO t=%{x:.3f} s<extra></extra>",
        ),
        row=row,
        col=1,
    )


def build_figure(
    side,
    trajectory,
    gait_params,
    head_df,
    gaze_df,
    horiz,
    height,
    ic_points,
    fo_points,
    subject,
    run,
):
    time = trajectory["time"].values
    foot_h = _foot_height(trajectory)

    head_center = head_df[["roll", "pitch", "yaw"]].values.mean(axis=0)
    head_orient_dev = np.linalg.norm(
        head_df[["roll", "pitch", "yaw"]].values - head_center, axis=1
    )
    gaze_center = gaze_df[["gaze_x", "gaze_y"]].values.mean(axis=0)
    gaze_mag = np.linalg.norm(
        gaze_df[["gaze_x", "gaze_y"]].values - gaze_center, axis=1
    )

    fig = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.05,
        row_heights=[0.22, 0.2, 0.2, 0.2, 0.18],
        subplot_titles=(
            f"Continuous side view — {side} (distance vs height)",
            f"Foot height vs time — {side}",
            "Head orientation (roll / pitch / yaw)",
            "Gaze position",
            "Head orientation deviation and gaze deviation",
        ),
    )

    # Row 1: stitched side view (independent x-axis)
    fig.add_trace(
        go.Scatter(
            x=horiz,
            y=height,
            mode="lines",
            name="Side view",
            line=dict(color="black", width=1),
            hovertemplate="distance=%{x:.3f} m<br>height=%{y:.3f} m<extra></extra>",
        ),
        row=1,
        col=1,
    )
    if MARK_GAIT_EVENTS and len(ic_points):
        fig.add_trace(
            go.Scatter(
                x=ic_points[:, 0],
                y=ic_points[:, 1],
                mode="markers",
                name="IC",
                marker=dict(color="red", size=6),
                hovertemplate="IC<br>distance=%{x:.3f} m<extra></extra>",
            ),
            row=1,
            col=1,
        )
    if MARK_GAIT_EVENTS and len(fo_points):
        fig.add_trace(
            go.Scatter(
                x=fo_points[:, 0],
                y=fo_points[:, 1],
                mode="markers",
                name="FO",
                marker=dict(color="purple", size=6),
                hovertemplate="FO<br>distance=%{x:.3f} m<extra></extra>",
            ),
            row=1,
            col=1,
        )

    # Rows 2–5: time-based signals (linked zoom on xaxis2)
    fig.add_trace(
        go.Scatter(
            x=time,
            y=foot_h,
            mode="lines",
            name="Foot height",
            line=dict(color="black", width=1),
            hovertemplate="t=%{x:.3f} s<br>height=%{y:.3f} m<extra></extra>",
        ),
        row=2,
        col=1,
    )
    for col_name, color in [
        ("roll", "#1f77b4"),
        ("pitch", "#ff7f0e"),
        ("yaw", "#2ca02c"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=head_df["time"],
                y=head_df[col_name],
                mode="lines",
                name=col_name,
                line=dict(color=color, width=1),
                hovertemplate=f"t=%{{x:.3f}} s<br>{col_name}=%{{y:.2f}} deg<extra></extra>",
            ),
            row=3,
            col=1,
        )
    for col_name, color in [("gaze_x", "#9467bd"), ("gaze_y", "#8c564b")]:
        fig.add_trace(
            go.Scatter(
                x=gaze_df["time"],
                y=gaze_df[col_name],
                mode="lines",
                name=col_name,
                line=dict(color=color, width=1),
                hovertemplate=f"t=%{{x:.3f}} s<br>{col_name}=%{{y:.1f}} px<extra></extra>",
            ),
            row=4,
            col=1,
        )
    fig.add_trace(
        go.Scatter(
            x=head_df["time"],
            y=head_orient_dev,
            mode="lines",
            name="head orient. deviation",
            line=dict(color="#1f77b4", width=1),
        ),
        row=5,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=gaze_df["time"],
            y=gaze_mag,
            mode="lines",
            name="gaze deviation",
            line=dict(color="#9467bd", width=1),
        ),
        row=5,
        col=1,
    )

    y_anchors = {
        2: float(np.min(foot_h)),
        3: float(head_df[["roll", "pitch", "yaw"]].min().min()),
        4: float(gaze_df[["gaze_x", "gaze_y"]].min().min()),
        5: float(min(head_orient_dev.min(), gaze_mag.min())),
    }
    for row in TIME_ROWS:
        _add_gait_markers(
            fig,
            gait_params,
            row,
            y_anchors[row],
            "IC (heel strike)",
            "FO (toe off)",
        )

    # Link time axes: row 2 is master; rows 3–5 follow when zooming
    for row in (3, 4, 5):
        fig.update_xaxes(matches="x2", row=row, col=1)

    fig.update_xaxes(title_text="Distance (m)", row=1, col=1)
    fig.update_yaxes(title_text="Height (m)", row=1, col=1)
    fig.update_yaxes(title_text="Foot height (m)", row=2, col=1)
    fig.update_yaxes(title_text="Head orientation (deg)", row=3, col=1)
    fig.update_yaxes(title_text="Gaze (px)", row=4, col=1)
    fig.update_yaxes(title_text="Magnitude", row=5, col=1)
    fig.update_xaxes(title_text="Time (s)", row=5, col=1)

    duration = time[-1]
    fig.update_layout(
        title=(
            f"{subject} | {run} | {side} — linked time zoom "
            f"(zoom any time panel below; duration {duration:.1f} s)"
        ),
        height=1100,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        template="plotly_white",
    )

    return fig


def main():
    with open(os.path.join(os.path.dirname(__file__), "..", "path.json")) as f:
        paths = json.load(f)
    data_base_path = paths[DATASET]
    raw_imu_dir = os.path.join(data_base_path, "raw", SUBJECT, RUN, "imu")
    output_dir = os.path.join(
        data_base_path,
        "processed",
        "figures_trajectory_head_gaze",
        SUBJECT,
    )
    os.makedirs(output_dir, exist_ok=True)

    head_df, gaze_df = load_head_gaze(raw_imu_dir)
    html_paths = []

    for side in ["LF", "RF"]:
        trajectory, gait_params, initial_ic = load_foot_data(
            data_base_path, SUBJECT, RUN, side
        )
        horiz, height, ic_points, fo_points = stitch_continuous_sideview(
            trajectory, gait_params, initial_ic
        )

        fig = build_figure(
            side,
            trajectory,
            gait_params,
            head_df,
            gaze_df,
            horiz,
            height,
            ic_points,
            fo_points,
            SUBJECT,
            RUN,
        )

        html_path = os.path.join(
            output_dir, f"trajectory_head_gaze_interactive_{RUN}_{side}.html"
        )
        if SAVE_HTML:
            fig.write_html(html_path, include_plotlyjs="cdn")
            print(f"Saved {html_path}")
        html_paths.append(html_path)

    if OPEN_IN_BROWSER and html_paths:
        uri = f"file:///{os.path.abspath(html_paths[0]).replace(chr(92), '/')}"
        webbrowser.open(uri)
        print("Opened first plot in your browser. Open the second HTML file manually if needed.")


if __name__ == "__main__":
    main()
