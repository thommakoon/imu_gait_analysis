"""
Plot continuous foot side-view trajectory (LF / RF) together with head
acceleration and gaze position on a shared time axis.

The side-view panel shows one connected path through the full recording
(distance vs height), similar to an unrolled stride sequence.
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


#### PARAMS ####
SUBJECT = "imu_thom_2026_06_06"
RUN = "visit5km"
DATASET = "data_charite"
MARK_GAIT_EVENTS = True
ONLY_VALID_STRIDES = False
SAVE_STANDALONE_SIDEVIEW = True
INTERACTIVE = False  # matplotlib windows (use plot_trajectory_head_gaze_interactive.py for browser)
SAVE_FIGURES = False  # set True to also write PNG files to disk
#### PARAMS END ####


def _finalize_figure(fig, output_path):
    if SAVE_FIGURES and output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved {output_path}")
    if not INTERACTIVE:
        plt.close(fig)


def load_head_gaze(raw_imu_dir):
    head = pd.read_csv(os.path.join(raw_imu_dir, "head_200hz.csv"))
    gaze = pd.read_csv(os.path.join(raw_imu_dir, "gaze_200hz.csv"))

    t0 = head["t_utc_ns"].iloc[0]
    time_s = (head["t_utc_ns"] - t0) / 1e9

    head_df = pd.DataFrame(
        {
            "time": time_s,
            "roll": head["roll [deg]"],
            "pitch": head["pitch [deg]"],
            "yaw": head["yaw [deg]"],
        }
    )
    gaze_df = pd.DataFrame(
        {
            "time": time_s,
            "gaze_x": gaze["gaze x [px]"],
            "gaze_y": gaze["gaze y [px]"],
        }
    )
    return head_df, gaze_df


def load_foot_data(data_base_path, subject, run, side):
    foot = "left" if side == "LF" else "right"
    interim_dir = os.path.join(data_base_path, "interim", subject, run)
    processed_dir = os.path.join(data_base_path, "processed", subject, run)

    trajectory = pd.read_json(
        os.path.join(interim_dir, f"_trajectory_estimation_{foot}.json")
    )
    gait_params = pd.read_csv(
        os.path.join(processed_dir, f"{foot}_foot_core_params.csv")
    )
    if ONLY_VALID_STRIDES:
        gait_params = gait_params[gait_params["is_outlier"] == False]

    ic_df = pd.read_csv(
        os.path.join(data_base_path, "interim", "imu_initial_contact_manual.csv")
    )
    ic_row = ic_df[(ic_df["subject"] == subject) & (ic_df["run"] == run)].iloc[0]
    initial_ic = float(
        ic_row["imu_initial_contact_left" if side == "LF" else "imu_initial_contact_right"]
    )

    return trajectory, gait_params, initial_ic


def _project_segment_sideview(segment):
    """Project one trajectory segment to local side view (same method as FootTrajectoryPlot)."""
    px = segment["position_x"].values
    py = segment["position_y"].values
    pz = segment["position_z"].values

    line_vec = np.array([px[-1] - px[0], py[-1] - py[0]])
    line_norm = np.linalg.norm(line_vec)
    line_vec_norm = np.array([1.0, 0.0]) if line_norm < 1e-6 else line_vec / line_norm

    horiz = np.array(
        [
            np.dot(np.array([px[i] - px[0], py[i] - py[0]]), line_vec_norm)
            for i in range(len(segment))
        ]
    )
    height = pz - pz.min()
    return horiz, height


def stitch_continuous_sideview(trajectory, gait_params, initial_ic):
    """
    Build one continuous side-view path by stitching stride segments end-to-end.
    Each stride runs from one heel strike (IC) to the next.
    """
    ic_times = [initial_ic] + gait_params["ic_time"].tolist()
    ic_times = sorted(set(ic_times))
    ic_times.append(float(trajectory["time"].iloc[-1]))

    horiz_parts = []
    height_parts = []
    ic_points = []
    fo_points = []
    offset = 0.0

    for stride_idx in range(len(ic_times) - 1):
        t_start, t_end = ic_times[stride_idx], ic_times[stride_idx + 1]
        segment = trajectory[
            (trajectory["time"] >= t_start) & (trajectory["time"] <= t_end)
        ]
        if len(segment) < 2:
            continue

        horiz_local, height_local = _project_segment_sideview(segment)
        horiz_parts.append(horiz_local + offset)
        height_parts.append(height_local)
        ic_points.append((offset, height_local[0]))

        fo_in_stride = gait_params[
            (gait_params["fo_time"] >= t_start) & (gait_params["fo_time"] <= t_end)
        ]["fo_time"]
        for fo_time in fo_in_stride:
            rel_idx = int(np.argmin(np.abs(segment["time"].values - fo_time)))
            fo_points.append(
                (offset + horiz_local[rel_idx], height_local[rel_idx])
            )

        offset += horiz_local[-1]

    return (
        np.concatenate(horiz_parts),
        np.concatenate(height_parts),
        np.array(ic_points),
        np.array(fo_points) if fo_points else np.empty((0, 2)),
    )


def add_gait_event_lines(ax, gait_params, labels_added):
    for event_col, color, label in [
        ("ic_time", "red", "Heel strike (IC)"),
        ("fo_time", "purple", "Toe off (FO)"),
    ]:
        for t in gait_params[event_col]:
            ax.axvline(
                t,
                color=color,
                alpha=0.15,
                linewidth=0.8,
                label=label if label not in labels_added else None,
            )
        labels_added.add(label)


def plot_continuous_sideview(
    side,
    horiz,
    height,
    ic_points,
    fo_points,
    output_path,
    subject,
    run,
    duration_s,
):
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.plot(horiz, height, color="black", linewidth=1.0)

    if MARK_GAIT_EVENTS and len(ic_points):
        ax.scatter(
            ic_points[:, 0],
            ic_points[:, 1],
            color="red",
            s=14,
            zorder=5,
            label="Heel strike (IC)",
        )
    if MARK_GAIT_EVENTS and len(fo_points):
        ax.scatter(
            fo_points[:, 0],
            fo_points[:, 1],
            color="purple",
            s=14,
            zorder=5,
            label="Toe off (FO)",
        )

    ax.set_xlabel("Distance along walking direction (m)")
    ax.set_ylabel("Foot height (m)")
    ax.set_title(
        f"{subject} | {run} | {side} — continuous side-view trajectory "
        f"({duration_s:.0f} s, {horiz[-1]:.1f} m stitched)"
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _finalize_figure(fig, output_path)
    return fig


def plot_combined(
    side,
    trajectory,
    gait_params,
    head_df,
    gaze_df,
    horiz,
    height,
    ic_points,
    fo_points,
    output_path,
    subject,
    run,
):
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(4, 1, height_ratios=[1.2, 1, 1, 1], hspace=0.08)

    # --- Continuous side view (distance vs height) ---
    ax_side = fig.add_subplot(gs[0])
    ax_side.plot(horiz, height, color="black", linewidth=1.0)
    if MARK_GAIT_EVENTS and len(ic_points):
        ax_side.scatter(
            ic_points[:, 0], ic_points[:, 1], color="red", s=14, zorder=5, label="IC"
        )
    if MARK_GAIT_EVENTS and len(fo_points):
        ax_side.scatter(
            fo_points[:, 0], fo_points[:, 1], color="purple", s=14, zorder=5, label="FO"
        )
    ax_side.set_ylabel("Height (m)")
    ax_side.set_title(
        f"{subject} | {run} | {side} — continuous side view + head/gaze vs time"
    )
    ax_side.legend(loc="upper right", fontsize=8)
    ax_side.grid(True, alpha=0.3)

    # Time-based panels share x-axis
    ax_head = fig.add_subplot(gs[1], sharex=None)
    ax_gaze = fig.add_subplot(gs[2], sharex=ax_head)
    ax_mag = fig.add_subplot(gs[3], sharex=ax_head)
    labels_added = set()

    ax_head.plot(head_df["time"], head_df["roll"], linewidth=0.7, label="roll")
    ax_head.plot(head_df["time"], head_df["pitch"], linewidth=0.7, label="pitch")
    ax_head.plot(head_df["time"], head_df["yaw"], linewidth=0.7, label="yaw")
    ax_head.set_ylabel("Head orientation (deg)")
    ax_head.legend(loc="upper right", fontsize=8)
    ax_head.grid(True, alpha=0.3)
    if MARK_GAIT_EVENTS:
        add_gait_event_lines(ax_head, gait_params, labels_added)

    ax_gaze.plot(gaze_df["time"], gaze_df["gaze_x"], linewidth=0.7, label="gaze x")
    ax_gaze.plot(gaze_df["time"], gaze_df["gaze_y"], linewidth=0.7, label="gaze y")
    ax_gaze.set_ylabel("Gaze (px)")
    ax_gaze.legend(loc="upper right", fontsize=8)
    ax_gaze.grid(True, alpha=0.3)
    if MARK_GAIT_EVENTS:
        add_gait_event_lines(ax_gaze, gait_params, labels_added)

    head_center = head_df[["roll", "pitch", "yaw"]].values.mean(axis=0)
    head_orient_dev = np.linalg.norm(
        head_df[["roll", "pitch", "yaw"]].values - head_center, axis=1
    )
    gaze_center = gaze_df[["gaze_x", "gaze_y"]].values.mean(axis=0)
    gaze_mag = np.linalg.norm(
        gaze_df[["gaze_x", "gaze_y"]].values - gaze_center, axis=1
    )
    ax_mag.plot(head_df["time"], head_orient_dev, linewidth=0.8, label="head orient. deviation")
    ax_mag.plot(gaze_df["time"], gaze_mag, linewidth=0.8, label="gaze deviation")
    ax_mag.set_xlabel("Time (s)")
    ax_mag.set_ylabel("Magnitude")
    ax_mag.legend(loc="upper right", fontsize=8)
    ax_mag.grid(True, alpha=0.3)
    if MARK_GAIT_EVENTS:
        add_gait_event_lines(ax_mag, gait_params, labels_added)

    # Link side-view x-axis label to cumulative distance; annotate time span
    ax_side.set_xlabel(
        f"Distance along walking direction (m)  |  duration = {trajectory['time'].iloc[-1]:.1f} s"
    )

    fig.subplots_adjust(hspace=0.35)
    _finalize_figure(fig, output_path)
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
    figures = []

    for side in ["LF", "RF"]:
        trajectory, gait_params, initial_ic = load_foot_data(
            data_base_path, SUBJECT, RUN, side
        )
        horiz, height, ic_points, fo_points = stitch_continuous_sideview(
            trajectory, gait_params, initial_ic
        )
        duration_s = float(trajectory["time"].iloc[-1])

        combined_path = (
            os.path.join(output_dir, f"trajectory_head_gaze_{RUN}_{side}.png")
            if SAVE_FIGURES
            else None
        )
        figures.append(
            plot_combined(
                side,
                trajectory,
                gait_params,
                head_df,
                gaze_df,
                horiz,
                height,
                ic_points,
                fo_points,
                combined_path,
                SUBJECT,
                RUN,
            )
        )

        if SAVE_STANDALONE_SIDEVIEW:
            sideview_path = (
                os.path.join(
                    output_dir, f"trajectory_sideview_continuous_{RUN}_{side}.png"
                )
                if SAVE_FIGURES
                else None
            )
            figures.append(
                plot_continuous_sideview(
                    side,
                    horiz,
                    height,
                    ic_points,
                    fo_points,
                    sideview_path,
                    SUBJECT,
                    RUN,
                    duration_s,
                )
            )

    if INTERACTIVE:
        print(
            "Interactive windows opened. Use the toolbar to zoom/pan; "
            "close all windows to exit."
        )
        plt.show()


if __name__ == "__main__":
    main()
