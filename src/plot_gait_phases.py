"""Plot heel strike (IC) and toe off (FO) events on foot IMU signals."""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_prominences

from data_reader.imu import IMU
from LFRF_parameters.event_detection.imu_event_detection import tunca_gait_events


def auto_detect_initial_contact(imu_path, prominence_threshold=3):
    imu = IMU(imu_path)
    imu.gyro_to_rad()
    time = imu.time()
    accel = np.transpose(imu.accel())
    accel_norm = np.linalg.norm(accel, axis=0)
    peaks, _ = find_peaks(accel_norm)
    prominences = peak_prominences(accel_norm, peaks)[0]
    valid_peaks = peaks[prominences > prominence_threshold]
    return float(time[valid_peaks[0]])


def plot_visit(subject, run, data_base_path, output_dir):
    interim = os.path.join(data_base_path, "interim")
    processed = os.path.join(data_base_path, "processed", subject, run)
    thresholds = pd.read_csv(
        os.path.join(interim, "stance_magnitude_thresholds_manual.csv")
    )
    trow = thresholds[(thresholds.subject == subject) & (thresholds.run == run)].iloc[0]

    os.makedirs(output_dir, exist_ok=True)

    for foot, kw in [("left", "LF"), ("right", "RF")]:
        imu_path = os.path.join(interim, subject, run, "imu", f"{kw}.csv")
        imu = IMU(imu_path)
        imu.acc_to_meter_per_square_sec()
        imu.gyro_to_rad()
        time = imu.time()
        gyro_mag = np.linalg.norm(imu.gyro(), axis=1)
        traj = pd.read_json(
            os.path.join(interim, subject, run, f"_trajectory_estimation_{foot}.json")
        )

        gmag = float(trow[f"stance_magnitude_threshold_{foot}"])
        count = int(trow[f"stance_count_threshold_{foot}"])
        ic_samples, fo_samples, ic_times, fo_times, stance = tunca_gait_events(
            imu,
            gmag,
            count,
            0.7,
            0.01,
            0.2,
            0,
            traj,
            "",
        )

        fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
        axes[0].plot(time, gyro_mag, color="steelblue", linewidth=0.8, label="Gyro magnitude")
        axes[0].axhline(gmag, color="orange", linestyle="--", label=f"Stance threshold ({gmag})")
        axes[0].fill_between(
            time,
            0,
            gyro_mag.max(),
            where=stance,
            alpha=0.15,
            color="green",
            label="Stance phase",
        )
        for t, label, color in [
            (ic_times, "Heel strike (IC)", "red"),
            (fo_times, "Toe off (FO)", "purple"),
        ]:
            axes[0].scatter(t, np.interp(t, time, gyro_mag), color=color, s=18, label=label, zorder=5)
        axes[0].set_ylabel("Gyro magnitude (rad/s)")
        axes[0].legend(loc="upper right", fontsize=8)
        axes[0].set_title(f"{subject} | {run} | {foot} foot | IC={len(ic_times)}, FO={len(fo_times)}")

        accel = np.transpose(imu.accel())
        for idx, axis_name in enumerate(["X", "Y", "Z"]):
            axes[1].plot(time, accel[idx], linewidth=0.8, label=f"Acc {axis_name}")
        axes[1].scatter(ic_times, np.zeros_like(ic_times), color="red", marker="|", s=80, label="IC")
        axes[1].scatter(fo_times, np.zeros_like(fo_times), color="purple", marker="|", s=80, label="FO")
        axes[1].set_xlabel("Time (s)")
        axes[1].set_ylabel("Acceleration (m/s²)")
        axes[1].legend(loc="upper right", fontsize=8)

        out_path = os.path.join(output_dir, f"gait_phases_{run}_{foot}.png")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved {out_path}")


def main():
    subject = "imu_thom_2026_06_06"
    runs = ["visit3km", "visit5km", "visit7km"]
    with open(os.path.join(os.path.dirname(__file__), "..", "path.json")) as f:
        paths = json.load(f)
    data_base_path = paths["data_charite"]
    output_dir = os.path.join(
        data_base_path, "processed", "figures_gait_phases", subject
    )
    for run in runs:
        plot_visit(subject, run, data_base_path, output_dir)


if __name__ == "__main__":
    main()
