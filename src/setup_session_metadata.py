"""Non-interactive setup of initial contact and stance thresholds for new sessions."""

import json
import os

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_prominences

from data_reader.imu import IMU


def auto_detect_initial_contact(imu_path, prominence_threshold=3):
    imu = IMU(imu_path)
    imu.gyro_to_rad()
    time = imu.time()
    accel = np.transpose(imu.accel())
    accel_norm = np.linalg.norm(accel, axis=0)
    peaks, _ = find_peaks(accel_norm)
    prominences = peak_prominences(accel_norm, peaks)[0]
    valid_peaks = peaks[prominences > prominence_threshold]
    if len(valid_peaks) == 0:
        raise ValueError(f"No IC peak found for {imu_path}")
    return round(float(time[valid_peaks[0]]), 6)


def append_if_missing(df, row, key_cols):
    mask = np.ones(len(df), dtype=bool)
    for col in key_cols:
        mask &= df[col] == row[col]
    if not mask.any():
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    return df


def main():
    subject = "imu_thom_2026_06_06"
    runs = ["visit3km", "visit5km", "visit7km"]

    with open(os.path.join(os.path.dirname(__file__), "..", "path.json")) as f:
        paths = json.load(f)
    interim_base_path = os.path.join(paths["data_charite"], "interim")

    ic_manual_path = os.path.join(interim_base_path, "imu_initial_contact_manual.csv")
    threshold_manual_path = os.path.join(
        interim_base_path, "stance_magnitude_thresholds_manual.csv"
    )

    ic_df = pd.read_csv(ic_manual_path)
    threshold_df = pd.read_csv(threshold_manual_path)

    for run in runs:
        imu_dir = os.path.join(interim_base_path, subject, run, "imu")
        ic_left = auto_detect_initial_contact(os.path.join(imu_dir, "LF.csv"))
        ic_right = auto_detect_initial_contact(os.path.join(imu_dir, "RF.csv"))
        ic_row = {
            "subject": subject,
            "run": run,
            "imu_initial_contact_left": ic_left,
            "imu_initial_contact_right": ic_right,
        }
        threshold_row = {
            "subject": subject,
            "run": run,
            "stance_magnitude_threshold_left": 0.7,
            "stance_magnitude_threshold_right": 0.7,
            "stance_count_threshold_left": 8,
            "stance_count_threshold_right": 8,
        }
        ic_df = append_if_missing(ic_df, ic_row, ["subject", "run"])
        threshold_df = append_if_missing(threshold_df, threshold_row, ["subject", "run"])
        print(f"{run}: IC left={ic_left}, IC right={ic_right}")

    ic_df.to_csv(ic_manual_path, index=False)
    threshold_df.to_csv(threshold_manual_path, index=False)
    print("Updated manual metadata CSV files.")


if __name__ == "__main__":
    main()
