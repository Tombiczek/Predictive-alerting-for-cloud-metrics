from pathlib import Path
from typing import Any

from src.data.features import extract_features
from src.data.loading import build_labeled_dataset, normalize_series, load_incident_windows
from src.data.windowing import make_sliding_windows

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = REPO_ROOT / "data"
LABELS_PATH = DATA_DIR / "combined_windows.json"

TRAIN_SERIES = [
    DATA_DIR / "realAWSCloudwatch/ec2_cpu_utilization_825cc2.csv",
    DATA_DIR / "realAWSCloudwatch/ec2_cpu_utilization_c6585a.csv",
    DATA_DIR / "realAWSCloudwatch/rds_cpu_utilization_cc0c53.csv",
    DATA_DIR / "realAWSCloudwatch/ec2_cpu_utilization_5f5533.csv",
    DATA_DIR / "realAWSCloudwatch/ec2_disk_write_bytes_c0d644.csv",
    DATA_DIR / "realAWSCloudwatch/ec2_network_in_5abac7.csv",
    DATA_DIR / "realAWSCloudwatch/ec2_cpu_utilization_ac20cd.csv",
    DATA_DIR / "realAWSCloudwatch/ec2_cpu_utilization_24ae8d.csv",
    DATA_DIR / "realAWSCloudwatch/ec2_disk_write_bytes_1ef3de.csv",
    DATA_DIR / "realAWSCloudwatch/iio_us-east-1_i-a2eb1cd9_NetworkIn.csv",
    DATA_DIR / "realAWSCloudwatch/ec2_cpu_utilization_53ea38.csv",
]

VAL_SERIES = [
    DATA_DIR / "realAWSCloudwatch/ec2_cpu_utilization_77c1ca.csv",
    DATA_DIR / "realAWSCloudwatch/ec2_network_in_257a54.csv",
    DATA_DIR / "realAWSCloudwatch/elb_request_count_8c0756.csv",
]

TEST_SERIES = [
    DATA_DIR / "realAWSCloudwatch/ec2_cpu_utilization_fe7f93.csv",
    DATA_DIR / "realAWSCloudwatch/rds_cpu_utilization_e47b3b.csv",
    DATA_DIR / "realAWSCloudwatch/grok_asg_anomaly.csv",
]


def load_timeseries_dataset(window_size: int, horizon: int) -> dict[str, Any]:

    train_df = build_labeled_dataset(TRAIN_SERIES, LABELS_PATH)
    train_df = normalize_series(train_df)
    X_train, y_train, meta_train = make_sliding_windows(train_df, window_size, horizon)

    val_df = build_labeled_dataset(VAL_SERIES, LABELS_PATH)
    val_df = normalize_series(val_df)
    X_val, y_val, meta_val = make_sliding_windows(val_df, window_size, horizon)

    test_df = build_labeled_dataset(TEST_SERIES, LABELS_PATH)
    test_df = normalize_series(test_df)
    X_test, y_test, meta_test = make_sliding_windows(test_df, window_size, horizon)

    incident_windows_by_series = load_incident_windows(LABELS_PATH)

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "meta_train": meta_train, "meta_val": meta_val,
        "meta_test": meta_test, "incident_windows_by_series": incident_windows_by_series
    }


def load_features_dataset(window_size: int, horizon: int) -> dict[str, Any]:

    train_df = build_labeled_dataset(TRAIN_SERIES, LABELS_PATH)
    train_df = normalize_series(train_df)
    X_train_raw, y_train, meta_train = make_sliding_windows(train_df, window_size, horizon)
    X_train = extract_features(X_train_raw)

    val_df = build_labeled_dataset(VAL_SERIES, LABELS_PATH)
    val_df = normalize_series(val_df)
    X_val_raw, y_val, meta_val = make_sliding_windows(val_df, window_size, horizon)
    X_val = extract_features(X_val_raw)

    test_df = build_labeled_dataset(TEST_SERIES, LABELS_PATH)
    test_df = normalize_series(test_df)
    X_test_raw, y_test, meta_test = make_sliding_windows(test_df, window_size, horizon)
    X_test = extract_features(X_test_raw)

    incident_windows_by_series = load_incident_windows(LABELS_PATH)

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "meta_train": meta_train, "meta_val": meta_val,
        "meta_test": meta_test, "incident_windows_by_series": incident_windows_by_series
    }
