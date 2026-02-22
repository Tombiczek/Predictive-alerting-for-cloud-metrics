import numpy as np
import pandas as pd

import torch


def predict_proba_tsai(clf, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
    model = clf.model if hasattr(clf, "model") else clf
    device = next(model.parameters()).device
    model.eval()

    probs_all = []

    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i:i+batch_size], dtype=torch.float32, device=device)
            logits = model(xb)

            if logits.ndim != 2 or logits.shape[1] != 2:
                raise ValueError(f"Expected logits shape [B, 2], got {tuple(logits.shape)}")

            probs1 = torch.nn.functional.softmax(logits, dim=1)[:, 1]
            probs_all.append(probs1.cpu().numpy())

    return np.concatenate(probs_all, axis=0)



def predict_proba_sklearn(clf, X: np.ndarray) -> np.ndarray:
    return clf.predict_proba(X)[:, 1]


def alerting_eval(
    meta_df: pd.DataFrame,
    y_probs: np.ndarray,
    incident_windows_by_series: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]],
    threshold: float,
    horizon_steps: int,
    step_seconds: int = 300,
) -> dict:
    """
    Simple alerting evaluation:
    - incident_recall: fraction of incidents that had >=1 alert before start
    - lead_time_median_min: median minutes before incident start
    - false_alerts_per_day: alerts not in any warning window

    Warning window: [start - H*step, start)
    """
    meta = meta_df.copy()
    meta["t_end_ts"] = pd.to_datetime(meta["t_end_ts"])
    meta["prob"] = y_probs
    meta["alert"] = (y_probs >= threshold).astype(int)

    warning_seconds = horizon_steps * step_seconds

    total_incidents = 0
    caught = 0
    lead_times_sec = []

    false_alerts = 0
    total_days = 0.0

    for series_id, g in meta.groupby("series_id", sort=False):
        g = g.sort_values("t_end_ts")
        windows = incident_windows_by_series.get(series_id, [])
        if not windows:
            continue

        # For false alerts/day
        t0, t1 = g["t_end_ts"].min(), g["t_end_ts"].max()
        total_days += max((t1 - t0).total_seconds(), 0.0) / 86400.0

        # Precompute warning intervals for this series
        warning_intervals = []
        for start, end in windows:
            warn_start = start - pd.Timedelta(seconds=warning_seconds)
            warning_intervals.append((warn_start, start))  # [warn_start, start)

        # Incident recall + lead time
        for start, end in windows:
            total_incidents += 1
            warn_start = start - pd.Timedelta(seconds=warning_seconds)

            pre_alerts = g[(g["alert"] == 1) & (g["t_end_ts"] >= warn_start) & (g["t_end_ts"] < start)]
            if not pre_alerts.empty:
                caught += 1
                first_alert_time = pre_alerts["t_end_ts"].iloc[0]
                lead_times_sec.append((start - first_alert_time).total_seconds())

        # False alerts: alert not in ANY warning interval (and not inside incident)
        alert_times = g.loc[g["alert"] == 1, "t_end_ts"].tolist()
        for t in alert_times:
            in_warning = any(ws <= t < we for (ws, we) in warning_intervals)
            in_incident = any(start <= t <= end for (start, end) in windows)
            if (not in_warning) and (not in_incident):
                false_alerts += 1

    incident_recall = caught / total_incidents if total_incidents else 0.0
    false_per_day = false_alerts / total_days if total_days > 0 else 0.0

    if lead_times_sec:
        lead_median_min = np.median(lead_times_sec) / 60.0
    else:
        lead_median_min = None

    return {
        "threshold": threshold,
        "incident_recall": incident_recall,
        "incidents_total": total_incidents,
        "incidents_caught": caught,
        "lead_time_median_min": lead_median_min,
        "false_alerts_per_day": false_per_day,
    }

def pick_threshold(
    meta_val: pd.DataFrame,
    probs_val: np.ndarray,
    incident_windows_by_series: dict,
    horizon_steps: int,
    step_seconds: int = 300,
):
    thresholds = [0.01, 0.02, 0.05, 0.1]  # keep it simple
    best = None

    for thr in thresholds:
        m = alerting_eval(
            meta_df=meta_val,
            y_probs=probs_val,
            incident_windows_by_series=incident_windows_by_series,
            threshold=thr,
            horizon_steps=horizon_steps,
            step_seconds=step_seconds,
        )

        # prioritize recall, then fewer false alerts
        if best is None:
            best = m
        else:
            if (m["incident_recall"] > best["incident_recall"]) or (
                m["incident_recall"] == best["incident_recall"] and m["false_alerts_per_day"] < best["false_alerts_per_day"]
            ):
                best = m

    return best

