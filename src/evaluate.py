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
) -> dict:
    """
    Simple alerting evaluation:
    - incident_recall: fraction of incidents that had >=1 alert before start
    - lead_time_median_min: median minutes before incident start
    - false_alerts_per_day: alerts not in any warning window
    """
    meta = meta_df.copy()
    meta["t_end_ts"] = pd.to_datetime(meta["t_end_ts"])
    meta["prob"] = y_probs
    meta["alert"] = (y_probs >= threshold).astype(int)

    horizon_length = horizon_steps * 300

    total_incidents = 0
    caught = 0
    lead_times_sec = []

    false_alerts = 0
    total_days = 0.0

    for series_id, events in meta.groupby("series_id", sort=False):
        events = events.sort_values("t_end_ts")
        windows = incident_windows_by_series[series_id]
        alert_times = events.loc[events["alert"] == 1, "t_end_ts"].tolist()

        # Calculate day span
        t0, t1 = events["t_end_ts"].min(), events["t_end_ts"].max()
        total_days += max((t1 - t0).total_seconds(), 0.0) / 86400.0

        covered_alert_times = set()
        for start, end in windows:
            total_incidents += 1
            warn_start = start - pd.Timedelta(seconds=horizon_length)

            pre_alerts = [t for t in alert_times if warn_start <= t < start]
            if pre_alerts:
                caught += 1
                # The first alert for this incident
                lead_times_sec.append((start - pre_alerts[0]).total_seconds())

            for t in alert_times:
                # I also check if the model fired late
                if warn_start <= t < start or start <= t <= end:
                    covered_alert_times.add(t)

        false_alerts += sum(t not in covered_alert_times for t in alert_times)

    incident_recall = caught / total_incidents if total_incidents else 0.0
    false_per_day = false_alerts / total_days if total_days > 0 else 0.0
    lead_median_min = np.median(lead_times_sec) / 60.0 if lead_times_sec else None

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
):

    thresholds = [0.01, 0.02, 0.05, 0.1]
    best = None

    for thr in thresholds:
        metrics = alerting_eval(
            meta_df=meta_val,
            y_probs=probs_val,
            incident_windows_by_series=incident_windows_by_series,
            threshold=thr,
            horizon_steps=horizon_steps,
        )

        # prioritize recall, then fewer false alerts
        if best is None:
            best = metrics
        else:
            if (metrics["incident_recall"] > best["incident_recall"]) or (
                metrics["incident_recall"] == best["incident_recall"] and
                metrics["false_alerts_per_day"] < best["false_alerts_per_day"]
            ):
                best = metrics

    return best
