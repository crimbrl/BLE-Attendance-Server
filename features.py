"""Turn stored five-second windows into the exact 11 pilot model features."""
import math
from statistics import median

WINDOW_NS = 5_000_000_000
FEATURES = [
    "sample_count", "has_signal", "last_seen_gap_s",
    "consecutive_empty_windows", "sample_count_15s", "sample_count_30s",
    "sample_count_ratio", "mean_rssi", "std_rssi", "rssi_slope",
    "rssi_drop_from_baseline",
]


def _slope(windows):
    if len(windows) < 2:
        return None
    xs = [w["window_index"] * 5 + 2.5 for w in windows]
    ys = [w["mean_rssi_dbm"] for w in windows]
    x_bar, y_bar = sum(xs) / len(xs), sum(ys) / len(ys)
    return sum((x-x_bar)*(y-y_bar) for x, y in zip(xs, ys)) / sum(
        (x-x_bar)**2 for x in xs
    )


def build_feature_rows(windows, start_elapsed_ns):
    """Return [(window_index, feature_dict | None, reason | None), ...].

    Windows must be ordered by index. Missing/INTERRUPTED segments are not
    invented as BLE silence. Following a break, wait for 6 valid consecutive
    windows (30 s) before trusting a prediction.
    """
    result = []
    past = []
    recent_valid = []
    last_seen_ns = None
    previous_index = -1
    after_break = False
    empty_streak = 0

    for window in windows:
        index = window["window_index"]
        if index <= previous_index:
            raise ValueError("Window indexes must be strictly increasing")
        gap = index != previous_index + 1
        if gap:
            recent_valid.clear()
            after_break = True
            empty_streak = 0
        previous_index = index

        if window["scan_state"] != "RUNNING":
            recent_valid.clear()
            after_break = True
            empty_streak = 0
            result.append((index, None, "INTERRUPTED"))
            continue

        count = window["sample_count"]
        empty_streak = 0 if count else empty_streak + 1
        if count:
            last_seen_ns = window["last_packet_elapsed_ns"]
        recent_valid.append(window)
        if len(recent_valid) > 6:
            recent_valid.pop(0)

        # The first six original windows are the pilot's calibration period.
        # Use only earlier positive-count windows: never read future data.
        calibration = [w for w in past if w["window_index"] < 6
                       and w["scan_state"] == "RUNNING" and w["sample_count"] > 0]
        count_base = median(w["sample_count"] for w in calibration) if calibration else None
        rssi_base = median(w["mean_rssi_dbm"] for w in calibration) if calibration else None
        if count:
            recent_rssi = []
            for w in reversed(recent_valid[-3:]):
                if w["sample_count"] == 0:
                    break
                recent_rssi.append(w)
            recent_rssi.reverse()
        else:
            recent_rssi = []

        features = {
            "sample_count": count,
            "has_signal": int(count > 0),
            "last_seen_gap_s": (
                (start_elapsed_ns + (index + 1) * WINDOW_NS - last_seen_ns) / 1e9
                if last_seen_ns is not None else None
            ),
            "consecutive_empty_windows": empty_streak,
            "sample_count_15s": (
                sum(w["sample_count"] for w in recent_valid[-3:])
                if index >= 2 and len(recent_valid) >= 3 else None
            ),
            "sample_count_30s": (
                sum(w["sample_count"] for w in recent_valid[-6:])
                if index >= 5 and len(recent_valid) >= 6 else None
            ),
            "sample_count_ratio": (count / count_base if count_base else None),
            "mean_rssi": window["mean_rssi_dbm"] if count else None,
            "std_rssi": window["std_rssi_dbm"] if count else None,
            "rssi_slope": _slope(recent_rssi),
            "rssi_drop_from_baseline": (
                window["mean_rssi_dbm"] - rssi_base
                if count and rssi_base is not None else None
            ),
        }
        if list(features) != FEATURES:
            raise RuntimeError("Feature order mismatch")
        reason = "RECALIBRATING" if after_break and len(recent_valid) < 6 else None
        if after_break and len(recent_valid) >= 6:
            after_break = False
        result.append((index, features, reason))
        past.append(window)
    return result
