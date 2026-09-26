"""Fit and save the exploratory BLE presence classifier from pilot_features_5s.csv."""
import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

FEATURES = [
    "sample_count", "has_signal", "last_seen_gap_s",
    "consecutive_empty_windows", "sample_count_15s", "sample_count_30s",
    "sample_count_ratio", "mean_rssi", "std_rssi", "rssi_slope",
    "rssi_drop_from_baseline",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-csv", type=Path,
                        default=Path("training/pilot_features_5s.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("model"))
    args = parser.parse_args()

    raw_bytes = args.features_csv.read_bytes()
    data = pd.read_csv(args.features_csv)
    required = set(FEATURES) | {"physical_state", "run_id"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if data.empty or data[FEATURES].isna().all().any():
        raise ValueError("Empty data or a model feature entirely missing")
    if data["physical_state"].isna().any():
        raise ValueError("Training labels contain missing values")
    if set(data["physical_state"]) != {"PRESENT", "ABSENT"}:
        raise ValueError("Expected exactly PRESENT and ABSENT labels")
    if data["run_id"].nunique() < 2:
        raise ValueError("Expected multiple measurement runs")
    if data.duplicated(["run_id", "window_index"]).any():
        raise ValueError("Duplicate run_id/window_index pairs")

    x = data[FEATURES].apply(pd.to_numeric, errors="raise")
    model = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    )
    model.fit(x, data["physical_state"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "ble_pilot_model.joblib"
    metadata_path = args.output_dir / "ble_pilot_model.json"
    joblib.dump(model, model_path)

    # A round-trip check verifies that the saved file retains feature names and predictions.
    restored = joblib.load(model_path)
    if list(restored.feature_names_in_) != FEATURES:
        raise RuntimeError("Saved feature names do not match")
    if not (restored.predict(x.head(20)) == model.predict(x.head(20))).all():
        raise RuntimeError("Reloaded predictions differ")

    metadata = {
        "model_version": "pilot-v1",
        "window_seconds": 5,
        "feature_names": FEATURES,
        "classes": list(map(str, restored.classes_)),
        "training_rows": len(data),
        "training_runs": sorted(data["run_id"].unique().tolist()),
        "class_counts": {str(k): int(v) for k, v in
                         data["physical_state"].value_counts().items()},
        "source_csv_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "scikit_learn_version": sklearn.__version__,
        "note": "Pilot headset model; all available runs used for fitting. "
                "Do not interpret training accuracy as held-out performance.",
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved model: {model_path}")
    print(f"Saved metadata: {metadata_path}")
    print(f"Windows: {len(data)}, runs: {data['run_id'].nunique()}")
    print(f"Labels: {metadata['class_counts']}")
    print(f"scikit-learn: {sklearn.__version__}")


if __name__ == "__main__":
    main()
