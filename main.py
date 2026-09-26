"""BLE attendance pilot: session and five-second window ingestion."""
import os
import sqlite3
import json
import asyncio
import logging
import time
from contextlib import contextmanager
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from uuid import UUID

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator
import joblib
import pandas as pd
import sklearn

from features import FEATURES, build_feature_rows
from attendance_rules import RULE_VERSION, PRESENT_STREAK, derive_events

WINDOW_NS = 5_000_000_000
WINDOW_MS = 5000
NO_DATA_TIMEOUT_MS = 10 * 60 * 1000
LIVE_WINDOW_TOLERANCE_MS = 30 * 1000
MONITOR_DURATION_MS = int(os.environ.get("BLE_CLASS_DURATION_MINUTES", "90")) * 60 * 1000
if MONITOR_DURATION_MS <= NO_DATA_TIMEOUT_MS:
    raise ValueError("BLE_CLASS_DURATION_MINUTES must exceed 10")
DB_PATH = Path(os.environ.get("BLE_DB_PATH", "ble_attendance.db"))
MODEL_DIR = Path(__file__).resolve().parent / "model"
logger = logging.getLogger(__name__)


def now_epoch_ms():
    return int(time.time() * 1000)


@asynccontextmanager
async def lifespan(app):
    async def monitor():
        while True:
            try:
                await asyncio.to_thread(check_no_data)
            except Exception:
                logger.exception("Server-side collection monitor failed")
            await asyncio.sleep(5)

    task = asyncio.create_task(monitor())
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


app = FastAPI(title="BLE Attendance Pilot", version="0.1", lifespan=lifespan)


def load_prediction_model():
    metadata = json.loads((MODEL_DIR / "ble_pilot_model.json").read_text(encoding="utf-8"))
    if metadata["feature_names"] != FEATURES:
        raise RuntimeError("Model and server feature order differ")
    if metadata["scikit_learn_version"] != sklearn.__version__:
        raise RuntimeError("Train and serve with the same scikit-learn version")
    model = joblib.load(MODEL_DIR / "ble_pilot_model.joblib")
    if list(model.feature_names_in_) != FEATURES:
        raise RuntimeError("Stored model feature order differs")
    return model, metadata["model_version"]


class SessionStart(BaseModel):
    protocol_version: Literal[1]
    session_id: UUID
    class_id: str = Field(min_length=1, max_length=100)
    beacon_id: str = Field(min_length=1, max_length=100)
    start_epoch_ms: int = Field(ge=0)
    start_elapsed_ns: int = Field(ge=0)


class Window(BaseModel):
    window_index: int = Field(ge=0)
    scan_state: Literal["RUNNING", "INTERRUPTED"]
    sample_count: int = Field(ge=0)
    mean_rssi_dbm: float | None
    std_rssi_dbm: float | None
    last_packet_elapsed_ns: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_signal(self):
        empty = self.sample_count == 0
        stats = (self.mean_rssi_dbm, self.std_rssi_dbm, self.last_packet_elapsed_ns)
        if self.scan_state == "INTERRUPTED" and not empty:
            raise ValueError("INTERRUPTED window must have zero samples")
        if empty and any(value is not None for value in stats):
            raise ValueError("empty window must have null RSSI and last packet")
        if not empty and any(value is None for value in stats):
            raise ValueError("nonempty window requires RSSI and last packet")
        if self.std_rssi_dbm is not None and self.std_rssi_dbm < 0:
            raise ValueError("std_rssi_dbm must be nonnegative")
        return self


class WindowBatch(BaseModel):
    protocol_version: Literal[1]
    session_id: UUID
    beacon_id: str = Field(min_length=1, max_length=100)
    windows: list[Window] = Field(min_length=1, max_length=120)


@contextmanager
def database():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH, timeout=10)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys=ON")
    con.execute("PRAGMA busy_timeout=10000")
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def init_db():
    with database() as con:
        con.execute("PRAGMA journal_mode=WAL")
        con.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                class_id TEXT NOT NULL,
                beacon_id TEXT NOT NULL,
                start_epoch_ms INTEGER NOT NULL,
                start_elapsed_ns INTEGER NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS windows (
                session_id TEXT NOT NULL REFERENCES sessions(session_id),
                beacon_id TEXT NOT NULL,
                window_index INTEGER NOT NULL,
                scan_state TEXT NOT NULL,
                sample_count INTEGER NOT NULL,
                mean_rssi_dbm REAL,
                std_rssi_dbm REAL,
                last_packet_elapsed_ns INTEGER,
                received_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (session_id, beacon_id, window_index)
            );
            CREATE TABLE IF NOT EXISTS attendance_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES sessions(session_id),
                class_id TEXT NOT NULL,
                beacon_id TEXT NOT NULL,
                window_index INTEGER NOT NULL,
                event_type TEXT NOT NULL CHECK (event_type IN ('LEFT', 'RETURNED')),
                event_time_epoch_ms INTEGER NOT NULL,
                rule_version TEXT NOT NULL,
                active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(session_id, beacon_id, window_index, event_type, rule_version)
            );
            CREATE TABLE IF NOT EXISTS no_data_alerts (
                alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES sessions(session_id),
                class_id TEXT NOT NULL,
                beacon_id TEXT NOT NULL,
                missing_since_epoch_ms INTEGER NOT NULL,
                suspected_since_epoch_ms INTEGER NOT NULL,
                reason TEXT NOT NULL,
                triggered_at_epoch_ms INTEGER NOT NULL,
                resolved_at_epoch_ms INTEGER,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE UNIQUE INDEX IF NOT EXISTS one_open_no_data_alert
                ON no_data_alerts(session_id) WHERE resolved_at_epoch_ms IS NULL;
        """)
        existing = {row["name"] for row in con.execute("PRAGMA table_info(windows)")}
        for name, sql_type in (
            ("predicted_state", "TEXT"),
            ("p_absent", "REAL"),
            ("model_version", "TEXT"),
            ("prediction_reason", "TEXT"),
            ("predicted_at", "TEXT"),
        ):
            if name not in existing:
                con.execute(f"ALTER TABLE windows ADD COLUMN {name} {sql_type}")
        existing_sessions = {row["name"] for row in con.execute("PRAGMA table_info(sessions)")}
        for name in ("monitor_started_epoch_ms", "last_live_epoch_ms",
                     "monitor_until_epoch_ms", "last_live_window_index",
                     "fresh_present_streak"):
            if name not in existing_sessions:
                con.execute(f"ALTER TABLE sessions ADD COLUMN {name} INTEGER")


init_db()
MODEL, MODEL_VERSION = load_prediction_model()


def refresh_predictions(con, session_id: str, beacon_id: str, start_elapsed_ns: int):
    """Recompute this pilot session after inserts, including late/out-of-order windows."""
    rows = con.execute("""SELECT * FROM windows
        WHERE session_id=? AND beacon_id=? ORDER BY window_index""",
        (session_id, beacon_id)).fetchall()
    calculated = build_feature_rows(rows, start_elapsed_ns)
    ready = [(index, features) for index, features, reason in calculated
             if features is not None and reason is None]
    predictions = {}
    if ready:
        frame = pd.DataFrame([values for _, values in ready], columns=FEATURES)
        labels = MODEL.predict(frame)
        probs = MODEL.predict_proba(frame)
        absent_col = list(MODEL.classes_).index("ABSENT")
        predictions = {
            index: (str(label), float(prob[absent_col]))
            for (index, _), label, prob in zip(ready, labels, probs)
        }
    for row, (index, _, reason) in zip(rows, calculated):
        label, p_absent = predictions.get(index, (None, None))
        version = MODEL_VERSION if label is not None else None
        previous_p = row["p_absent"]
        if (row["predicted_state"] == label
                and row["model_version"] == version
                and row["prediction_reason"] == reason
                and (previous_p == p_absent or (
                    previous_p is not None and p_absent is not None
                    and abs(previous_p - p_absent) < 1e-12))):
            continue
        con.execute("""UPDATE windows SET predicted_state=?, p_absent=?,
            model_version=?, prediction_reason=?, predicted_at=CURRENT_TIMESTAMP
            WHERE session_id=? AND beacon_id=? AND window_index=?""",
            (label, p_absent, version,
             reason, session_id, beacon_id, index))


def refresh_attendance_events(con, session):
    """Keep the event log consistent when missing windows arrive out of order."""
    rows = con.execute("""SELECT window_index, scan_state, predicted_state
        FROM windows WHERE session_id=? AND beacon_id=? ORDER BY window_index""",
        (session["session_id"], session["beacon_id"])).fetchall()
    events = derive_events(rows)
    # Retire alerts generated by earlier rules as well as superseded current alerts.
    con.execute("""UPDATE attendance_events SET active=0
        WHERE session_id=? AND beacon_id=?""",
        (session["session_id"], session["beacon_id"]))
    for index, kind in events:
        end_epoch_ms = session["start_epoch_ms"] + (index + 1) * 5000
        con.execute("""INSERT INTO attendance_events
            (session_id, class_id, beacon_id, window_index, event_type,
             event_time_epoch_ms, rule_version, active)
            VALUES (?, ?, ?, ?, ?, ?, ?, 1)
            ON CONFLICT(session_id, beacon_id, window_index, event_type, rule_version)
            DO UPDATE SET active=1""",
            (session["session_id"], session["class_id"], session["beacon_id"],
             index, kind, end_epoch_ms, RULE_VERSION))


def record_live_windows(con, session, accepted_indexes, received_ms):
    """Only recent, newly accepted predictions advance server-observed liveness."""
    previous_index = session["last_live_window_index"]
    present_streak = session["fresh_present_streak"] or 0
    last_live_ms = session["last_live_epoch_ms"]
    for index in sorted(accepted_indexes):
        window_end_ms = session["start_epoch_ms"] + (index + 1) * WINDOW_MS
        if abs(received_ms - window_end_ms) > LIVE_WINDOW_TOLERANCE_MS:
            continue  # Replayed/offline Window: useful history, not proof of current state.
        row = con.execute("""SELECT scan_state, predicted_state FROM windows
            WHERE session_id=? AND beacon_id=? AND window_index=?""",
            (session["session_id"], session["beacon_id"], index)).fetchone()
        if row["scan_state"] != "RUNNING" or row["predicted_state"] not in ("PRESENT", "ABSENT"):
            continue
        if previous_index is not None and index <= previous_index:
            continue
        if row["predicted_state"] == "PRESENT":
            present_streak = present_streak + 1 if previous_index == index - 1 else 1
        else:
            present_streak = 0
        previous_index = index
        last_live_ms = received_ms
        if present_streak >= PRESENT_STREAK:
            con.execute("""UPDATE no_data_alerts SET resolved_at_epoch_ms=?
                WHERE session_id=? AND resolved_at_epoch_ms IS NULL""",
                (received_ms, session["session_id"]))
    con.execute("""UPDATE sessions SET last_live_epoch_ms=?, last_live_window_index=?,
        fresh_present_streak=? WHERE session_id=?""",
        (last_live_ms, previous_index, present_streak, session["session_id"]))


def check_no_data(now_ms=None):
    """Run on a server timer; no phone POST is required to detect silence."""
    now_ms = now_epoch_ms() if now_ms is None else now_ms
    with database() as con:
        con.execute("BEGIN IMMEDIATE")  # Serialize workers sharing this SQLite file.
        sessions = con.execute("""SELECT * FROM sessions
            WHERE monitor_started_epoch_ms IS NOT NULL
              AND monitor_until_epoch_ms > ?""", (now_ms,)).fetchall()
        for session in sessions:
            last_live = (session["last_live_epoch_ms"] if session["last_live_epoch_ms"] is not None
                         else session["monitor_started_epoch_ms"])
            if con.execute("""SELECT 1 FROM no_data_alerts WHERE session_id=?
                AND resolved_at_epoch_ms IS NULL""", (session["session_id"],)).fetchone():
                continue
            latest_event = con.execute("""SELECT event_type FROM attendance_events
                WHERE session_id=? AND beacon_id=? AND active=1
                ORDER BY window_index DESC, event_id DESC LIMIT 1""",
                (session["session_id"], session["beacon_id"])).fetchone()
            if latest_event and latest_event["event_type"] == "LEFT":
                continue  # The professor already has a BLE departure event.

            deadline = last_live + NO_DATA_TIMEOUT_MS
            suspected_since = last_live
            reason = "NO_DATA"
            last_index = session["last_live_window_index"]
            if last_index is not None:
                rows = con.execute("""SELECT window_index, scan_state, predicted_state
                    FROM windows WHERE session_id=? AND beacon_id=? AND window_index<=?
                    ORDER BY window_index""",
                    (session["session_id"], session["beacon_id"], last_index)).fetchall()
                _, candidate = derive_events(rows, include_candidate=True)
                if candidate is not None:
                    candidate_start = max(session["monitor_started_epoch_ms"],
                                          last_live - (last_index - candidate) * WINDOW_MS)
                    if candidate_start < last_live:
                        suspected_since = candidate_start
                        reason = "ABSENCE_AND_NO_DATA"
                        deadline = candidate_start + NO_DATA_TIMEOUT_MS
            if now_ms < deadline:
                continue
            con.execute("""INSERT OR IGNORE INTO no_data_alerts
                (session_id, class_id, beacon_id, missing_since_epoch_ms,
                 suspected_since_epoch_ms, reason, triggered_at_epoch_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (session["session_id"], session["class_id"], session["beacon_id"],
                 last_live, suspected_since, reason, now_ms))


def backfill_stored_windows():
    """Predict rows saved before this server version, preserving raw data."""
    with database() as con:
        sessions = con.execute("""SELECT s.session_id, s.class_id, s.beacon_id,
            s.start_epoch_ms, s.start_elapsed_ns
            FROM sessions s WHERE EXISTS (
                SELECT 1 FROM windows w WHERE w.session_id=s.session_id
            )""").fetchall()
        for session in sessions:
            refresh_predictions(con, session["session_id"], session["beacon_id"],
                                session["start_elapsed_ns"])
            refresh_attendance_events(con, session)


backfill_stored_windows()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/collection-sessions", status_code=201)
def create_session(payload: SessionStart):
    session_id = str(payload.session_id)
    created_ms = now_epoch_ms()
    incoming = (
        payload.class_id, payload.beacon_id,
        payload.start_epoch_ms, payload.start_elapsed_ns,
    )
    with database() as con:
        row = con.execute("SELECT * FROM sessions WHERE session_id=?", (session_id,)).fetchone()
        if row:
            existing = tuple(row[field] for field in
                             ("class_id", "beacon_id", "start_epoch_ms", "start_elapsed_ns"))
            if existing != incoming:
                raise HTTPException(409, "session_id already has different metadata")
            return {"session_id": session_id, "already_exists": True}
        con.execute("""INSERT INTO sessions
            (session_id, class_id, beacon_id, start_epoch_ms, start_elapsed_ns,
             monitor_started_epoch_ms, monitor_until_epoch_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (session_id, *incoming, created_ms, created_ms + MONITOR_DURATION_MS))
    return {"session_id": session_id, "already_exists": False}


@app.post("/v1/ble-windows:batch")
def receive_windows(payload: WindowBatch):
    session_id = str(payload.session_id)
    accepted = duplicates = 0
    accepted_indexes = []
    with database() as con:
        session = con.execute(
            "SELECT * FROM sessions WHERE session_id=?", (session_id,)
        ).fetchone()
        if session is None:
            raise HTTPException(404, "register collection session first")
        if session["beacon_id"] != payload.beacon_id:
            raise HTTPException(409, "beacon_id does not match session")

        for window in payload.windows:
            start = session["start_elapsed_ns"] + window.window_index * WINDOW_NS
            end = start + WINDOW_NS
            if window.last_packet_elapsed_ns is not None and not (
                start <= window.last_packet_elapsed_ns < end
            ):
                raise HTTPException(422, f"window {window.window_index}: last packet outside window")

            values = (
                window.scan_state, window.sample_count, window.mean_rssi_dbm,
                window.std_rssi_dbm, window.last_packet_elapsed_ns,
            )
            existing = con.execute("""SELECT scan_state, sample_count, mean_rssi_dbm,
                std_rssi_dbm, last_packet_elapsed_ns FROM windows
                WHERE session_id=? AND beacon_id=? AND window_index=?""",
                (session_id, payload.beacon_id, window.window_index)).fetchone()
            if existing is not None:
                if tuple(existing) != values:
                    raise HTTPException(409, f"window {window.window_index}: conflicting retry")
                duplicates += 1
                continue

            con.execute("""INSERT INTO windows
                (session_id, beacon_id, window_index, scan_state, sample_count,
                 mean_rssi_dbm, std_rssi_dbm, last_packet_elapsed_ns)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (session_id, payload.beacon_id, window.window_index, *values))
            accepted += 1
            accepted_indexes.append(window.window_index)
        if accepted:
            refresh_predictions(con, session_id, payload.beacon_id,
                                session["start_elapsed_ns"])
            refresh_attendance_events(con, session)
            record_live_windows(con, session, accepted_indexes, now_epoch_ms())
        requested = sorted({w.window_index for w in payload.windows})
        marks = ",".join("?" for _ in requested)
        predictions = [dict(row) for row in con.execute(f"""
            SELECT window_index, predicted_state, p_absent, prediction_reason
            FROM windows WHERE session_id=? AND beacon_id=?
            AND window_index IN ({marks}) ORDER BY window_index""",
            (session_id, payload.beacon_id, *requested)).fetchall()]
        events = [dict(row) for row in con.execute(f"""SELECT event_id, window_index,
            event_type, event_time_epoch_ms FROM attendance_events
            WHERE session_id=? AND beacon_id=? AND active=1
            AND window_index IN ({marks}) ORDER BY window_index""",
            (session_id, payload.beacon_id, *requested)).fetchall()]
    return {"session_id": session_id, "accepted": accepted,
            "duplicates": duplicates, "predictions": predictions, "events": events}


@app.get("/v1/classes/{class_id}/alerts")
def list_class_alerts(class_id: str):
    """Demo dashboard feed. Add professor authentication before deployment."""
    with database() as con:
        rows = con.execute("""SELECT event_id, session_id, beacon_id, window_index,
            event_type, event_time_epoch_ms, rule_version, created_at
            FROM attendance_events WHERE class_id=? AND active=1
            ORDER BY event_id DESC LIMIT 100""", (class_id,)).fetchall()
        alerts = []
        open_no_data = {row[0] for row in con.execute("""SELECT session_id FROM no_data_alerts
            WHERE class_id=? AND resolved_at_epoch_ms IS NULL""", (class_id,))}
        for row in rows:
            event = dict(row)
            event["alert_key"] = f"ble:{event['event_id']}"
            event["source"] = "BLE"
            returned = con.execute("""SELECT 1 FROM attendance_events
                WHERE session_id=? AND active=1 AND event_type='RETURNED'
                AND window_index>? LIMIT 1""",
                (event["session_id"], event["window_index"])).fetchone()
            event["status"] = "RESOLVED" if event["event_type"] == "RETURNED" or returned else "OPEN"
            if (event["event_type"] == "LEFT" and event["status"] == "OPEN"
                    and event["session_id"] in open_no_data):
                continue  # The same episode already has an open server-side alert.
            alerts.append(event)
        for row in con.execute("""SELECT alert_id AS event_id, session_id, beacon_id,
            missing_since_epoch_ms, suspected_since_epoch_ms, reason,
            triggered_at_epoch_ms AS event_time_epoch_ms,
            resolved_at_epoch_ms, created_at FROM no_data_alerts
            WHERE class_id=? ORDER BY alert_id DESC LIMIT 100""", (class_id,)):
            alert = dict(row)
            alert["alert_key"] = f"no_data:{alert['event_id']}"
            alert["event_type"] = "NO_DATA"
            alert["rule_version"] = "no-data-10m-v1"
            alert["source"] = "SERVER"
            alert["status"] = "RESOLVED" if alert["resolved_at_epoch_ms"] is not None else "OPEN"
            alerts.append(alert)
        alerts.sort(key=lambda row: row["event_time_epoch_ms"], reverse=True)
        return {"class_id": class_id, "alerts": alerts[:100]}


@app.get("/v1/collection-sessions/{session_id}/windows")
def list_windows(session_id: UUID):
    with database() as con:
        session = con.execute(
            "SELECT session_id FROM sessions WHERE session_id=?", (str(session_id),)
        ).fetchone()
        if session is None:
            raise HTTPException(404, "session not found")
        rows = con.execute("""SELECT beacon_id, window_index, scan_state, sample_count,
            mean_rssi_dbm, std_rssi_dbm, last_packet_elapsed_ns,
            predicted_state, p_absent, model_version, prediction_reason, predicted_at FROM windows
            WHERE session_id=? ORDER BY window_index LIMIT 1000""",
            (str(session_id),)).fetchall()
        return {"session_id": str(session_id), "windows": [dict(row) for row in rows]}
