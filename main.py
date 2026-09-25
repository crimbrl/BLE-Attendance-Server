"""BLE attendance pilot: session and five-second window ingestion."""
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Literal
from uuid import UUID

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

WINDOW_NS = 5_000_000_000
DB_PATH = Path(os.environ.get("BLE_DB_PATH", "ble_attendance.db"))
app = FastAPI(title="BLE Attendance Pilot", version="0.1")


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
        """)


init_db()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/collection-sessions", status_code=201)
def create_session(payload: SessionStart):
    session_id = str(payload.session_id)
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
            (session_id, class_id, beacon_id, start_epoch_ms, start_elapsed_ns)
            VALUES (?, ?, ?, ?, ?)""", (session_id, *incoming))
    return {"session_id": session_id, "already_exists": False}


@app.post("/v1/ble-windows:batch")
def receive_windows(payload: WindowBatch):
    session_id = str(payload.session_id)
    accepted = duplicates = 0
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
    return {"session_id": session_id, "accepted": accepted, "duplicates": duplicates}


@app.get("/v1/collection-sessions/{session_id}/windows")
def list_windows(session_id: UUID):
    with database() as con:
        session = con.execute(
            "SELECT session_id FROM sessions WHERE session_id=?", (str(session_id),)
        ).fetchone()
        if session is None:
            raise HTTPException(404, "session not found")
        rows = con.execute("""SELECT beacon_id, window_index, scan_state, sample_count,
            mean_rssi_dbm, std_rssi_dbm, last_packet_elapsed_ns FROM windows
            WHERE session_id=? ORDER BY window_index LIMIT 1000""",
            (str(session_id),)).fetchall()
        return {"session_id": str(session_id), "windows": [dict(row) for row in rows]}
