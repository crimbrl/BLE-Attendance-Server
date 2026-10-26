"""Editable classroom catalog; existing mappings are never overwritten on startup."""
import os
import re
import sqlite3
from pathlib import Path

CATALOG_PATH = Path(os.environ.get("BLE_CLASSROOM_DB_PATH", str(Path(__file__).resolve().parent / "classrooms.db")))

def init_catalog():
    with sqlite3.connect(CATALOG_PATH) as con:
        con.execute("CREATE TABLE IF NOT EXISTS classrooms (classroom_id TEXT PRIMARY KEY, classroom_name TEXT NOT NULL, beacon_name TEXT NOT NULL, beacon_address TEXT NOT NULL)")
        con.executemany("INSERT OR IGNORE INTO classrooms VALUES (?,?,?,?)",
                        [(str(n), str(n), "MOMENTUM 4", "80:C3:BA:66:21:26") for n in range(10001, 10011)])

def list_classrooms():
    with sqlite3.connect(CATALOG_PATH) as con:
        con.row_factory = sqlite3.Row
        return [dict(r) for r in con.execute("SELECT * FROM classrooms ORDER BY classroom_id")]

def get_classroom(classroom_id):
    with sqlite3.connect(CATALOG_PATH) as con:
        con.row_factory = sqlite3.Row
        row = con.execute("SELECT * FROM classrooms WHERE classroom_id=?", (classroom_id,)).fetchone()
    if row is None:
        return None
    room = dict(row)
    room["beacon_address"] = room["beacon_address"].upper()
    if not re.fullmatch(r"(?:[0-9A-F]{2}:){5}[0-9A-F]{2}", room["beacon_address"]):
        raise ValueError("강의실 DB의 MAC 주소 형식을 확인하세요")
    return room

if __name__ == "__main__":
    init_catalog()
