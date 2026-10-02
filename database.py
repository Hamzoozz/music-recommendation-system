"""SQLite storage for recommendation history.

The song table itself stays in the CSV. This database only remembers which
songs were selected in the app, on this computer.
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DB_PATH = Path(__file__).resolve().parent / "data" / "app.db"


def connect(db_path=None):
    path = Path(db_path) if db_path else DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection


def init_db(db_path=None):
    with connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS recommendation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT,
                track_id TEXT NOT NULL,
                track_name TEXT NOT NULL,
                track_artist TEXT NOT NULL,
                recommendations_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )


def save_recommendation(selected, recommendations, query="", db_path=None):
    init_db(db_path)
    payload = [
        {
            "track_id": item["track_id"],
            "track_name": item["track_name"],
            "track_artist": item["track_artist"],
            "similarity": item["similarity"],
        }
        for item in recommendations
    ]
    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    with connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO recommendation_history (
                query, track_id, track_name, track_artist,
                recommendations_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                query or "",
                selected["track_id"],
                selected["track_name"],
                selected["track_artist"],
                json.dumps(payload),
                created_at,
            ),
        )


def recent_history(limit=8, db_path=None):
    init_db(db_path)
    with connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT id, query, track_id, track_name, track_artist,
                   recommendations_json, created_at
            FROM recommendation_history
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()

    history = []
    for row in rows:
        history.append(
            {
                "id": row["id"],
                "query": row["query"],
                "track_id": row["track_id"],
                "track_name": row["track_name"],
                "track_artist": row["track_artist"],
                "recommendations": json.loads(row["recommendations_json"]),
                "created_at": row["created_at"],
            }
        )
    return history
