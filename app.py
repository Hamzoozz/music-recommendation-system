"""Flask app: search songs, show features, and recommend similar tracks."""

import os
import sqlite3

from flask import Flask, jsonify, render_template, request

from data_processing import (
    FEATURE_INFO,
    DatasetError,
    clean_songs,
    feature_summary,
    load_raw_songs,
    scale_features,
)
from database import recent_history, save_recommendation
from recommendation import SongNotFoundError, SongRecommender

app = Flask(__name__)
app.config["DB_PATH"] = os.environ.get("APP_DB_PATH")

_state = {
    "recommender": None,
    "summary": None,
    "error": None,
}


def reset_state():
    """Clear the cached dataset. Tests use this between cases."""
    _state["recommender"] = None
    _state["summary"] = None
    _state["error"] = None


def db_path():
    return app.config.get("DB_PATH")


def load_application_data():
    if _state["recommender"] is not None or _state["error"] is not None:
        return
    try:
        raw = load_raw_songs()
        songs = clean_songs(raw)
        matrix, _scaler, ranges = scale_features(songs)
        summary = feature_summary(songs)
        summary["raw_count"] = int(len(raw))
        _state["recommender"] = SongRecommender(songs, matrix, ranges)
        _state["summary"] = summary
    except DatasetError as exc:
        _state["error"] = str(exc)


def get_recommender():
    load_application_data()
    if _state["error"]:
        raise DatasetError(_state["error"])
    return _state["recommender"]


def parse_bounded_int(value, name, default, low, high):
    if value is None or value == "":
        number = default
    else:
        try:
            number = int(value)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be a whole number.") from None
    if number < low or number > high:
        raise ValueError(f"{name} must be between {low} and {high}.")
    return number


@app.route("/")
def index():
    load_application_data()
    summary = _state["summary"] or {}
    return render_template(
        "index.html",
        dataset_error=_state["error"],
        song_count=summary.get("song_count"),
        raw_count=summary.get("raw_count"),
    )


@app.route("/api/search")
def search():
    try:
        limit = parse_bounded_int(request.args.get("limit"), "limit", 12, 1, 50)
        results = get_recommender().search(request.args.get("q", ""), limit=limit)
    except DatasetError as exc:
        return jsonify({"error": str(exc)}), 500
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"results": results})


@app.route("/api/recommend")
def recommend():
    track_id = request.args.get("track_id", "").strip()
    if not track_id:
        return jsonify({"error": "Choose a song before asking for recommendations."}), 400
    try:
        count = parse_bounded_int(request.args.get("n"), "n", 8, 1, 20)
        payload = get_recommender().recommend(track_id, n=count)
    except DatasetError as exc:
        return jsonify({"error": str(exc)}), 500
    except SongNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    save_recommendation(
        payload["selected"],
        payload["recommendations"],
        query=request.args.get("q", "").strip(),
        db_path=db_path(),
    )
    payload["feature_info"] = FEATURE_INFO
    return jsonify(payload)


@app.route("/api/overview")
def overview():
    load_application_data()
    if _state["error"]:
        return jsonify({"error": _state["error"]}), 500
    return jsonify(_state["summary"])


@app.route("/api/history")
def history():
    try:
        items = recent_history(limit=8, db_path=db_path())
    except sqlite3.Error as exc:
        return jsonify({"error": f"Could not read recommendation history: {exc}"}), 500
    return jsonify({"history": items})


if __name__ == "__main__":
    load_application_data()
    if _state["error"]:
        print("Warning:", _state["error"])
    else:
        summary = _state["summary"]
        print(
            f"Loaded {summary['song_count']} songs "
            f"from {summary['raw_count']} rows."
        )
    port = int(os.environ.get("PORT", "5000"))
    app.run(debug=True, port=port)
