"""Tests for cleaning, similarity, and the Flask routes.

The small table in sample_frame() is handwritten so the tests can check the
logic without pretending it is the real music dataset.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data_processing import (
    DEFAULT_DATA_PATH,
    FEATURE_COLUMNS,
    DatasetError,
    clean_songs,
    feature_summary,
    load_raw_songs,
    scale_features,
)
from recommendation import SongNotFoundError, SongRecommender


def sample_frame():
    columns = [
        "track_id",
        "track_name",
        "track_artist",
        "track_popularity",
        "playlist_genre",
        "playlist_subgenre",
        *FEATURE_COLUMNS,
    ]
    records = [
        ["s1", "Alpha", "Ada", 80, "pop", "dance", 0.90, 0.80, 0.70, 120.0, 0.10, 0.00, -5.0],
        ["s2", "Alpha Twin", "Bea", 60, "pop", "dance", 0.90, 0.80, 0.70, 120.0, 0.10, 0.00, -5.0],
        ["s3", "Distant", "Cy", 40, "rock", "album", 0.10, 0.15, 0.20, 60.0, 0.95, 0.90, -28.0],
        ["s4", "Middle", "Dee", 70, "r&b", "neo", 0.55, 0.50, 0.45, 100.0, 0.40, 0.10, -10.0],
        # Same track id on another playlist, with a lower popularity. Should be dropped.
        ["s1", "Alpha", "Ada", 10, "edm", "electro", 0.10, 0.10, 0.10, 90.0, 0.20, 0.20, -12.0],
        # Same title and artist, different id and lower popularity. Should be dropped.
        ["s5", "alpha", "ada", 20, "latin", "other", 0.20, 0.20, 0.20, 80.0, 0.30, 0.30, -14.0],
        # Missing title, and a tempo of 0. Both should be dropped.
        ["s6", None, "Eve", 30, "pop", "dance", 0.40, 0.40, 0.40, 110.0, 0.20, 0.20, -8.0],
        ["s7", "No Tempo", "Finn", 30, "pop", "dance", 0.40, 0.40, 0.40, 0.0, 0.20, 0.20, -8.0],
    ]
    return pd.DataFrame(records, columns=columns)


def fitted_recommender(frame=None):
    songs = clean_songs(frame if frame is not None else sample_frame())
    matrix, _scaler, ranges = scale_features(songs)
    return songs, SongRecommender(songs, matrix, ranges)


def test_clean_songs_drops_bad_rows_and_duplicates():
    songs = clean_songs(sample_frame())
    assert songs["track_id"].tolist() == ["s1", "s4", "s2", "s3"]
    alpha = songs.loc[songs["track_id"] == "s1"].iloc[0]
    assert alpha["playlist_genre"] == "pop"
    assert alpha["danceability"] == pytest.approx(0.90)
    assert "s5" not in set(songs["track_id"])
    assert "s7" not in set(songs["track_id"])


def test_clean_songs_rejects_missing_columns():
    broken = sample_frame().drop(columns=["loudness"])
    with pytest.raises(DatasetError, match="loudness"):
        clean_songs(broken)


def test_scale_features_centers_columns():
    songs, _recommender = fitted_recommender()
    matrix, _scaler, ranges = scale_features(songs)
    assert matrix.shape == (len(songs), len(FEATURE_COLUMNS))
    assert np.allclose(matrix.mean(axis=0), 0, atol=1e-8)
    assert np.allclose(matrix.std(axis=0), 1, atol=1e-8)
    assert set(ranges) == set(FEATURE_COLUMNS)


def test_recommender_finds_the_similar_song():
    _songs, recommender = fitted_recommender()
    matches = recommender.search("alpha")
    assert [song["track_name"] for song in matches] == ["Alpha", "Alpha Twin"]

    result = recommender.recommend("s1", n=2)
    assert result["selected"]["track_name"] == "Alpha"
    assert result["selected"]["track_id"] not in {
        song["track_id"] for song in result["recommendations"]
    }
    assert result["recommendations"][0]["track_id"] == "s2"
    assert result["recommendations"][0]["similarity"] == pytest.approx(1.0)
    similarities = [song["similarity"] for song in result["recommendations"]]
    assert similarities == sorted(similarities, reverse=True)
    assert set(result["selected"]["features"]) == set(FEATURE_COLUMNS)


def test_recommender_rejects_bad_input():
    _songs, recommender = fitted_recommender()
    with pytest.raises(ValueError, match="2 characters"):
        recommender.search("a")
    with pytest.raises(SongNotFoundError):
        recommender.recommend("missing-id")
    with pytest.raises(ValueError, match="between 1 and 20"):
        recommender.recommend("s1", n=0)


def test_missing_dataset_file(tmp_path):
    missing = tmp_path / "no-such-file.csv"
    with pytest.raises(DatasetError, match="download_data.py"):
        load_raw_songs(missing)
    assert not Path(missing).exists()


@pytest.fixture
def client(tmp_path):
    import app as app_module

    app_module.reset_state()
    songs = clean_songs(sample_frame())
    matrix, _scaler, ranges = scale_features(songs)
    app_module._state["recommender"] = SongRecommender(songs, matrix, ranges)
    app_module._state["summary"] = feature_summary(songs)
    app_module._state["error"] = None
    app_module.app.config["DB_PATH"] = str(tmp_path / "test.db")
    app_module.app.config["TESTING"] = True
    yield app_module.app.test_client()
    app_module.reset_state()


def test_api_search_recommend_and_history(client):
    page = client.get("/")
    assert page.status_code == 200
    assert b"Music Recommendation" in page.data

    short = client.get("/api/search?q=a")
    assert short.status_code == 400

    found = client.get("/api/search?q=alpha")
    assert found.status_code == 200
    assert found.get_json()["results"][0]["track_id"] == "s1"

    missing = client.get("/api/recommend?track_id=does-not-exist")
    assert missing.status_code == 404

    recommended = client.get("/api/recommend?track_id=s1&n=2&q=alpha")
    assert recommended.status_code == 200
    body = recommended.get_json()
    assert body["recommendations"][0]["track_name"] == "Alpha Twin"
    assert len(body["feature_info"]) == 7

    overview = client.get("/api/overview")
    assert overview.status_code == 200
    correlation = overview.get_json()["correlation"]
    assert len(correlation["matrix"]) == 7
    assert len(correlation["matrix"][0]) == 7

    history = client.get("/api/history")
    assert history.status_code == 200
    saved = history.get_json()["history"]
    assert saved[0]["track_name"] == "Alpha"
    assert saved[0]["query"] == "alpha"
    assert saved[0]["recommendations"][0]["track_name"] == "Alpha Twin"


def test_api_missing_dataset(monkeypatch):
    import app as app_module

    app_module.reset_state()

    def missing(_path=None):
        raise DatasetError(
            "Dataset file not found at data/spotify_songs.csv. "
            "From the project folder, run: python download_data.py"
        )

    monkeypatch.setattr(app_module, "load_raw_songs", missing)
    test_client = app_module.app.test_client()
    response = test_client.get("/api/search?q=hello")
    assert response.status_code == 500
    assert "download_data.py" in response.get_json()["error"]

    page = test_client.get("/")
    assert page.status_code == 200
    assert b"download_data.py" in page.data
    app_module.reset_state()


@pytest.mark.skipif(not DEFAULT_DATA_PATH.is_file(), reason="dataset file was not downloaded")
def test_public_dataset_pipeline():
    raw = load_raw_songs()
    songs = clean_songs(raw)
    assert len(raw) > len(songs)
    assert 20000 < len(songs) < 30000
    assert songs["track_id"].is_unique
    assert not songs[FEATURE_COLUMNS].isna().any().any()

    matrix, _scaler, ranges = scale_features(songs)
    assert np.allclose(matrix.mean(axis=0), 0, atol=1e-6)
    assert np.allclose(matrix.std(axis=0), 1, atol=1e-6)

    summary = feature_summary(songs)
    assert len(summary["correlation"]["matrix"]) == 7
    assert len(summary["sample_points"]) == 400

    recommender = SongRecommender(songs, matrix, ranges)
    matches = recommender.search("blinding lights", limit=5)
    assert matches
    assert matches[0]["track_name"].lower() == "blinding lights"

    result = recommender.recommend(matches[0]["track_id"], n=5)
    assert len(result["recommendations"]) == 5
    assert all(song["track_id"] != matches[0]["track_id"] for song in result["recommendations"])
    similarities = [song["similarity"] for song in result["recommendations"]]
    assert similarities == sorted(similarities, reverse=True)
    assert ranges["tempo"]["max"] > ranges["tempo"]["min"]
