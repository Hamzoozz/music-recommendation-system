"""Load, clean, and scale the Spotify songs CSV."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# These seven columns are the ones the recommender compares.
FEATURE_COLUMNS = [
    "danceability",
    "energy",
    "valence",
    "tempo",
    "acousticness",
    "instrumentalness",
    "loudness",
]

# Labels sent to the page so the Python list stays the source of truth.
FEATURE_INFO = [
    {"key": "danceability", "label": "Danceability", "short": "Dance", "hint": "0 to 1"},
    {"key": "energy", "label": "Energy", "short": "Energy", "hint": "0 to 1"},
    {"key": "valence", "label": "Valence", "short": "Valence", "hint": "0 (darker) to 1 (happier)"},
    {"key": "tempo", "label": "Tempo", "short": "Tempo", "hint": "beats per minute"},
    {"key": "acousticness", "label": "Acousticness", "short": "Acoustic", "hint": "0 to 1"},
    {
        "key": "instrumentalness",
        "label": "Instrumentalness",
        "short": "Instr.",
        "hint": "0 (more vocals) to 1 (more instrumental)",
    },
    {"key": "loudness", "label": "Loudness", "short": "Loud", "hint": "decibels, usually negative"},
]

METADATA_COLUMNS = [
    "track_id",
    "track_name",
    "track_artist",
    "track_popularity",
    "playlist_genre",
    "playlist_subgenre",
]

REQUIRED_COLUMNS = METADATA_COLUMNS + FEATURE_COLUMNS

DEFAULT_DATA_PATH = Path(__file__).resolve().parent / "data" / "spotify_songs.csv"


class DatasetError(Exception):
    """Raised when the CSV is missing or cannot be used."""


def load_raw_songs(path=None):
    csv_path = Path(path) if path else DEFAULT_DATA_PATH
    if not csv_path.is_file():
        raise DatasetError(
            f"Dataset file not found at {csv_path}. "
            "From the project folder, run: python download_data.py"
        )
    try:
        return pd.read_csv(csv_path)
    except (OSError, UnicodeDecodeError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise DatasetError(f"Could not read the dataset at {csv_path}: {exc}") from exc


def clean_songs(df):
    """Drop unusable rows and keep one row per song title and artist.

    The source file lists a track again for every playlist it appeared in.
    The same title can also show up under a second track id. Popularity is
    used only to decide which copy to keep, not to measure similarity.
    """
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise DatasetError(
            "Dataset is missing expected columns: "
            + ", ".join(missing)
            + ". Use the TidyTuesday spotify_songs.csv file."
        )

    cleaned = df[REQUIRED_COLUMNS].copy()
    text_columns = [
        "track_id",
        "track_name",
        "track_artist",
        "playlist_genre",
        "playlist_subgenre",
    ]
    for column in text_columns:
        cleaned[column] = cleaned[column].astype("string").str.strip()

    cleaned["playlist_genre"] = cleaned["playlist_genre"].fillna("Unknown")
    cleaned["playlist_subgenre"] = cleaned["playlist_subgenre"].fillna("Unknown")
    cleaned.loc[cleaned["playlist_genre"].str.len().fillna(0) == 0, "playlist_genre"] = "Unknown"
    cleaned.loc[
        cleaned["playlist_subgenre"].str.len().fillna(0) == 0, "playlist_subgenre"
    ] = "Unknown"

    cleaned["track_popularity"] = (
        pd.to_numeric(cleaned["track_popularity"], errors="coerce").fillna(0)
    )
    for column in FEATURE_COLUMNS:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    cleaned = cleaned.dropna(
        subset=["track_id", "track_name", "track_artist", *FEATURE_COLUMNS]
    ).copy()
    usable = (
        (cleaned["track_id"].str.len() > 0)
        & (cleaned["track_name"].str.len() > 0)
        & (cleaned["track_artist"].str.len() > 0)
        & (cleaned["tempo"] > 0)
    )
    cleaned = cleaned.loc[usable].copy()

    cleaned["_name_key"] = cleaned["track_name"].str.lower()
    cleaned["_artist_key"] = cleaned["track_artist"].str.lower()
    cleaned = cleaned.sort_values(
        ["track_popularity", "track_name"],
        ascending=[False, True],
        kind="mergesort",
    )
    cleaned = cleaned.drop_duplicates(subset=["track_id"], keep="first")
    cleaned = cleaned.drop_duplicates(subset=["_name_key", "_artist_key"], keep="first")
    cleaned = cleaned.drop(columns=["_name_key", "_artist_key"]).reset_index(drop=True)

    if cleaned.empty:
        raise DatasetError("The dataset has no usable songs after cleaning.")
    return cleaned


def scale_features(df):
    """Standardize each feature across songs.

    Tempo is around 50-200 BPM and loudness is negative dB, while the other
    features are already between 0 and 1. Without scaling, tempo would
    dominate the distance. The min/max ranges are only for drawing bars.
    """
    scaler = StandardScaler()
    matrix = scaler.fit_transform(df[FEATURE_COLUMNS].to_numpy(dtype=float))
    ranges = {}
    for column in FEATURE_COLUMNS:
        low = float(df[column].min())
        high = float(df[column].max())
        ranges[column] = {"min": low, "max": high}
    return matrix, scaler, ranges


def feature_summary(df, sample_size=400):
    """Correlation table and a fixed scatter sample for the overview charts."""
    correlation = df[FEATURE_COLUMNS].corr()
    matrix = []
    for row in correlation.to_numpy():
        matrix.append(
            [None if pd.isna(value) else round(float(value), 3) for value in row]
        )

    sample_n = min(sample_size, len(df))
    sample = df.sample(n=sample_n, random_state=21)
    point_columns = ["energy", "valence", "danceability", "acousticness"]
    points = []
    for record in sample[point_columns].to_dict("records"):
        points.append({key: round(float(value), 4) for key, value in record.items()})

    means = {
        column: round(float(df[column].mean()), 4) for column in FEATURE_COLUMNS
    }
    return {
        "song_count": int(len(df)),
        "feature_info": FEATURE_INFO,
        "means": means,
        "correlation": {"columns": FEATURE_COLUMNS, "matrix": matrix},
        "sample_points": points,
    }


def display_scale(value, low, high):
    """Map a raw feature onto 0-1 using the dataset min and max."""
    width = high - low
    if width == 0:
        return 0.0
    scaled = (value - low) / width
    return float(np.clip(scaled, 0.0, 1.0))
