"""Find songs with similar audio features."""

import numpy as np
from sklearn.neighbors import NearestNeighbors

from data_processing import FEATURE_COLUMNS, display_scale


class SongNotFoundError(Exception):
    """Raised when a track id is not in the cleaned table."""


class SongRecommender:
    def __init__(self, songs, feature_matrix, display_ranges):
        if len(songs) == 0:
            raise ValueError("Cannot build a recommender with an empty song table.")
        if len(songs) != len(feature_matrix):
            raise ValueError("The song table and feature matrix must have the same length.")

        self.songs = songs.reset_index(drop=True)
        self.feature_matrix = np.asarray(feature_matrix, dtype=float)
        self.display_ranges = display_ranges
        if self.songs["track_id"].duplicated().any():
            raise ValueError("track_id values must be unique.")

        self._id_to_index = {
            track_id: index for index, track_id in enumerate(self.songs["track_id"].tolist())
        }
        self._names = [str(name).lower() for name in self.songs["track_name"].tolist()]
        self._artists = [str(artist).lower() for artist in self.songs["track_artist"].tolist()]
        self._popularity = [int(round(float(value))) for value in self.songs["track_popularity"]]

        # Brute force is enough at this size, and cosine distance needs it.
        # fit() stores the matrix; each query compares one song with the rest.
        self.model = NearestNeighbors(metric="cosine", algorithm="brute")
        self.model.fit(self.feature_matrix)

    def search(self, query, limit=12):
        text = (query or "").strip().lower()
        if len(text) < 2:
            raise ValueError("Enter at least 2 characters.")
        if limit < 1 or limit > 50:
            raise ValueError("The search limit must be between 1 and 50.")

        matches = []
        for index, (name, artist) in enumerate(zip(self._names, self._artists)):
            rank = self._match_rank(name, artist, text)
            if rank is None:
                continue
            matches.append((rank, -self._popularity[index], name, index))

        # Exact title, then title prefix, then artist, then a substring.
        # Popularity only breaks ties. It is not a similarity feature.
        matches.sort()
        return [self._song_brief(item[3]) for item in matches[:limit]]

    def recommend(self, track_id, n=8):
        if track_id not in self._id_to_index:
            raise SongNotFoundError(
                "That song is not in the dataset. Search for a title and choose one from the list."
            )
        if n < 1 or n > 20:
            raise ValueError("The number of recommendations must be between 1 and 20.")

        index = self._id_to_index[track_id]
        neighbor_count = min(n + 1, len(self.songs))
        distances, indices = self.model.kneighbors(
            self.feature_matrix[index].reshape(1, -1),
            n_neighbors=neighbor_count,
        )

        recommendations = []
        for distance, neighbor_index in zip(distances[0], indices[0]):
            neighbor_index = int(neighbor_index)
            if neighbor_index == index:
                continue
            song = self._song_brief(neighbor_index)
            # sklearn's cosine distance is 1 - cosine similarity.
            similarity = 1.0 - float(distance)
            similarity = max(-1.0, min(1.0, similarity))
            song["similarity"] = round(similarity, 4)
            recommendations.append(song)

        return {
            "selected": self.song_detail(track_id),
            "recommendations": recommendations,
        }

    def song_detail(self, track_id):
        if track_id not in self._id_to_index:
            raise SongNotFoundError(
                "That song is not in the dataset. Search for a title and choose one from the list."
            )
        index = self._id_to_index[track_id]
        detail = self._song_brief(index)
        row = self.songs.iloc[index]
        features = {}
        display_features = {}
        for column in FEATURE_COLUMNS:
            raw_value = float(row[column])
            features[column] = round(raw_value, 4)
            bounds = self.display_ranges[column]
            display_features[column] = round(
                display_scale(raw_value, bounds["min"], bounds["max"]), 4
            )
        detail["features"] = features
        detail["display_features"] = display_features
        return detail

    def _song_brief(self, index):
        row = self.songs.iloc[index]
        return {
            "track_id": str(row["track_id"]),
            "track_name": str(row["track_name"]),
            "track_artist": str(row["track_artist"]),
            "playlist_genre": str(row["playlist_genre"]),
            "playlist_subgenre": str(row["playlist_subgenre"]),
            "track_popularity": int(round(float(row["track_popularity"]))),
        }

    @staticmethod
    def _match_rank(name, artist, query):
        if name == query:
            return 0
        if name.startswith(query):
            return 1
        if artist == query:
            return 2
        if artist.startswith(query):
            return 3
        if query in name:
            return 4
        if query in artist:
            return 5
        return None
