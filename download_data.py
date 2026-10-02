"""Download the public Spotify songs CSV this project uses."""

import urllib.request
from pathlib import Path

# TidyTuesday, 2020-01-21. The same file is documented in that week's readme.
DATA_URL = (
    "https://raw.githubusercontent.com/rfordatascience/tidytuesday/"
    "master/data/2020/2020-01-21/spotify_songs.csv"
)
DESTINATION = Path(__file__).resolve().parent / "data" / "spotify_songs.csv"


def main():
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    if DESTINATION.exists() and DESTINATION.stat().st_size > 0:
        print(f"Dataset already present: {DESTINATION}")
        print(f"Size: {DESTINATION.stat().st_size} bytes")
        return

    print(f"Downloading {DATA_URL}")
    urllib.request.urlretrieve(DATA_URL, DESTINATION)
    print(f"Saved {DESTINATION} ({DESTINATION.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
