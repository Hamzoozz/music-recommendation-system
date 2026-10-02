# Music Recommendation & Analysis System

A small Flask app that cleans a public Spotify songs dataset, compares songs by audio features, and recommends the nearest tracks. It is a course-style project: one Python app, one SQLite file, and a plain HTML page.

The recommender does not predict whether someone will like a song. It also does not have an accuracy score, because this dataset has no ground-truth labels for "similar song."

## Setup

Use Python 3.11 or newer. These steps were run with Python 3.14 on Windows.

From this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python download_data.py
python -m pytest
python app.py
```

If PowerShell blocks `Activate.ps1`, you can call the virtual environment directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe download_data.py
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe app.py
```

On macOS or Linux, use `python3 -m venv .venv`, then `source .venv/bin/activate`, and the same `pip` / `python` commands.

Then open [http://127.0.0.1:5000](http://127.0.0.1:5000).

`app.py` runs with Flask's debugger on so tracebacks are visible while you work. Use it on your own machine only. If port 5000 is busy:

```powershell
$env:PORT = "5001"
python app.py
```

## Dataset

The file is `data/spotify_songs.csv` from TidyTuesday, week of 2020-01-21:

- Data dictionary: <https://github.com/rfordatascience/tidytuesday/blob/main/data/2020/2020-01-21/readme.md>
- Direct download: <https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-01-21/spotify_songs.csv>

The tracks were collected from the Spotify Web API with the `spotifyr` R package. This project does not call Spotify. If the CSV is missing, run `python download_data.py`. The file is about 8 MB. The home page also shows an error, with the same instruction, if the file is not there.

The raw file has **32,833** rows. Each row is a track inside a playlist, so the same song appears more than once. After cleaning, **26,158** songs remain.

Cleaning, in `data_processing.py`:

- Keep the id, title, artist, popularity, playlist genre, playlist subgenre, and the seven audio features below.
- Drop rows with a missing id, title, artist, or feature. This file has 5 rows without a title or artist.
- Drop rows with tempo of 0. This file has 1 such row. A tempo of 0 is not a usable BPM value.
- If the same `track_id` appears in several playlists, keep the row with the highest `track_popularity`.
- If the same title and artist appear under a second id, keep the most popular row.
- Fill a blank playlist genre or subgenre with `Unknown`.

Features used for similarity:

| Feature | What Spotify's number means | Typical range |
| --- | --- | --- |
| danceability | How suitable the track is for dancing | 0 to 1 |
| energy | How intense or active the track feels | 0 to 1 |
| valence | Musical positiveness, from darker to happier | 0 to 1 |
| tempo | Speed in beats per minute | roughly 50 to 200 |
| acousticness | Confidence that the track is acoustic | 0 to 1 |
| instrumentalness | Confidence that the track has no vocals | 0 to 1 |
| loudness | Average loudness | decibels, usually -60 to 0 |

`playlist_genre` is the genre of the playlist the track was pulled from. It is shown as context. It is not an official genre label, and it is not an input to the similarity math. Popularity is used only to choose which duplicate row to keep and to sort search hits. It is not a model feature.

## How recommendations work

This is content-based filtering. There are no user accounts and no listening histories.

1. Each song becomes a vector of the seven features.
2. `StandardScaler` replaces each feature with a z-score across songs: `(value - column mean) / column standard deviation`. Tempo and loudness are on much larger numeric scales than danceability. Without this step, those two columns would dominate the comparison. Scaling is fit on the whole cleaned table when the app starts.
3. For the song you select, scikit-learn's `NearestNeighbors` scans the other songs with cosine distance. A brute-force scan is enough for about 26,000 songs and 7 numbers. The closest match would otherwise be the selected song itself, so the app asks for one extra neighbor and drops that row.
4. Cosine similarity is `1 - cosine distance`, which is the same as:

```text
similarity = dot(x, y) / (length(x) * length(y))
```

`x` and `y` are the standardized vectors. The score is about 1 when the vectors point the same way, about 0 when they are perpendicular, and negative when they point apart. The page rounds it to three decimals.

Column scaling is not the same thing as correlation between two songs. Correlation would center each song's own seven numbers. This project centers each feature across songs, then compares directions.

In this dataset the nearest songs often have cosine similarity above 0.95. Lots of tracks sit in a similar part of the feature space (for example, somewhat loud, not acoustic, not instrumental). The ranking is the useful part. The score is not a percent correct, and a 0.99 result is not "99% accuracy."

The charts on the page are descriptive:

- A Pearson correlation table for the seven features. In this file, energy and loudness move together (about 0.69), and energy and acousticness move in opposite directions (about -0.55).
- A scatter plot of energy vs. valence for a fixed sample of 400 songs (`random_state=21`). Darker dots are more danceable. The ring is the song you selected.

The bars on a selected song are scaled from that feature's minimum to its maximum so different units fit on one chart. The model does not use those bar lengths. It uses the z-scores.

## What is stored in SQLite

`data/app.db` stores recommendation history only: the search text, the song you picked, the recommended titles, and a UTC timestamp. The song catalog stays in the CSV because that file is the dataset and it is easy to reload. `data/app.db` is created the first time you select a song. It is listed in `.gitignore`.

## Project layout

| File | Role |
| --- | --- |
| `app.py` | Flask routes for the page, search, recommendations, charts, and history |
| `data_processing.py` | Load the CSV, clean it, standardize features, build chart data |
| `recommendation.py` | Search by title or artist, then nearest neighbors by cosine similarity |
| `database.py` | Create the SQLite file and read or write history |
| `download_data.py` | Download the CSV if it is not already in `data/` |
| `templates/index.html` | The one page |
| `static/style.css` | Page layout |
| `static/app.js` | Search, song details, charts, and history requests |
| `tests/test_core.py` | Checks for cleaning, similarity, errors, and the API |
| `data/spotify_songs.csv` | The public dataset |

## Tests

```powershell
python -m pytest
```

The tests use a tiny handwritten table for the logic checks. One test loads the real CSV if `data/spotify_songs.csv` is present. Nothing in the tests is a fake copy of the Spotify dataset.

## Errors you can hit

- Missing CSV: the page and `/api/search` explain that you should run `python download_data.py`.
- Search shorter than 2 characters: the API returns 400.
- A track id that is not in the cleaned table: the API returns 404.
- A recommendation count outside 1–20: the API returns 400.
- A CSV with the wrong columns: loading fails with a message naming the missing columns.

Restart the app after you add the CSV. The missing-file error is cached until the process starts again.

## Limitations

- The catalog is playlist tracks gathered around early 2020, not the full Spotify catalog. A song that is not in the file cannot be selected or recommended.
- "Similar" means similar on these seven Spotify audio features. Lyrics, melody, artist, and culture are ignored. Two songs can be nearest neighbors and still sound unrelated to a person.
- Playlist genre can be wrong for the song, because it describes the playlist.
- Duplicate recordings sometimes survive when the title text is slightly different, such as a remix name.
- There is no train/test split and no accuracy, precision, or recall. Those numbers would need labeled pairs of songs that people judged to be similar. This file does not have that.
- The nearest-neighbor scan and the scaler are fit on all of the cleaned songs. That is reasonable for a similarity index. It would be leakage if this were a supervised classifier, which it is not.
- History is local and unauthenticated. Do not expose the Flask debug server on a public network.

## Possible later additions

- Let the user down-weight same-artist results, or compare against a genre filter, and show how the list changes.
- Add a second distance (Euclidean on the same z-scores) and compare the two lists.
- Hold out a handful of songs you label by hand and report how often the top 5 agree with you. Keep the sample size honest.
- Add speechiness, liveness, and duration, then check whether the neighbors actually change.
- Cache the scaled matrix to disk so startup does not reread the CSV every time.

## GitHub

The `.gitignore` already skips the virtual environment, `__pycache__`, and `data/app.db`. The CSV is about 8 MB, which is small enough to commit if you want the project to run without a download.

```powershell
git init
git add .
git status
git commit -m "Add music recommendation and analysis app"
```

Create an empty repository on GitHub, then push this folder. Do not commit `.venv` or `data/app.db`.
