// Song titles are inserted as text. escapeHtml keeps a title from being treated as markup.
const messageEl = document.getElementById("message");
const resultsEl = document.getElementById("results");
const selectedEmpty = document.getElementById("selected-empty");
const selectedBody = document.getElementById("selected-body");
const recsEmpty = document.getElementById("recs-empty");
const recsEl = document.getElementById("recs");
const historyEl = document.getElementById("history");
const heatmapEl = document.getElementById("heatmap");
const scatterCanvas = document.getElementById("scatter");

let featureInfo = [];
let samplePoints = [];
let selectedPoint = null;
let selectedId = "";
let lastQuery = "";

document.getElementById("search-form").addEventListener("submit", onSearch);
resultsEl.addEventListener("click", onPick);
recsEl.addEventListener("click", onPick);
historyEl.addEventListener("click", onPick);
window.addEventListener("resize", () => drawScatter(samplePoints, selectedPoint));

loadOverview();
loadHistory();

async function onSearch(event) {
  event.preventDefault();
  const query = document.getElementById("query").value.trim();
  lastQuery = query;
  clearMessage();
  resultsEl.innerHTML = "";
  try {
    const response = await fetch("/api/search?q=" + encodeURIComponent(query));
    const data = await response.json();
    if (!response.ok) {
      showMessage(data.error || "Search failed.");
      return;
    }
    renderResults(data.results);
  } catch (error) {
    showMessage("Could not reach the server. Is the Flask app running?");
  }
}

function onPick(event) {
  const button = event.target.closest("button[data-id]");
  if (!button) {
    return;
  }
  // History rows remember the search text from when they were saved.
  if (button.closest("#history")) {
    lastQuery = button.dataset.query || "";
  }
  chooseSong(button.dataset.id);
}

async function chooseSong(trackId) {
  const count = document.getElementById("count").value;
  clearMessage();
  selectedId = trackId;
  markActive(trackId);
  try {
    const url =
      "/api/recommend?track_id=" +
      encodeURIComponent(trackId) +
      "&n=" +
      encodeURIComponent(count) +
      "&q=" +
      encodeURIComponent(lastQuery);
    const response = await fetch(url);
    const data = await response.json();
    if (!response.ok) {
      showMessage(data.error || "Could not load that song.");
      return;
    }
    if (data.feature_info) {
      featureInfo = data.feature_info;
    }
    renderSelected(data.selected);
    renderRecommendations(data.recommendations);
    selectedPoint = {
      energy: data.selected.features.energy,
      valence: data.selected.features.valence,
    };
    drawScatter(samplePoints, selectedPoint);
    loadHistory();
  } catch (error) {
    showMessage("Could not reach the server. Is the Flask app running?");
  }
}

function renderResults(results) {
  if (!results.length) {
    resultsEl.innerHTML = '<li class="empty">No songs matched that search.</li>';
    return;
  }
  resultsEl.innerHTML = results.map((song) => songButton(song)).join("");
  markActive(selectedId);
}

function renderSelected(song) {
  selectedEmpty.hidden = true;
  selectedBody.hidden = false;
  const rows = featureInfo.map((info) => {
    const raw = song.features[info.key];
    const width = Math.round(song.display_features[info.key] * 100);
    return `
      <div class="feature-row">
        <span class="feature-label" title="${escapeHtml(info.hint)}">${escapeHtml(info.label)}</span>
        <span class="bar"><span class="bar-fill" style="width: ${width}%"></span></span>
        <span class="feature-value">${escapeHtml(formatFeature(info.key, raw))}</span>
      </div>`;
  }).join("");

  selectedBody.innerHTML = `
    <h3 class="song-heading">${escapeHtml(song.track_name)}</h3>
    <p>${escapeHtml(song.track_artist)}</p>
    <p class="hint">Playlist context: ${escapeHtml(song.playlist_genre)} / ${escapeHtml(song.playlist_subgenre)}</p>
    <p class="hint">Popularity in the dataset: ${song.track_popularity}. Not used for similarity.</p>
    <div class="feature-list">${rows}</div>
    <p class="hint">Bar length maps each raw value between the minimum and maximum in this dataset, so different units can share one chart. The model uses z-scores, not these bar lengths.</p>
  `;
}

function renderRecommendations(recommendations) {
  if (!recommendations.length) {
    recsEmpty.hidden = false;
    recsEmpty.textContent = "No other songs were available to recommend.";
    recsEl.innerHTML = "";
    return;
  }
  recsEmpty.hidden = true;
  recsEl.innerHTML = recommendations.map((song) => `
    <li>
      <button type="button" data-id="${escapeHtml(song.track_id)}">
        <span class="song-title">${escapeHtml(song.track_name)}</span>
        <span class="song-meta">${escapeHtml(song.track_artist)} · ${escapeHtml(song.playlist_genre)}</span>
        <span class="similarity">cosine similarity ${Number(song.similarity).toFixed(3)}</span>
      </button>
    </li>
  `).join("");
}

async function loadOverview() {
  try {
    const response = await fetch("/api/overview");
    const data = await response.json();
    if (!response.ok) {
      heatmapEl.textContent = data.error || "Could not load the charts.";
      return;
    }
    featureInfo = data.feature_info || [];
    samplePoints = data.sample_points || [];
    renderHeatmap(data.correlation);
    drawScatter(samplePoints, selectedPoint);
  } catch (error) {
    heatmapEl.textContent = "Could not load the charts.";
  }
}

function renderHeatmap(correlation) {
  if (!correlation) {
    return;
  }
  const shorts = {};
  featureInfo.forEach((info) => {
    shorts[info.key] = info.short;
  });
  const headers = correlation.columns.map((key) => `<th title="${escapeHtml(key)}">${escapeHtml(shorts[key] || key)}</th>`).join("");
  const body = correlation.matrix.map((row, rowIndex) => {
    const label = shorts[correlation.columns[rowIndex]] || correlation.columns[rowIndex];
    const cells = row.map((value) => {
      const text = value === null ? "" : Number(value).toFixed(2);
      const amount = value === null ? 0 : Math.min(1, Math.abs(Number(value)));
      const textColor = amount > 0.55 ? "#ffffff" : "#243038";
      return `<td style="background:${corrColor(value)};color:${textColor}" title="${escapeHtml(text)}">${escapeHtml(text)}</td>`;
    }).join("");
    return `<tr><th>${escapeHtml(label)}</th>${cells}</tr>`;
  }).join("");
  heatmapEl.innerHTML = `<table class="heatmap"><thead><tr><th></th>${headers}</tr></thead><tbody>${body}</tbody></table>`;
}

function corrColor(value) {
  if (value === null || Number.isNaN(Number(value))) {
    return "#f3f0ea";
  }
  const amount = Math.min(1, Math.abs(Number(value)));
  const toward = Number(value) >= 0 ? [15, 110, 107] : [158, 74, 58];
  const red = Math.round(255 + (toward[0] - 255) * amount);
  const green = Math.round(255 + (toward[1] - 255) * amount);
  const blue = Math.round(255 + (toward[2] - 255) * amount);
  return `rgb(${red}, ${green}, ${blue})`;
}

function drawScatter(points, selected) {
  const canvas = scatterCanvas;
  const width = canvas.clientWidth || 640;
  const height = 340;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.floor(width * dpr);
  canvas.height = Math.floor(height * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);

  const pad = { left: 46, right: 16, top: 16, bottom: 36 };
  const plotWidth = width - pad.left - pad.right;
  const plotHeight = height - pad.top - pad.bottom;

  function xOf(energy) {
    return pad.left + clamp01(energy) * plotWidth;
  }
  function yOf(valence) {
    return pad.top + (1 - clamp01(valence)) * plotHeight;
  }

  ctx.strokeStyle = "#d9d0c3";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.left, pad.top);
  ctx.lineTo(pad.left, pad.top + plotHeight);
  ctx.lineTo(pad.left + plotWidth, pad.top + plotHeight);
  ctx.stroke();

  ctx.fillStyle = "#5d6b73";
  ctx.font = "12px Segoe UI, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText("Energy", pad.left + plotWidth / 2, height - 8);
  ctx.save();
  ctx.translate(14, pad.top + plotHeight / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("Valence", 0, 0);
  ctx.restore();

  ctx.textAlign = "center";
  ["0", "0.5", "1"].forEach((tick) => {
    const value = Number(tick);
    ctx.fillText(tick, xOf(value), pad.top + plotHeight + 16);
  });

  points.forEach((point) => {
    const shade = Math.round(180 - clamp01(point.danceability) * 140);
    ctx.fillStyle = `rgb(${shade}, ${shade - 10}, ${shade - 20})`;
    ctx.beginPath();
    ctx.arc(xOf(point.energy), yOf(point.valence), 3, 0, Math.PI * 2);
    ctx.fill();
  });

  if (selected) {
    ctx.strokeStyle = "#0f6e6b";
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.arc(xOf(selected.energy), yOf(selected.valence), 7, 0, Math.PI * 2);
    ctx.stroke();
  }
}

async function loadHistory() {
  try {
    const response = await fetch("/api/history");
    const data = await response.json();
    if (!response.ok) {
      historyEl.innerHTML = `<li class="empty">${escapeHtml(data.error || "Could not read history.")}</li>`;
      return;
    }
    if (!data.history.length) {
      historyEl.innerHTML = '<li class="empty">No recommendations have been saved yet.</li>';
      return;
    }
    historyEl.innerHTML = data.history.map((item) => {
      const top = item.recommendations.slice(0, 3).map((song) => song.track_name).join(", ");
      return `
        <li>
          <button type="button" data-id="${escapeHtml(item.track_id)}" data-query="${escapeHtml(item.query)}">
            <span class="song-title">${escapeHtml(item.track_name)} — ${escapeHtml(item.track_artist)}</span>
            <span class="song-meta">${escapeHtml(item.created_at)}${top ? " · " + escapeHtml(top) : ""}</span>
          </button>
        </li>`;
    }).join("");
  } catch (error) {
    historyEl.innerHTML = '<li class="empty">Could not read recommendation history.</li>';
  }
}

function songButton(song) {
  return `
    <li>
      <button type="button" data-id="${escapeHtml(song.track_id)}">
        <span class="song-title">${escapeHtml(song.track_name)}</span>
        <span class="song-meta">${escapeHtml(song.track_artist)} · ${escapeHtml(song.playlist_genre)} / ${escapeHtml(song.playlist_subgenre)}</span>
      </button>
    </li>`;
}

function markActive(trackId) {
  resultsEl.querySelectorAll("button[data-id]").forEach((button) => {
    button.classList.toggle("is-active", button.dataset.id === trackId);
  });
}

function formatFeature(key, value) {
  const number = Number(value);
  if (key === "tempo") {
    return `${Math.round(number)} BPM`;
  }
  if (key === "loudness") {
    return `${number.toFixed(1)} dB`;
  }
  return number.toFixed(3);
}

function showMessage(text) {
  messageEl.hidden = false;
  messageEl.textContent = text;
}

function clearMessage() {
  messageEl.hidden = true;
  messageEl.textContent = "";
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function clamp01(value) {
  return Math.min(1, Math.max(0, Number(value) || 0));
}
