const apiRoot = window.location.origin;
const fluxNames = Array.from({ length: 3197 }, (_, i) => `FLUX.${i + 1}`);
const state = { file: null, results: [], curves: new Map(), actual: false, page: "overview" };
const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

function navigate(page) {
  if (!document.querySelector(`[data-page="${page}"]`)) return;
  state.page = page;
  $$(".page").forEach((el) => el.classList.toggle("active", el.dataset.page === page));
  $$(".nav-link").forEach((el) => el.classList.toggle("active", el.dataset.nav === page));
  $("#main-nav").classList.remove("open");
  window.history.replaceState(null, "", `#${page}`);
  window.scrollTo({ top: 0, behavior: "smooth" });
}

$$(`[data-nav]`).forEach((element) => {
  element.onclick = () => navigate(element.dataset.nav);
});
$("#menu-toggle").onclick = () => {
  const open = $("#main-nav").classList.toggle("open");
  $("#menu-toggle").setAttribute("aria-expanded", String(open));
};
window.addEventListener("hashchange", () => navigate(location.hash.slice(1) || "overview"));
if (location.hash) navigate(location.hash.slice(1));

async function checkApi() {
  const dot = $("#api-dot");
  const label = $("#api-label");
  const workspace = $("#workspace-api");
  try {
    const response = await fetch(`${apiRoot}/health`, { cache: "no-store" });
    if (!response.ok) throw new Error("API unavailable");
    const health = await response.json();
    if (!health.model_loaded) throw new Error("Model artifact missing");
    $(".api-state").classList.add("online");
    $(".api-state").classList.remove("offline");
    dot.style.background = "#65d6af";
    label.textContent = "AstroLens ready";
    workspace.textContent = "AstroLens API connected";
    $(".guide-status i").style.background = "#65d6af";
    $("#analyze-button").disabled = !state.file;
  } catch {
    $(".api-state").classList.add("offline");
    label.textContent = "Model offline";
    workspace.textContent = "Start the Kepler Vision API to analyze";
    $("#analyze-button").disabled = true;
  }
}
checkApi();
setInterval(checkApi, 12000);

const fileInput = $("#csv-file");
const dropZone = $("#drop-zone");
fileInput.addEventListener("change", () => acceptFile(fileInput.files[0]));
dropZone.addEventListener("dragover", (event) => { event.preventDefault(); dropZone.classList.add("dragging"); });
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragging"));
dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropZone.classList.remove("dragging");
  acceptFile(event.dataTransfer.files[0]);
});

function acceptFile(file) {
  if (!file) return;
  clearResults();
  if (!file.name.toLowerCase().endsWith(".csv")) {
    showError("Please choose a CSV file.");
    return;
  }
  state.file = file;
  $("#file-summary").classList.remove("hidden");
  $("#file-name").textContent = file.name;
  $("#file-size").textContent = formatBytes(file.size);
  $("#upload-title").textContent = "CSV file selected";
  $("#upload-hint").textContent = "Choose another file to replace it";
  $("#analysis-status").textContent = "Ready to validate and screen.";
  $("#analyze-button").disabled = false;
  checkApi();
}

// Parse one CSV record, including quoted fields such as "FLUX.1" and
// escaped quotes. The source Kepler CSV files quote every value.
function parseCsvLine(line) {
  const cells = [];
  let value = "";
  let quoted = false;
  for (let i = 0; i < line.length; i++) {
    const char = line[i];
    if (char === '"') {
      if (quoted && line[i + 1] === '"') {
        value += '"';
        i++;
      } else {
        quoted = !quoted;
      }
    } else if (char === "," && !quoted) {
      cells.push(value);
      value = "";
    } else {
      value += char;
    }
  }
  cells.push(value);
  return cells;
}

$("#remove-file").addEventListener("click", () => {
  state.file = null;
  fileInput.value = "";
  $("#file-summary").classList.add("hidden");
  $("#upload-title").textContent = "Drop a CSV file here";
  $("#upload-hint").innerHTML = "or <u>browse your files</u>";
  $("#analysis-status").textContent = "Select a file to begin.";
  $("#analyze-button").disabled = true;
  clearResults();
});

$("#analyze-button").addEventListener("click", async () => {
  if (!state.file) return;
  clearResults();
  hideError();
  $("#analyze-button").disabled = true;
  $("#progress-wrap").classList.remove("hidden");
  $("#analysis-status").textContent = "Screening in progress…";
  state.results = [];
  state.curves = new Map();
  state.actual = false;
  let rowNumber = 0;
  let headers = null;
  let fluxIndexes = [];
  let labelIndex = -1;
  let pending = [];
  const reader = state.file.stream().getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let bytesRead = 0;

  const handleLine = async (rawLine) => {
    const line = rawLine.replace(/\r$/, "");
    if (!line.trim()) return;
    if (!headers) {
      headers = parseCsvLine(line.replace(/^﻿/, "")).map((value) => value.trim());
      const headerIndex = new Map(headers.map((name, index) => [name, index]));
      const missing = fluxNames.filter((name) => !headerIndex.has(name));
      if (missing.length) throw new Error(`This file is missing ${missing.length} flux columns. It needs FLUX.1 through FLUX.3197.`);
      fluxIndexes = fluxNames.map((name) => headerIndex.get(name));
      labelIndex = headerIndex.has("LABEL") ? headerIndex.get("LABEL") : -1;
      state.actual = labelIndex >= 0;
      return;
    }
    const cells = parseCsvLine(line);
    const flux = new Array(fluxIndexes.length);
    for (let i = 0; i < fluxIndexes.length; i++) {
      const value = Number(cells[fluxIndexes[i]]);
      if (!Number.isFinite(value)) throw new Error(`Row ${rowNumber + 2} has a missing or non-numeric flux value.`);
      flux[i] = value;
    }
    let actualLabel = null;
    if (labelIndex >= 0) {
      const sourceLabel = Number(cells[labelIndex]);
      if (sourceLabel !== 1 && sourceLabel !== 2) throw new Error(`Row ${rowNumber + 2} has an invalid LABEL. Expected 1 or 2.`);
      actualLabel = sourceLabel === 2 ? "planet" : "no_planet";
    }
    pending.push({ flux, actualLabel, row: rowNumber });
    rowNumber++;
    if (pending.length >= 128) await sendBatch();
  };

  const sendBatch = async () => {
    if (!pending.length) return;
    const batch = pending;
    pending = [];
    const response = await fetch(`${apiRoot}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ observations: batch.map((item) => ({ flux: item.flux })) }),
    });
    if (!response.ok) {
      const detail = await response.text();
      throw new Error(`Prediction API returned ${response.status}: ${detail.slice(0, 260)}`);
    }
    const output = (await response.json()).predictions;
    output.forEach((prediction, index) => {
      const source = batch[index];
      const item = {
        row: source.row + 1,
        label: prediction.label,
        class_name: prediction.class_name,
        planet_probability: prediction.planet_probability,
        confidence: prediction.confidence,
        actual: source.actualLabel,
      };
      state.results.push(item);
      if (state.curves.size < 30 || (item.class_name === "planet" && state.curves.size < 80)) {
        state.curves.set(source.row, source.flux);
      }
    });
    const percent = Math.min(99, Math.round((bytesRead / state.file.size) * 100));
    $("#progress-fill").style.width = `${percent}%`;
    $("#progress-percent").textContent = `${percent}%`;
    $("#progress-label").textContent = `Screened ${state.results.length.toLocaleString()} light curves…`;
  };

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      bytesRead += value.byteLength;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";
      for (const line of lines) await handleLine(line);
    }
    buffer += decoder.decode();
    if (buffer.trim()) await handleLine(buffer);
    await sendBatch();
    if (!headers) throw new Error("The CSV file is empty.");
    if (!rowNumber) throw new Error("The CSV has a header but no light-curve rows.");
    $("#progress-fill").style.width = "100%";
    $("#progress-percent").textContent = "100%";
    $("#progress-label").textContent = `Complete · ${rowNumber.toLocaleString()} light curves screened`;
    $("#analysis-status").textContent = "Screening complete.";
    renderResults();
  } catch (error) {
    showError(error.message || "Could not screen this CSV file.");
    $("#analysis-status").textContent = "Could not complete screening.";
  } finally {
    $("#analyze-button").disabled = false;
  }
});

function renderResults() {
  $("#results-section").classList.remove("hidden");
  $("#results-section").scrollIntoView({ behavior: "smooth", block: "start" });
  const flagged = state.results.filter((item) => item.class_name === "planet");
  const median = [...state.results].sort((a, b) => a.planet_probability - b.planet_probability)[Math.floor(state.results.length / 2)]?.planet_probability || 0;
  $("#results-metrics").innerHTML = [
    metric("SCREENED", state.results.length.toLocaleString(), "light curves"),
    metric("PLANET FLAGS", flagged.length.toLocaleString(), "model threshold · 0.50"),
    metric("MEDIAN PROBABILITY", `${(median * 100).toFixed(1)}%`, "planet class"),
  ].join("");
  if (state.actual) {
    const positives = state.results.filter((item) => item.actual === "planet");
    const tp = positives.filter((item) => item.class_name === "planet").length;
    const fp = flagged.filter((item) => item.actual === "no_planet").length;
    const recall = positives.length ? tp / positives.length : 0;
    const precision = flagged.length ? tp / flagged.length : 0;
    $("#scorecard").classList.remove("hidden");
    $("#scorecard").innerHTML = `<b>Known-label scorecard</b><span>This file has ${positives.length} planet examples. AstroLens found ${tp} (${(recall * 100).toFixed(0)}% recall) and flagged ${fp} no-planet curves (precision ${(precision * 100).toFixed(0)}%). Small samples make these metrics unstable.</span>`;
  }
  drawProbabilityChart();
  renderCurveOptions();
  renderTable();
}

function metric(label, value, note) {
  return `<div class="metric"><span>${label}</span><b>${value}</b><small>${note}</small></div>`;
}

function drawProbabilityChart() {
  const canvas = $("#probability-chart");
  const width = canvas.clientWidth || 500;
  const height = 250;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = width * ratio;
  canvas.height = height * ratio;
  canvas.style.height = `${height}px`;
  const ctx = canvas.getContext("2d");
  ctx.scale(ratio, ratio);
  ctx.clearRect(0, 0, width, height);
  const pad = { left: 40, right: 14, top: 14, bottom: 23 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  ctx.font = "10px DM Mono, monospace";
  ctx.strokeStyle = "#26384d";
  ctx.fillStyle = "#8195ab";
  for (let step = 0; step <= 4; step++) {
    const value = step / 4;
    const y = pad.top + plotH * (1 - value);
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(width - pad.right, y); ctx.stroke();
    ctx.fillText(`${Math.round(value * 100)}%`, 3, y + 3);
  }
  const thresholdY = pad.top + plotH / 2;
  ctx.setLineDash([4, 5]); ctx.strokeStyle = "#dfb87688";
  ctx.beginPath(); ctx.moveTo(pad.left, thresholdY); ctx.lineTo(width - pad.right, thresholdY); ctx.stroke(); ctx.setLineDash([]);
  const stepX = plotW / Math.max(1, state.results.length - 1);
  state.results.forEach((item, index) => {
    const x = pad.left + index * stepX;
    const y = pad.top + plotH * (1 - item.planet_probability);
    ctx.fillStyle = item.class_name === "planet" ? "#ffbd74" : "#69d5d0";
    ctx.globalAlpha = 0.76;
    ctx.beginPath(); ctx.arc(x, y, state.results.length > 1200 ? 1.5 : 2.5, 0, Math.PI * 2); ctx.fill();
  });
  ctx.globalAlpha = 1;
}

function renderCurveOptions() {
  const select = $("#curve-select");
  const available = [...state.curves.keys()].sort((a, b) => a - b);
  select.innerHTML = available.map((index) => {
    const item = state.results[index];
    return `<option value="${index}">Curve ${index + 1} · ${item?.class_name === "planet" ? "planet flag" : "no-planet flag"}</option>`;
  }).join("");
  select.onchange = drawSelectedCurve;
  drawSelectedCurve();
}

function drawSelectedCurve() {
  const selected = Number($("#curve-select").value || 0);
  const flux = state.curves.get(selected);
  if (!flux) return;
  const canvas = $("#curve-chart");
  const width = canvas.clientWidth || 400;
  const height = 220;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = width * ratio;
  canvas.height = height * ratio;
  canvas.style.height = `${height}px`;
  const ctx = canvas.getContext("2d");
  ctx.scale(ratio, ratio);
  ctx.clearRect(0, 0, width, height);
  const pad = { left: 12, right: 8, top: 12, bottom: 12 };
  let low = Math.min(...flux);
  let high = Math.max(...flux);
  if (high === low) high = low + 1;
  const range = high - low || 1;
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  ctx.strokeStyle = "#67d9d1"; ctx.lineWidth = 1.25; ctx.beginPath();
  const stride = Math.max(1, Math.floor(flux.length / Math.max(1, plotW * 2)));
  for (let i = 0; i < flux.length; i += stride) {
    const x = pad.left + (i / (flux.length - 1)) * plotW;
    const y = pad.top + (1 - (flux[i] - low) / range) * plotH;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();
  const result = state.results[selected];
  $("#curve-detail").textContent = `Curve ${selected + 1} · AstroLens probability ${(result.planet_probability * 100).toFixed(1)}% · confidence ${(result.confidence * 100).toFixed(1)}%`;
}

function renderTable() {
  const showPlanets = $("#candidate-filter").checked;
  const visible = state.results.filter((item) => !showPlanets || item.class_name === "planet");
  if (!visible.length) {
    $("#results-body").innerHTML = '<tr><td colspan="5" class="empty-state">No planet flags at the current threshold.</td></tr>';
    return;
  }
  $("#results-body").innerHTML = visible.map((item) => {
    const classFlag = item.class_name === "planet" ? "planet" : "clear";
    const actual = item.actual ? (item.actual === "planet" ? "Planet" : "No planet") : "—";
    return `<tr><td>${item.row}</td><td><span class="flag ${classFlag}">${item.class_name === "planet" ? "Planet flag" : "No planet"}</span></td><td class="prob-cell">${(item.planet_probability * 100).toFixed(2)}%</td><td>${(item.confidence * 100).toFixed(1)}%</td><td>${actual}</td></tr>`;
  }).join("");
}

$("#candidate-filter").addEventListener("change", renderTable);
$("#download-button").addEventListener("click", () => {
  const rows = [["row", "prediction", "planet_probability", "confidence", "actual_label"]];
  state.results.forEach((item) => rows.push([item.row, item.class_name, item.planet_probability, item.confidence, item.actual || ""]));
  const csv = rows.map((row) => row.map((cell) => `"${String(cell).replaceAll('"', '""')}"`).join(",")).join("\r\n");
  const link = document.createElement("a");
  link.href = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
  link.download = "keplervision_predictions.csv";
  link.click();
  URL.revokeObjectURL(link.href);
});

function clearResults() {
  state.results = [];
  state.curves.clear();
  $("#results-section").classList.add("hidden");
  $("#scorecard").classList.add("hidden");
  $("#progress-wrap").classList.add("hidden");
  hideError();
}
function showError(message) {
  $("#analysis-error").textContent = message;
  $("#analysis-error").classList.remove("hidden");
  $("#toast").textContent = message;
  $("#toast").classList.add("visible");
  setTimeout(() => $("#toast").classList.remove("visible"), 4500);
}
function hideError() { $("#analysis-error").classList.add("hidden"); }
function formatBytes(bytes) {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}
$("#candidate-filter").checked = false;
window.addEventListener("resize", () => {
  if (state.results.length) drawProbabilityChart();
  if ($("#curve-select").value) drawSelectedCurve();
});
