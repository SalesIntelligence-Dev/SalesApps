/* KI-Prognose Web App – Frontend Logic */
'use strict';

// ── State ────────────────────────────────────────────────────────────────
let currentJobId    = null;
let pollInterval    = null;
let progressTimer   = null;
let jobStartTime    = null;
let activeTab       = 'upload';
let selectedDataset = null;
let currentChart1   = null;
let currentChart2   = null;

// ── DOM refs ─────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);
const form         = $('forecast-form');
const runBtn       = $('run-btn');
const fileInput    = $('file-input');
const dropzone     = $('dropzone');
const fileName     = $('file-name');
const modelBadge   = $('model-status');
const emptyState   = $('empty-state');
const loadingState = $('loading-state');
const resultState  = $('result-state');
const errorState   = $('error-state');
const loadingProg  = $('loading-progress');
const reportHtml   = $('report-html');
const dlBtn        = $('dl-btn');
const colConfig    = $('col-config');
const targetCol    = $('target-col');
const timeCol      = $('time-col');
const samplesVal   = $('samples-val');
const bufferVal    = $('buffer-val');
const datasetList  = $('dataset-list');

// ── Init ──────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  checkModelStatus();
  loadDatasets();
  setupEventListeners();
  setInterval(checkModelStatus, 30_000);
});

// ── Model Status ─────────────────────────────────────────────────────────
async function checkModelStatus() {
  try {
    const r = await fetch('/api/health');
    const d = await r.json();
    modelBadge.className = 'model-badge ' + (d.runpod_ready ? 'ready' : 'loading');
    modelBadge.textContent = d.runpod_ready
      ? `RunPod bereit · GPU`
      : `RunPod wird konfiguriert …`;
  } catch {
    modelBadge.className = 'model-badge error';
    modelBadge.textContent = 'Verbindung fehlgeschlagen';
  }
}

// ── Datasets ──────────────────────────────────────────────────────────────
async function loadDatasets() {
  try {
    const r = await fetch('/api/datasets');
    const datasets = await r.json();
    datasetList.innerHTML = '';
    if (!datasets.length) {
      datasetList.innerHTML = '<div class="loading-dots">Keine Beispieldaten gefunden.</div>';
      return;
    }
    datasets.forEach(ds => {
      const card = document.createElement('div');
      card.className = 'dataset-card';
      card.dataset.id = ds.id;
      card.innerHTML = `
        <div class="dataset-name">${ds.description}</div>
        <div class="dataset-desc">${ds.name}</div>
        <div class="dataset-cols">
          ${ds.columns.map(c => `<span class="col-chip">${c}</span>`).join('')}
        </div>`;
      card.addEventListener('click', () => selectDataset(ds, card));
      datasetList.appendChild(card);
    });
  } catch (e) {
    datasetList.innerHTML = '<div class="loading-dots">Fehler beim Laden der Beispieldaten.</div>';
  }
}

async function selectDataset(ds, card) {
  document.querySelectorAll('.dataset-card').forEach(c => c.classList.remove('selected'));
  card.classList.add('selected');
  selectedDataset = ds.id;
  $('dataset-id-input').value = ds.id;

  // Spalten aus Preview laden
  try {
    const r = await fetch(`/api/datasets/${ds.id}/preview`);
    const d = await r.json();
    populateColumnSelects(d.columns);
    colConfig.classList.remove('hidden');
  } catch {}
}

function populateColumnSelects(columns) {
  [targetCol, timeCol].forEach(sel => {
    const prev = sel.value;
    sel.innerHTML = sel === timeCol
      ? '<option value="">– automatisch erkennen –</option>'
      : '<option value="">– automatisch –</option>';
    columns.forEach(c => {
      const opt = document.createElement('option');
      opt.value = opt.textContent = c;
      sel.appendChild(opt);
    });
    if (prev && columns.includes(prev)) sel.value = prev;
  });

  // Intelligente Defaults setzen
  const lower = columns.map(c => c.toLowerCase());
  const tsIdx = lower.findIndex(c => ['timestamp','date','datum','time','ts'].some(k => c.includes(k)));
  if (tsIdx >= 0) timeCol.value = columns[tsIdx];

  const numCols = columns.filter((_, i) => i !== tsIdx);
  if (numCols.length) targetCol.value = numCols[0];
}

// ── Tabs ──────────────────────────────────────────────────────────────────
function setupEventListeners() {
  // Upload/Default Tabs
  document.querySelectorAll('.tab').forEach(btn => {
    btn.addEventListener('click', () => {
      activeTab = btn.dataset.tab;
      document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
      btn.classList.add('active');
      $('tab-upload').classList.toggle('hidden',  activeTab !== 'upload');
      $('tab-default').classList.toggle('hidden', activeTab !== 'default');

      if (activeTab === 'default') {
        fileInput.value = '';
        fileName.textContent = '';
      } else {
        selectedDataset = null;
        $('dataset-id-input').value = '';
        document.querySelectorAll('.dataset-card').forEach(c => c.classList.remove('selected'));
      }
    });
  });

  // Ergebnis-Tabs
  document.querySelectorAll('.rtab').forEach(btn => {
    btn.addEventListener('click', () => {
      const id = btn.dataset.rtab;
      document.querySelectorAll('.rtab').forEach(t => t.classList.remove('active'));
      btn.classList.add('active');
      document.querySelectorAll('.rtab-content').forEach(c => c.classList.add('hidden'));
      $(`rtab-${id}`).classList.remove('hidden');

      // Charts beim ersten Anzeigen rendern
      if (id === 'chart' && currentChart1 && !$('forecast-chart').children.length) {
        Plotly.newPlot('forecast-chart', currentChart1.data, currentChart1.layout, {responsive: true});
      }
      if (id === 'daily' && currentChart2 && !$('daily-chart').children.length) {
        Plotly.newPlot('daily-chart', currentChart2.data, currentChart2.layout, {responsive: true});
      }
    });
  });

  // Dropzone
  dropzone.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', onFileSelected);
  dropzone.addEventListener('dragover', e => { e.preventDefault(); dropzone.classList.add('drag-over'); });
  dropzone.addEventListener('dragleave', () => dropzone.classList.remove('drag-over'));
  dropzone.addEventListener('drop', e => {
    e.preventDefault(); dropzone.classList.remove('drag-over');
    if (e.dataTransfer.files.length) {
      fileInput.files = e.dataTransfer.files;
      onFileSelected();
    }
  });

  // Slider Labels
  $('num-samples').addEventListener('input', e => { samplesVal.textContent = e.target.value; });
  $('safety-buffer').addEventListener('input', e => {
    bufferVal.textContent = Math.round(e.target.value * 100) + '%';
  });

  // Prompt Templates
  document.querySelectorAll('.tmpl-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      $('text-prompt').value = btn.dataset.prompt;
    });
  });

  // Formular
  form.addEventListener('submit', onSubmit);

  // Download
  dlBtn.addEventListener('click', downloadExcel);
}

async function onFileSelected() {
  const f = fileInput.files[0];
  if (!f) return;
  fileName.textContent = f.name;

  // CSV-Vorschau für Spalten-Auswahl
  if (f.name.endsWith('.csv')) {
    const text = await f.text();
    const cols  = text.split('\n')[0].split(',').map(c => c.trim().replace(/"/g, ''));
    populateColumnSelects(cols);
    colConfig.classList.remove('hidden');
  } else {
    colConfig.classList.add('hidden');
  }
}

// ── Forecast-Start ────────────────────────────────────────────────────────
async function onSubmit(e) {
  e.preventDefault();

  // Validierung
  if (activeTab === 'upload' && !fileInput.files[0]) {
    alert('Bitte eine Datei hochladen.');
    return;
  }
  if (activeTab === 'default' && !selectedDataset) {
    alert('Bitte einen Beispieldatensatz auswählen.');
    return;
  }

  const fd = new FormData(form);
  if (activeTab === 'default') {
    fd.delete('file');
    fd.set('dataset_id', selectedDataset);
  }

  setUIState('loading');
  runBtn.disabled = true;
  clearInterval(pollInterval);
  startProgressTimer();

  try {
    const r = await fetch('/api/forecast', { method: 'POST', body: fd });
    const d = await r.json();
    if (!r.ok || d.error) { showError(d.error || 'Unbekannter Fehler'); return; }
    currentJobId = d.job_id;
    pollInterval = setInterval(() => pollJob(currentJobId), 3000);
  } catch (err) {
    showError('Netzwerkfehler: ' + err.message);
  }
}

// ── Fortschrittsbalken ───────────────────────────────────────────────────
function startProgressTimer() {
  jobStartTime = Date.now();
  clearInterval(progressTimer);
  setProgress(0);
  // Zeitplan: 0-20s → 0-30%, 20-60s → 30-70%, 60-90s → 70-88%, 90s+ → 88-95%
  progressTimer = setInterval(() => {
    const elapsed = (Date.now() - jobStartTime) / 1000;
    let pct;
    if      (elapsed < 20)  pct = (elapsed / 20)  * 30;
    else if (elapsed < 60)  pct = 30 + ((elapsed - 20)  / 40) * 40;
    else if (elapsed < 90)  pct = 70 + ((elapsed - 60)  / 30) * 18;
    else                    pct = Math.min(95, 88 + ((elapsed - 90) / 60) * 7);
    setProgress(Math.round(pct));
  }, 1000);
}

function setProgress(pct) {
  const bar   = document.getElementById('progress-bar');
  const label = document.getElementById('progress-label');
  if (bar)   bar.style.width = pct + '%';
  if (label) label.textContent = pct + ' %';
}

function stopProgressTimer(success) {
  clearInterval(progressTimer);
  setProgress(success ? 100 : 0);
}

// ── Polling ───────────────────────────────────────────────────────────────
async function pollJob(jobId) {
  try {
    const r = await fetch(`/api/forecast/${jobId}`);
    const d = await r.json();

    if (d.progress) loadingProg.textContent = d.progress;

    if (d.status === 'done') {
      clearInterval(pollInterval);
      stopProgressTimer(true);
      showResults(d);
    } else if (d.status === 'error') {
      clearInterval(pollInterval);
      stopProgressTimer(false);
      showError(d.error || 'Inference fehlgeschlagen');
    }
  } catch (err) {
    console.error('Poll error:', err);
  }
}

// ── Ergebnisse anzeigen ───────────────────────────────────────────────────
function showResults(job) {
  setUIState('result');
  runBtn.disabled = false;

  // Bericht
  reportHtml.innerHTML = job.report?.html || '<p>Kein Bericht verfügbar.</p>';

  // Charts parsen und speichern
  if (job.chart1) {
    try { currentChart1 = JSON.parse(job.chart1); } catch {}
  }
  if (job.chart2) {
    try { currentChart2 = JSON.parse(job.chart2); } catch {}
  }

  // Charts zurücksetzen (werden beim Tab-Klick gerendert)
  $('forecast-chart').innerHTML = '';
  $('daily-chart').innerHTML    = '';

  // Direkt Chart-Tab initialisieren wenn aktiv
  const activeRtab = document.querySelector('.rtab.active')?.dataset.rtab;
  if (activeRtab === 'chart' && currentChart1) {
    Plotly.newPlot('forecast-chart', currentChart1.data, currentChart1.layout, {responsive: true});
  }
  if (activeRtab === 'daily' && currentChart2) {
    Plotly.newPlot('daily-chart', currentChart2.data, currentChart2.layout, {responsive: true});
  }

  checkModelStatus();
}

function downloadExcel() {
  if (currentJobId) {
    window.location.href = `/api/forecast/${currentJobId}/download`;
  }
}

// ── UI-State-Management ───────────────────────────────────────────────────
function setUIState(state) {
  emptyState.classList.add('hidden');
  loadingState.classList.add('hidden');
  resultState.classList.add('hidden');
  errorState.classList.add('hidden');

  if (state === 'loading') loadingState.classList.remove('hidden');
  else if (state === 'result') resultState.classList.remove('hidden');
  else if (state === 'error') errorState.classList.remove('hidden');
  else emptyState.classList.remove('hidden');
}

function showError(msg) {
  setUIState('error');
  $('error-msg').textContent = msg;
  runBtn.disabled = false;
}

function resetUI() {
  setUIState('empty');
  currentJobId = null;
  clearInterval(pollInterval);
}

// Global für retry-Buttons in Templates
window.resetUI = resetUI;
