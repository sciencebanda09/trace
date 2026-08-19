/**
 * TRACE · Physical-AI Demonstration Acquisition Engine
 * Dynamic deterministic priority scoring, on-device vision, and evidence pipeline.
 */
import { TRACE_SCHEMA as SCHEMA, CAPTURE_CONFIG, DEFAULT_WEIGHTS } from './core/schema.js';
import { createCaptureController } from './core/capture-controller.js';
import { scoreExperiments, calculateDatasetReadiness } from './core/priority-engine.js';
import { getRecords, putRecord, removeRecord, clearRecords, storageEstimate } from './core/storage.js';
import { detectDeviceCapabilities, describeSourceDevice } from './core/capture.js';
import { OPTIONAL_SIMULATOR_ENABLED } from './core/simulator.js';
import { createSpatialController } from './core/spatial-controller.js';
import { renderDatasetRows, setExclusiveView } from './core/ui.js';
import { createCaptureEntryController, createOnboardingController } from './core/workflow.js';
import { buildRecommendationViewModel } from './core/recommendation-presenter.js';

// Dynamic Schema Definition
const taskProfile = {
  id: 'generic-manipulation',
  task: 'manipulation',
  maxDurationSeconds: CAPTURE_CONFIG.defaultMaxDurationSeconds
};

// Seed dataset
const seedSpecs = [
  ['none','normal','upright','bench','success','no'],
  ['none','normal','upright','bench','success','no'],
  ['none','normal','upright','floor','success','no'],
  ['none','normal','upright','bench','success','no'],
  ['none','normal','rotated','bench','success','no'],
  ['none','normal','upright','floor','success','no'],
  ['none','normal','upright','bench','success','no'],
  ['none','normal','upright','bench','success','no'],
  ['none','normal','rotated','floor','success','no'],
  ['none','normal','upright','bench','success','no'],
  ['none','normal','upright','floor','success','no'],
  ['none','normal','upright','bench','success','no'],
  ['none','normal','upright','bench','success','no'],
  ['none','normal','rotated','floor','success','no'],
  ['partial','normal','upright','bench','failure','no'],
  ['partial','low-light','upright','bench','failure','no'],
  ['partial','low-light','upright','floor','success','no'],
  ['none','low-light','upright','bench','success','no'],
  ['none','bright','rotated','floor','success','no'],
  ['none','bright','upright','bench','success','no']
];

const seed = seedSpecs.map((x, i) => ({
  id: `seed-${String(i + 1).padStart(2, '0')}`,
  created: Date.now() - (seedSpecs.length - i) * 3600000,
  source: 'seed',
  durationSeconds: 6 + (i % 7),
  sourceDevice: { platform: 'imported-demo-dataset', mobile: false },
  task: 'manipulation',
  object: i % 4 === 0 ? 'bottle' : 'cup',
  occlusion: x[0],
  lighting: x[1],
  orientation: x[2],
  environment: x[3],
  result: x[4],
  recovery: x[5],
  notes: 'Deterministic biased demonstration dataset',
  evidence: []
}));

const $ = s => document.querySelector(s);
const $$ = s => [...document.querySelectorAll(s)];

// Global State
let clips = [];
let rankedExperiments = [];
let recommendation = null;
let activeRankIndex = 0;

// Dynamic Engine Weights (Configurable by user in Real-Time)
let engineWeights = { ...DEFAULT_WEIGHTS };
let deviceCapabilities = null;
let captureController;
let onboardingController;
let captureEntryController;

function cosineDistance(a, b) {
  let dot = 0, aa = 0, bb = 0;
  const size = Math.min(a.length, b.length);
  for (let i = 0; i < size; i++) {
    dot += a[i] * b[i];
    aa += a[i] * a[i];
    bb += b[i] * b[i];
  }
  return aa && bb ? 1 - dot / (Math.sqrt(aa) * Math.sqrt(bb)) : 0;
}

function embeddingNoveltyFor(context, fallback) {
  const embedded = clips.filter(c => Array.isArray(c.embedding));
  const related = embedded.filter(c =>
    ['occlusion', 'lighting', 'orientation', 'environment'].filter(k => c[k] === context[k]).length >= 2
  );
  if (embedded.length < 2 || !related.length) return fallback;
  const distances = related.flatMap(a =>
    embedded.filter(b => b.id !== a.id).map(b => cosineDistance(a.embedding, b.embedding))
  );
  return distances.length
    ? Math.min(1, distances.reduce((a, b) => a + b, 0) / distances.length)
    : fallback;
}

// UI Navigation & Filters
let activeMatrixFilter = 'all';
let activeVaultFilter = 'all';
let vaultSearchQuery = '';

let spatialController;


async function loadDataset() {
  try {
    clips = await getRecords();

    if (!clips.length) {
      clips = [...seed];
      await Promise.all(seed.map(saveRecord));
    }
  } catch (err) {
    console.warn('DB fallback to session memory:', err);
    clips = [...seed];
    toast('Using session memory dataset');
  }
  renderAll();
  try {
    const savedProject = JSON.parse(localStorage.getItem('trace-project') || 'null');
    if (savedProject?.task) taskProfile.task = savedProject.task;
    if (!localStorage.getItem('trace-onboarding-complete')) onboardingController.open(1, 'first-run');
  } catch (_) {
    onboardingController.open(1, 'first-run');
  }
}

async function saveRecord(row) {
  try {
    await putRecord(row);
  } catch (err) {
    console.warn('Record save error:', err);
  }
}

async function deleteRecord(id) {
  try {
    await removeRecord(id);
    clips = clips.filter(c => c.id !== id);
    renderAll();
    toast(`Demonstration ${id} deleted`);
  } catch (err) {
    toast('Failed to delete demonstration');
  }
}

async function resetDataset() {
  try {
    await clearRecords();
    clips = [...seed];
    await Promise.all(seed.map(saveRecord));
  } catch {
    clips = [...seed];
  }
  renderAll();
  toast('Deterministic demo dataset loaded');
}

function scoreAll() {
  return scoreExperiments({ clips, schema: SCHEMA, weights: engineWeights, noveltyFor: embeddingNoveltyFor });
}

function calculateReadiness() {
  return { total: calculateDatasetReadiness(clips, SCHEMA) };
}

function pct(n) {
  return `${Math.round(n * 100)}%`;
}

function titleCase(str) {
  if (!str) return '';
  return String(str).replace(/-/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"]/g, character => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[character]);
}

function generateInstruction(x) {
  return buildRecommendationViewModel({ clips, target: x, rank: activeRankIndex }).instruction;
}

function recommendationEvidence(target = recommendation) {
  const viewModel = buildRecommendationViewModel({ clips, target, rank: activeRankIndex });
  return { matching: viewModel.matching, failures: viewModel.failures, text: viewModel.reason };
}

async function finishOnboarding({ skip = false, mode = 'first-run' } = {}) {
  const robot = $('#setupRobot')?.value.trim() || 'Generic robot';
  const task = $('#setupTask')?.value.trim() || taskProfile.task;
  taskProfile.task = task;
  try {
    localStorage.setItem('trace-project', JSON.stringify({ robot, task }));
    localStorage.setItem('trace-onboarding-complete', 'true');
  } catch (_) {}

  if (!skip) {
    const datasetFiles = $('#setupDataset')?.files;
    if (datasetFiles?.length) await importDatasetFiles(datasetFiles);
    const notes = $('#setupFailureNotes')?.value.trim();
    const video = $('#setupFailureVideo')?.files?.[0];
    if (notes || video || mode === 'failure') {
      const target = recommendation || scoreAll()[0] || { occlusion:'partial', lighting:'normal', orientation:'upright', environment:'bench' };
      const failure = {
        id: `failure-${Date.now().toString(36).toUpperCase()}`,
        created: Date.now(), source: 'operator-report', task, object: 'unknown object',
        occlusion: target.occlusion, lighting: target.lighting, orientation: target.orientation, environment: target.environment,
        result: 'failure', recovery: 'no', failureType: $('#setupFailureType')?.value || 'other',
        notes: notes || 'Failure logged by operator', video: video || undefined,
        sourceDevice: describeSourceDevice(), evidence: [{ attribute:'result', method:'operator-confirmed', confidence:1 }]
      };
      clips.push(failure);
      await saveRecord(failure);
      renderAll();
      toast('Failure logged and priorities recalculated');
    }
  }
}

/* ==========================================================================
   UI Rendering
   ========================================================================== */
function renderAll() {
  rankedExperiments = scoreAll();
  
  // Keep active recommendation index or default to #0
  if (!recommendation || !rankedExperiments.some(r => r === recommendation)) {
    activeRankIndex = 0;
    recommendation = rankedExperiments[0];
  }

  renderSpotlight();
  renderCandidateQueue();
  renderRecentFailures();
  renderCoverageMatrix();
  renderDatasetVault();
  renderWeightsStudio();
  spatialController?.draw();
}

function renderSpotlight() {
  if (!recommendation) return;

  $('#instruction').textContent = generateInstruction(recommendation);

  const failures = clips.filter(c => c.result === 'failure');
  const matching = failures.filter(f => 
    ['occlusion', 'lighting', 'orientation', 'environment'].filter(k => f[k] === recommendation[k]).length >= 2
  ).length;

  $('#rationale').textContent = recommendationEvidence(recommendation).text;
  const story = $('#failureStory');
  if (story) story.innerHTML = `
    <div><b>${failures.length}</b><span>logged failures</span></div><i>→</i>
    <div><b>${matching}</b><span>matching pattern</span></div><i>→</i>
    <div><b>${pct(recommendation.gap)}</b><span>dataset gap</span></div><i>→</i>
    <div class="story-target"><b>#${activeRankIndex+1}</b><span>collect next</span></div>`;

  // Signal Metrics
  $('#failureScore').textContent = `${matching}/${failures.length}`;
  $('#gapScore').textContent = String(recommendation.count);
  $('#priorityScore').textContent = `#${activeRankIndex + 1}`;

  $('#failureBar').style.width = pct(recommendation.failure);
  $('#gapBar').style.width = pct(recommendation.gap);
  $('#priorityBar').style.width = `${Math.min(100, (recommendation.priority / 2.5) * 100)}%`;

  $('#activeRankBadge').textContent = `RANK #${String(activeRankIndex + 1).padStart(2, '0')}`;
  $('#coverage').textContent = calculateReadiness().total;
  $('#tabClipCount').textContent = clips.length;

  // Dynamic Target Configuration Badges
  const pillsContainer = $('#targetTagPills');
  if (pillsContainer) {
    pillsContainer.innerHTML = Object.entries(recommendation)
      .filter(([k]) => ['occlusion', 'lighting', 'orientation', 'environment'].includes(k))
      .map(([k, v]) => `
        <span class="target-badge-chip">
          <span class="chip-k">${titleCase(k)}:</span>
          <b>${titleCase(v)}</b>
        </span>
      `).join('');
  }

  // Formula Breakdown Details
  $('#paramWeightFail').textContent = engineWeights.failure.toFixed(2);
  $('#paramWeightGap').textContent = engineWeights.gap.toFixed(2);
  $('#calcFailTrace').textContent = `Matches ${matching} failure incident contexts with ${pct(recommendation.failure)} correlation.`;
  $('#calcGapTrace').textContent = `${recommendation.count} existing demos in this cell produces ${pct(recommendation.gap)} gap score.`;
  $('#calcCostTrace').textContent = `${recommendation.cost.toFixed(2)}x`;

  // Capture HUD Target Label
  const hudTarget = $('#captureTarget');
  if (hudTarget) hudTarget.textContent = generateInstruction(recommendation);

  // Spatial HUD Labels & Telemetry
  updateSpatialHUD();
}

function renderCandidateQueue() {
  const container = $('#runnerUpList');
  if (!container) return;

  const candidates = rankedExperiments.slice(0, 5);
  container.innerHTML = candidates.map((item, idx) => {
    const isCurrent = item === recommendation;
    return `
      <div class="runner-item-card ${isCurrent ? 'active-target' : ''}">
        <span class="r-rank">#${String(idx + 1).padStart(2, '0')}</span>
        <div class="r-info">
          <span class="r-name">${titleCase(item.occlusion)} occl · ${titleCase(item.orientation)} on ${titleCase(item.environment)}</span>
          <span class="r-meta">${titleCase(item.lighting)} lighting · Gap: ${pct(item.gap)} · Fail wgt: ${pct(item.failure)}</span>
        </div>
        <div class="r-score-col">
          <span class="r-score-val">${item.priority.toFixed(2)}</span>
          <span class="r-score-lbl">PRIORITY</span>
        </div>
        <button class="btn-glass btn-sm" onclick="window.selectTargetCandidate(${idx})">
          ${isCurrent ? 'Selected' : 'Focus'}
        </button>
      </div>
    `;
  }).join('');
}

window.selectTargetCandidate = function(idx) {
  if (rankedExperiments[idx]) {
    activeRankIndex = idx;
    recommendation = rankedExperiments[idx];
    renderSpotlight();
    renderCandidateQueue();
    spatialController?.draw();
    toast(`Switched target to Rank #${String(idx + 1).padStart(2, '0')}`);
  }
};

function renderRecentFailures() {
  const container = $('#failureCards');
  if (!container) return;

  const failures = clips.filter(c => c.result === 'failure');
  if (!failures.length) {
    container.innerHTML = '<p class="b-desc">No robot breakdowns recorded yet.</p>';
    return;
  }

  container.innerHTML = failures.slice(0, 4).map((f, i) => `
    <div class="failure-item-card">
      <div class="f-top">
        <span class="f-title">Incident #${String(i + 1).padStart(2, '0')} — ${escapeHtml(titleCase(f.occlusion))} Occlusion</span>
        <span class="f-tag">FAILURE LOGGED</span>
      </div>
      <span class="f-desc">
        ${escapeHtml(titleCase(f.lighting))} lighting · ${escapeHtml(titleCase(f.orientation))} on ${escapeHtml(titleCase(f.environment))} · ${f.recovery === 'yes' ? 'Recovery attempted' : 'No recovery'}
      </span>
    </div>
  `).join('');
}

function renderCoverageMatrix() {
  // Dimension distributions
  const dimsContainer = $('#coverageMap');
  if (dimsContainer) {
    const dimensions = ['occlusion', 'lighting', 'orientation', 'environment', 'recovery'];
    
    dimsContainer.innerHTML = dimensions.map(dim => {
      const terms = SCHEMA[dim] || ['no', 'yes'];
      const rows = terms.map(val => {
        const matching = clips.filter(c => c[dim] === val);
        const count = matching.length;
        const fails = matching.filter(c => c.result === 'failure').length;
        const succ = count - fails;
        const total = Math.max(1, clips.length);

        return `
          <div class="dim-row">
            <span class="dim-label">${titleCase(val)}</span>
            <div class="dim-track">
              <div class="dim-fill-succ" style="width: ${(succ / total) * 100}%" title="${succ} Successful"></div>
              <div class="dim-fill-fail" style="width: ${(fails / total) * 100}%" title="${fails} Failures"></div>
            </div>
            <span class="dim-counts-text">${count}${fails ? ` (${fails}F)` : ''}</span>
          </div>
        `;
      }).join('');

      return `
        <div class="dim-section-title">${dim.toUpperCase()} DISTRIBUTION</div>
        ${rows}
      `;
    }).join('');
  }

  // Permutation Grid
  const gridContainer = $('#permutationGrid');
  if (gridContainer) {
    const allCombos = scoreAll();
    let filtered = allCombos;

    if (activeMatrixFilter === 'gap') {
      filtered = allCombos.filter(c => c.count === 0);
    } else if (activeMatrixFilter === 'failures') {
      filtered = allCombos.filter(c => c.failCount > 0);
    }

    gridContainer.innerHTML = filtered.map(item => {
      let statusClass = 'empty';
      let statusText = '0 Demos';
      if (item.count > 0) {
        statusClass = item.failCount > 0 ? 'fail' : 'ok';
        statusText = `${item.count} demo${item.count === 1 ? '' : 's'}${item.failCount ? ` · ${item.failCount}F` : ''}`;
      }

      return `
        <div class="heatmap-cell ${item.count === 0 ? 'is-gap' : ''} ${item.failCount > 0 ? 'has-fail' : ''}">
          <div class="cell-header-row">
            <span class="cell-count-tag ${statusClass}">${statusText}</span>
            <code style="font-size:10px; color:var(--azure-light)">${item.priority.toFixed(2)}</code>
          </div>
          <span class="cell-body-text">
            <b>${titleCase(item.occlusion)}</b> · ${titleCase(item.lighting)}<br>
            ${titleCase(item.orientation)} · ${titleCase(item.environment)}
          </span>
        </div>
      `;
    }).join('');
  }
}

function renderDatasetVault() {
  const tbody = $('#datasetTableBody');
  if (!tbody) return;
  updateStorageUsage();

  const total = clips.length;
  const failures = clips.filter(c => c.result === 'failure').length;
  const recoveries = clips.filter(c => c.recovery === 'yes').length;
  const live = clips.filter(c => c.source === 'live').length;

  if ($('#countAll')) $('#countAll').textContent = total;
  if ($('#countFailures')) $('#countFailures').textContent = failures;
  if ($('#countRecovery')) $('#countRecovery').textContent = recoveries;
  if ($('#countLive')) $('#countLive').textContent = live;

  let filtered = [...clips].reverse();

  if (activeVaultFilter === 'failures') {
    filtered = filtered.filter(c => c.result === 'failure');
  } else if (activeVaultFilter === 'recovery') {
    filtered = filtered.filter(c => c.recovery === 'yes');
  } else if (activeVaultFilter === 'live') {
    filtered = filtered.filter(c => c.source === 'live');
  }

  if (vaultSearchQuery) {
    const q = vaultSearchQuery.toLowerCase();
    filtered = filtered.filter(c => 
      c.occlusion.toLowerCase().includes(q) ||
      c.lighting.toLowerCase().includes(q) ||
      c.orientation.toLowerCase().includes(q) ||
      c.environment.toLowerCase().includes(q) ||
      c.result.toLowerCase().includes(q) ||
      (c.id && c.id.toLowerCase().includes(q))
    );
  }

  if (!filtered.length) {
    tbody.innerHTML = `
      <tr>
        <td colspan="9" style="text-align:center; padding:36px; color:var(--ink-muted)">
          No demonstrations found matching criteria.
        </td>
      </tr>
    `;
    return;
  }

  renderDatasetRows(tbody, filtered, { titleCase });
}

function renderWeightsStudio() {
  $('#valWeightFail').textContent = engineWeights.failure.toFixed(2);
  $('#valWeightGap').textContent = engineWeights.gap.toFixed(2);
  $('#valWeightNov').textContent = engineWeights.novelty.toFixed(2);
  $('#valWeightCost').textContent = `${engineWeights.costSensitivity.toFixed(2)}x`;

  const previewList = $('#liveRankingPreview');
  if (!previewList) return;

  const top3 = rankedExperiments.slice(0, 3);
  previewList.innerHTML = top3.map((item, i) => `
    <div class="runner-item-card">
      <span class="r-rank">#0${i + 1}</span>
      <div class="r-info">
        <span class="r-name">${titleCase(item.occlusion)} occl · ${titleCase(item.orientation)} on ${titleCase(item.environment)}</span>
        <span class="r-meta">${titleCase(item.lighting)} · Gap: ${pct(item.gap)} · Fail: ${pct(item.failure)}</span>
      </div>
      <div class="r-score-col">
        <span class="r-score-val">${item.priority.toFixed(2)}</span>
      </div>
    </div>
  `).join('');
}

/* ==========================================================================
   Synthetic Demonstration & JSON Export
   ========================================================================== */
function injectSyntheticDemo() {
  if (!recommendation) recommendation = scoreAll()[0];

  const target = recommendation;
  const isFail = Math.random() < 0.30;
  const hasRecovery = isFail && Math.random() < 0.5 ? 'yes' : 'no';

  const row = {
    id: `sim-${Date.now().toString(36).toUpperCase()}`,
    created: Date.now(),
    source: 'live',
    occlusion: target.occlusion,
    lighting: target.lighting,
    orientation: target.orientation,
    environment: target.environment,
    result: isFail ? 'failure' : 'success',
    recovery: hasRecovery,
    notes: 'Synthetic demonstration generated for validation'
  };

  const beforeReadiness = calculateReadiness().total;
  clips.push(row);
  saveRecord(row);
  renderAll();

  const afterReadiness = calculateReadiness().total;
  $('#oldCoverage').textContent = `${beforeReadiness}%`;
  $('#newCoverage').textContent = `${afterReadiness}%`;

  $('#successCopy').textContent = `Injected demonstration for target experiment: ${target.occlusion} occlusion, ${target.orientation} object on ${target.environment} (${row.result.toUpperCase()}). Next-best experiment re-ranked instantly.`;
  showView('successView');
  toast('Demonstration added to repository');
}

function exportDatasetJson() {
  const exportData = clips.map(({ video, ...rest }) => rest);
  const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `trace-dataset-${Date.now()}.json`;
  a.click();
  URL.revokeObjectURL(url);
  toast('Exported dataset to JSON');
}

async function importDatasetFiles(files) {
  const list = [...files];
  const jsonFiles = list.filter(file => file.type === 'application/json' || file.name.toLowerCase().endsWith('.json'));
  const videoFiles = list.filter(file => file.type.startsWith('video/'));
  let imported = 0;
  for (const file of jsonFiles) {
    try {
      const parsed = JSON.parse(await file.text());
      const rows = Array.isArray(parsed) ? parsed : parsed.demonstrations;
      if (!Array.isArray(rows)) throw new Error('Expected an array of demonstrations');
      for (const raw of rows) {
        if (!raw || typeof raw !== 'object') continue;
        const allowed = (key, value, fallback) => SCHEMA[key]?.includes(value) ? value : fallback;
        const row = {
          id: String(raw.id || `import-${crypto.randomUUID?.() || Date.now()}`).slice(0, 120),
          created: Number.isFinite(Number(raw.created)) ? Number(raw.created) : Date.now(),
          source: 'import',
          task: String(raw.task || taskProfile.task).slice(0, 160),
          object: String(raw.object || 'unknown object').slice(0, 160),
          occlusion: allowed('occlusion', raw.occlusion, 'none'),
          lighting: allowed('lighting', raw.lighting, 'normal'),
          orientation: allowed('orientation', raw.orientation, 'upright'),
          environment: allowed('environment', raw.environment, 'bench'),
          result: allowed('result', raw.result, 'failure'),
          recovery: allowed('recovery', raw.recovery, 'no'),
          notes: String(raw.notes || '').slice(0, 2000),
          evidence: Array.isArray(raw.evidence) ? raw.evidence.slice(0, 100) : []
        };
        clips.push(row); await saveRecord(row); imported++;
      }
    } catch (error) { toast(`${file.name}: invalid TRACE JSON`); }
  }
  if (imported) { renderAll(); toast(`Imported ${imported} demonstrations`); }
  if (videoFiles.length) {
    if (videoFiles.length > 1) toast('Opening the first video; confirm each clip separately');
    await captureController.analyzeUpload(videoFiles[0]);
  }
}

async function updateStorageUsage() {
  const text = $('#storageUsageText'), bar = $('#storageUsageBar');
  if (!text || !bar) return;
  const {usage=0,quota=0} = await storageEstimate();
  const percent = quota ? usage/quota*100 : 0;
  text.textContent = quota ? `Local storage ${(usage/1048576).toFixed(1)} MB of ${(quota/1048576).toFixed(0)} MB (${percent.toFixed(1)}%)` : `${clips.length} local records · quota unavailable`;
  bar.style.width = `${Math.min(100,percent)}%`;
  bar.classList.toggle('warning', percent >= 80);
  if (percent >= 90) toast('Storage nearly full. Export or delete recordings.');
}

/* ==========================================================================
   Toast Notification & View Switching
   ========================================================================== */
function toast(msg) {
  const t = $('#toast');
  if (!t) return;
  t.textContent = msg;
  t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 2600);
}

function showView(viewId) {
  setExclusiveView($$('.view'), viewId);
  $$('.tab-btn').forEach(t => t.classList.toggle('active', t.dataset.view === viewId));
  $$('.mobile-tab-btn').forEach(b => b.classList.toggle('active', b.dataset.view === viewId));

  const menu = $('#moreMenu');
  if (menu) menu.classList.remove('show');

  document.body.classList.toggle('capture-focus', viewId === 'captureView');

  window.scrollTo({ top: 0, behavior: 'smooth' });

}

/* ==========================================================================
   Event Listeners & Initialization
   ========================================================================== */
function initEventListeners() {
  $('#datasetTableBody')?.addEventListener('click', event => {
    const button = event.target.closest('[data-remove-record]');
    if (!button) return;
    const id = button.dataset.removeRecord;
    if (id && confirm(`Delete demonstration record ${id}?`)) deleteRecord(id);
  });

  // Navigation tabs
  $$('[data-view]').forEach(btn => {
    btn.onclick = () => showView(btn.dataset.view);
  });

  // Capture buttons
  $('#quickRecordBtn')?.addEventListener('click', captureEntryController.open);
  $('#quickRecordMobile')?.addEventListener('click', captureEntryController.open);
  $('#recordThis')?.addEventListener('click', captureEntryController.open);
  $('#logFailureBtn')?.addEventListener('click', () => onboardingController.open(2, 'failure'));
  $('#quickSimulateBtn')?.addEventListener('click', injectSyntheticDemo);
  $('#datasetImportInput')?.addEventListener('change', event => importDatasetFiles(event.target.files));

  // Progressive Disclosure: Formula Inspector
  const formulaBtn = $('#toggleFormulaBtn');
  const formulaPanel = $('#formulaPanel');
  if (formulaBtn && formulaPanel) {
    formulaBtn.addEventListener('click', () => {
      const isOpen = formulaPanel.classList.toggle('open');
      formulaBtn.classList.toggle('open', isOpen);
    });
  }

  // Weight Sliders
  $('#sliderWeightFail')?.addEventListener('input', (e) => {
    engineWeights.failure = parseFloat(e.target.value);
    renderAll();
  });
  $('#sliderWeightGap')?.addEventListener('input', (e) => {
    engineWeights.gap = parseFloat(e.target.value);
    renderAll();
  });
  $('#sliderWeightNov')?.addEventListener('input', (e) => {
    engineWeights.novelty = parseFloat(e.target.value);
    renderAll();
  });
  $('#sliderWeightCost')?.addEventListener('input', (e) => {
    engineWeights.costSensitivity = parseFloat(e.target.value);
    renderAll();
  });

  // Presets
  $('#presetFailure')?.addEventListener('click', () => {
    engineWeights = { gap: 0.30, failure: 0.50, novelty: 0.20, costSensitivity: 1.00 };
    $('#sliderWeightFail').value = 0.50;
    $('#sliderWeightGap').value = 0.30;
    $('#sliderWeightNov').value = 0.20;
    $('#sliderWeightCost').value = 1.00;
    renderAll();
    toast('Loaded Failure-Driven preset');
  });

  $('#presetExploration')?.addEventListener('click', () => {
    engineWeights = { gap: 0.70, failure: 0.10, novelty: 0.20, costSensitivity: 0.50 };
    $('#sliderWeightFail').value = 0.10;
    $('#sliderWeightGap').value = 0.70;
    $('#sliderWeightNov').value = 0.20;
    $('#sliderWeightCost').value = 0.50;
    renderAll();
    toast('Loaded Pure Exploration preset');
  });

  $('#presetLowCost')?.addEventListener('click', () => {
    engineWeights = { gap: 0.35, failure: 0.25, novelty: 0.15, costSensitivity: 1.80 };
    $('#sliderWeightFail').value = 0.25;
    $('#sliderWeightGap').value = 0.35;
    $('#sliderWeightNov').value = 0.15;
    $('#sliderWeightCost').value = 1.80;
    renderAll();
    toast('Loaded Low-Cost Batch preset');
  });

  $('#resetWeightsBtn')?.addEventListener('click', () => {
    engineWeights = { gap: 0.30, failure: 0.50, novelty: 0.20, costSensitivity: 1.00 };
    $('#sliderWeightFail').value = 0.50;
    $('#sliderWeightGap').value = 0.30;
    $('#sliderWeightNov').value = 0.20;
    $('#sliderWeightCost').value = 1.00;
    renderAll();
    toast('Weights reset to defaults');
  });

  // Dropdown Menu
  const moreBtn = $('#moreActionsBtn');
  const moreMenu = $('#moreMenu');
  if (moreBtn && moreMenu) {
    moreBtn.onclick = (e) => {
      e.stopPropagation();
      moreMenu.classList.toggle('show');
    };
    document.addEventListener('click', () => {
      moreMenu.classList.remove('show');
    });
  }

  $('#menuAddSynthetic')?.addEventListener('click', injectSyntheticDemo);
  $('#vaultAddSynthetic')?.addEventListener('click', injectSyntheticDemo);
  $('#menuExportJson')?.addEventListener('click', exportDatasetJson);
  $('#vaultExportJson')?.addEventListener('click', exportDatasetJson);
  $('#resetBtn')?.addEventListener('click', resetDataset);

  // Filters
  $$('[data-matrix-filter]').forEach(btn => {
    btn.onclick = () => {
      $$('[data-matrix-filter]').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      activeMatrixFilter = btn.dataset.matrixFilter;
      renderCoverageMatrix();
    };
  });

  $$('[data-filter]').forEach(btn => {
    btn.onclick = () => {
      $$('[data-filter]').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      activeVaultFilter = btn.dataset.filter;
      renderDatasetVault();
    };
  });

  $('#datasetSearch')?.addEventListener('input', (e) => {
    vaultSearchQuery = e.target.value;
    renderDatasetVault();
  });
}

// Launch
document.addEventListener('DOMContentLoaded', () => {
  deviceCapabilities = detectDeviceCapabilities();
  captureController = createCaptureController({
    query: $, queryAll: $, taskProfile,
    getClips: () => clips, getRecommendation: () => recommendation,
    getRankedExperiments: () => rankedExperiments, getDeviceCapabilities: () => deviceCapabilities,
    saveRecord, renderAll, calculateReadiness, recommendationEvidence, showView, toast, titleCase, escapeHtml
  });
  spatialController = createSpatialController({
    query: $, queryAll: $$,
    getRecommendation: () => recommendation,
    getRankedExperiments: () => rankedExperiments,
    titleCase, toast, beginCapture: () => captureController.begin()
  });
  onboardingController = createOnboardingController({ query: $, queryAll: $$, onFinish: finishOnboarding });
  captureEntryController = createCaptureEntryController({
    query: $,
    getInstruction: () => recommendation ? generateInstruction(recommendation) : '',
    getReason: () => recommendationEvidence().text,
    onStart: () => captureController.begin()
  });
  onboardingController.bind();
  captureEntryController.bind();
  captureController.bind();
  captureController.loadModels();
  initEventListeners();
  if (OPTIONAL_SIMULATOR_ENABLED) spatialController?.init();
  loadDataset();
});
