/**
 * TRACE · Physical-AI Demonstration Acquisition Engine
 * Dynamic deterministic priority scoring, on-device vision, and evidence pipeline.
 */
import { initYolo, detectObjects, drawDetections, getYoloRuntime } from './yolo.js';
import { TRACE_SCHEMA as SCHEMA, CAPTURE_CONFIG, DEFAULT_WEIGHTS } from './core/schema.js';
import { scoreExperiments, calculateDatasetReadiness } from './core/priority-engine.js';
import { getRecords, putRecord, removeRecord, clearRecords, storageEstimate } from './core/storage.js';
import { calibrationCorrection, confidenceLabel } from './core/evidence.js';
import { detectDeviceCapabilities, describeSourceDevice, formatClock } from './core/capture.js';
import { OPTIONAL_SIMULATOR_ENABLED } from './core/simulator.js';
import { setExclusiveView } from './core/ui.js';

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
let calibrationCorrections = [];

// Camera & Capture State
let stream = null;
let recorder = null;
let chunks = [];
let seconds = 0;
let ticker = null;
let sampleTicker = null;
let recordedBlob = null;
let proposedTags = {};
let proposedEvidence = {};
let proposedTask = taskProfile.task;
let proposedObject = 'unknown object';
let frameMetrics = [];
let capturedFrameCanvases = [];
let currentFacingMode = 'environment';
let recordingStartedAt = 0;
let recordedDurationSeconds = 0;
let imuSamples = [];
let audioEnabled = false;
let deviceCapabilities = null;

function collectMotionSample(event) {
  if (!recordingStartedAt || imuSamples.length >= 300) return;
  const a = event.accelerationIncludingGravity;
  const r = event.rotationRate;
  imuSamples.push({
    t: performance.now() - recordingStartedAt,
    acceleration: a ? [a.x, a.y, a.z] : null,
    rotation: r ? [r.alpha, r.beta, r.gamma] : null
  });
}
let visionModel = null;
let modelBackend = 'unavailable';
let embeddingSamples = [];
let embeddingPromises = [];
let objectLabels = [];
let yoloReady = false;
let yoloBusy = false;
let detectionFrames = [];
let detectionPromises = [];
let detectionCanvases = [];
let yoloLatencyMs = 0;
let playbackUrl = null;
let outcomeConfirmed = false;
let onboardingStep = 1;
let onboardingMode = 'first-run';

async function loadYoloModel() {
  const status = $('#yoloLiveStatus');
  try {
    await initYolo();
    yoloReady = true;
    if (status) status.textContent = `YOLOv8n · ${getYoloRuntime().backend.toUpperCase()}`;
  } catch (error) {
    console.error('YOLO load failed:', error);
    if (status) status.textContent = 'YOLO UNAVAILABLE';
  }
  renderModelHealth();
}

async function loadVisionModel() {
  const status = $('#systemStatusText');
  if (!globalThis.tf || !globalThis.mobilenet) {
    if (status) status.textContent = 'Model Library Offline · Heuristic Fallback';
    return;
  }
  try {
    if (status) status.textContent = 'Loading MobileNet On-Device';
    await tf.setBackend('webgl');
    await tf.ready();
    modelBackend = tf.getBackend();
    visionModel = await mobilenet.load({ version: 1, alpha: 0.25 });
    if (status) status.textContent = `MobileNet Ready · ${modelBackend.toUpperCase()}`;
  } catch (error) {
    console.error('MobileNet load failed:', error);
    if (status) status.textContent = 'Model Unavailable · Heuristic Fallback';
  }
  renderModelHealth();
}

function renderModelHealth() {
  const container = $('#modelHealthStrip');
  if (!container) return;
  const items = [
    ['YOLO', yoloReady ? getYoloRuntime().backend.toUpperCase() : 'loading', yoloReady],
    ['MobileNet', visionModel ? modelBackend.toUpperCase() : 'loading', Boolean(visionModel)],
    ['Storage', deviceCapabilities?.indexedDB ? 'local' : 'memory', Boolean(deviceCapabilities?.indexedDB)],
    ['Latency', yoloLatencyMs ? `${Math.round(yoloLatencyMs)}ms` : 'not measured', !yoloLatencyMs || yoloLatencyMs < 1500]
  ];
  container.innerHTML = items.map(([name,value,ok])=>`<span class="${ok?'healthy':'degraded'}"><i></i>${name}<b>${value}</b></span>`).join('');
}

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

// Interactive 3D Spatial Visualizer State
let spatial = {
  yaw: 0.65,
  pitch: 0.46,
  zoom: 1.0,
  targetYaw: 0.65,
  targetPitch: 0.46,
  targetZoom: 1.0,
  isDragging: false,
  dragStartX: 0,
  dragStartY: 0,
  lastDragX: 0,
  lastDragY: 0,
  velX: 0,
  velY: 0,
  cameraMode: 'iso',       // 'iso', 'top', 'front', 'tcp'
  renderMode: 'shaded',    // 'shaded', 'wireframe', 'collision'
  autoOrbit: false,
  orbitSpeed: 0.007,
  gyroActive: false,
  gyroBaseline: { beta: null, gamma: null },
  gyroSensitivity: 0.025,
  isSimGrasp: true,
  graspProgress: 0.0,
  graspSpeed: 0.004,
  override: null,          // testing override or null (target recommendation)
  particles: [],
  fps: 60,
  fpsCount: 0,
  fpsTimer: performance.now(),
  modalOpen: false,
  animId: null
};

/* ==========================================================================
   IndexedDB Store
   ========================================================================== */
function openDB() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open('trace-store', 1);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains('clips')) {
        db.createObjectStore('clips', { keyPath: 'id' });
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

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
    if (!localStorage.getItem('trace-onboarding-complete')) openOnboarding(1, 'first-run');
  } catch (_) {
    openOnboarding(1, 'first-run');
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
  if (!x) return 'Loading experiment targets...';
  const hasLowRecovery = clips.filter(c => c.recovery === 'yes').length / Math.max(1, clips.length) < 0.15;
  const occl = x.occlusion === 'none' ? 'unobstructed' : `${x.occlusion}-occlusion`;
  const recoveryNote = hasLowRecovery ? ' Record the retry too.' : '';
  return `Test this grasp: ${titleCase(occl)} · ${titleCase(x.orientation)} object · ${titleCase(x.lighting)} light · ${titleCase(x.environment)}.${recoveryNote}`;
}

function recommendationEvidence(target = recommendation) {
  if (!target) return { matching: 0, failures: 0, text: 'TRACE is calculating the strongest evidence.' };
  const failures = clips.filter(clip => clip.result === 'failure');
  const matching = failures.filter(failure =>
    ['occlusion', 'lighting', 'orientation', 'environment'].filter(key => failure[key] === target[key]).length >= 2
  ).length;
  const exampleWord = target.count === 1 ? 'demonstration' : 'demonstrations';
  return {
    matching,
    failures: failures.length,
    text: `Ranked #${activeRankIndex + 1} because ${matching} of ${failures.length} logged failures share these conditions, but the dataset contains only ${target.count} matching ${exampleWord}.`
  };
}

function setOnboardingStep(step) {
  onboardingStep = Math.max(1, Math.min(3, step));
  $$('.workflow-step').forEach(panel => panel.classList.toggle('active', Number(panel.dataset.onboardingStep) === onboardingStep));
  if ($('#onboardingProgress')) $('#onboardingProgress').style.width = `${onboardingStep * 33.333}%`;
  if ($('#onboardingBack')) $('#onboardingBack').hidden = onboardingStep === 1;
  if ($('#onboardingNext')) $('#onboardingNext').textContent = onboardingStep === 3 ? (onboardingMode === 'failure' ? 'LOG FAILURE' : 'BUILD FIRST RECOMMENDATION') : 'Continue';
  if ($('#setupSummary') && onboardingStep === 3) {
    const robot = $('#setupRobot')?.value.trim() || 'your robot';
    const task = $('#setupTask')?.value.trim() || 'the manipulation task';
    $('#setupSummary').textContent = `TRACE will analyze ${robot} performing ${task}, connect the reported failure to dataset gaps, and rank the next capture.`;
  }
}

function openOnboarding(step = 1, mode = 'first-run') {
  onboardingMode = mode;
  const modal = $('#onboardingModal');
  if (!modal) return;
  modal.hidden = false;
  setOnboardingStep(step);
  setTimeout(() => (step === 1 ? $('#setupRobot') : $('#setupFailureNotes'))?.focus(), 50);
}

function closeOnboarding() {
  if ($('#onboardingModal')) $('#onboardingModal').hidden = true;
}

async function finishOnboarding({ skip = false } = {}) {
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
    if (notes || video || onboardingMode === 'failure') {
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
  closeOnboarding();
}

function openCaptureBriefing() {
  if (!recommendation) return;
  $('#briefingInstruction').textContent = generateInstruction(recommendation);
  $('#briefingReason').textContent = recommendationEvidence().text;
  $('#captureBriefing').hidden = false;
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
  drawSpatialVisualizer();
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
  $('#failureScore').textContent = pct(recommendation.failure);
  $('#gapScore').textContent = pct(recommendation.gap);
  $('#priorityScore').textContent = recommendation.priority.toFixed(2);

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
    drawSpatialVisualizer();
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
        <span class="f-title">Incident #${String(i + 1).padStart(2, '0')} — ${titleCase(f.occlusion)} Occlusion</span>
        <span class="f-tag">FAILURE LOGGED</span>
      </div>
      <span class="f-desc">
        ${titleCase(f.lighting)} lighting · ${titleCase(f.orientation)} on ${titleCase(f.environment)} · ${f.recovery === 'yes' ? 'Recovery attempted' : 'No recovery'}
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

  tbody.innerHTML = filtered.map(row => {
    const isLive = row.source === 'live';
    const dateStr = new Date(row.created).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    return `
      <tr>
        <td><span class="source-tag ${isLive ? 'live' : 'seed'}">${row.id}</span></td>
        <td>${titleCase(row.occlusion)}</td>
        <td>${titleCase(row.lighting)}</td>
        <td>${titleCase(row.orientation)}</td>
        <td>${titleCase(row.environment)}</td>
        <td><span class="status-badge ${row.result === 'success' ? 'success' : 'failure'}">${row.result.toUpperCase()}</span></td>
        <td>${row.recovery === 'yes' ? 'Yes' : 'No'}</td>
        <td class="font-mono" style="font-size:11px; color:var(--ink-muted)">${dateStr}</td>
        <td class="align-right">
          <button class="action-icon-btn" onclick="window.removeDemo('${row.id}')" title="Delete record">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path></svg>
          </button>
        </td>
      </tr>
    `;
  }).join('');
}

window.removeDemo = function(id) {
  if (confirm(`Delete demonstration record ${id}?`)) {
    deleteRecord(id);
  }
};

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
   Interactive 3D Physical Scene Visualizer & Spatial Simulator Engine
   ========================================================================== */

function getActiveSpatialConfig() {
  if (spatial.override) {
    return {
      environment: spatial.override.environment || recommendation?.environment || 'bench',
      orientation: spatial.override.orientation || recommendation?.orientation || 'upright',
      occlusion: spatial.override.occlusion || recommendation?.occlusion || 'none',
      lighting: spatial.override.lighting || recommendation?.lighting || 'normal',
      isOverride: true
    };
  }
  return recommendation || {
    environment: 'bench',
    orientation: 'upright',
    occlusion: 'none',
    lighting: 'normal',
    isOverride: false
  };
}

function updateSpatialHUD() {
  const target = getActiveSpatialConfig();

  // Ground Surface Tag
  const surfaceTag = $('#spatialSurfaceTag');
  if (surfaceTag) {
    surfaceTag.textContent = target.environment === 'floor' ? '🏭 Floor [Epoxy]' : '⚙️ Bench [Al-6061]';
  }

  // Lighting Tag
  const luxTag = $('#spatialLuxTag');
  if (luxTag) {
    if (target.lighting === 'bright') {
      luxTag.textContent = '💡 1,250 Lux (Bright Direct)';
    } else if (target.lighting === 'low-light') {
      luxTag.textContent = '🌙 35 Lux (Low-Light IR)';
    } else {
      luxTag.textContent = '💡 480 Lux (Studio Nominal)';
    }
  }

  // Occlusion Tag
  const occlTag = $('#spatialOcclLabel');
  if (occlTag) {
    if (target.occlusion === 'heavy') {
      occlTag.textContent = 'Occlusion: Heavy (20% LOS · Rerouted)';
    } else if (target.occlusion === 'partial') {
      occlTag.textContent = 'Occlusion: Partial (58% LOS · Acrylic)';
    } else {
      occlTag.textContent = 'Occlusion: None (100% Sightline)';
    }
  }

  // Orientation Tag
  const orientTag = $('#spatialOrientLabel');
  if (orientTag) {
    if (target.orientation === 'rotated') {
      orientTag.textContent = 'Orientation: Rotated (+45° Yaw)';
    } else if (target.orientation === 'inverted') {
      orientTag.textContent = 'Orientation: Inverted (180° Pitch)';
    } else {
      orientTag.textContent = 'Orientation: Upright (0° Nominal)';
    }
  }

  // Clearance Tag
  const clearTag = $('#spatialClearanceLabel');
  if (clearTag) {
    if (target.occlusion === 'heavy') {
      clearTag.className = 'hud-tag hud-tag-rose';
      clearTag.textContent = 'Clearance: 12mm (Caution)';
    } else if (target.occlusion === 'partial') {
      clearTag.className = 'hud-tag hud-tag-amber';
      clearTag.textContent = 'Clearance: 28mm (Moderate)';
    } else {
      clearTag.className = 'hud-tag hud-tag-emerald';
      clearTag.textContent = 'Clearance: 45mm (Nominal)';
    }
  }

  // Cost Tag
  const costTag = $('#spatialCostLabel');
  if (costTag) {
    let cost = 1.0;
    if (target.occlusion === 'heavy') cost += 0.35;
    if (target.occlusion === 'partial') cost += 0.15;
    if (target.orientation === 'inverted') cost += 0.30;
    if (target.orientation === 'rotated') cost += 0.10;
    if (target.lighting === 'low-light') cost += 0.20;
    costTag.textContent = `Cost: ${cost.toFixed(2)}x`;
  }

  // Sync Quick Override Chips
  $$('.override-subgroup').forEach(group => {
    const cat = group.dataset.overrideCategory;
    const currentVal = target[cat];
    group.querySelectorAll('.override-chip').forEach(chip => {
      chip.classList.toggle('active', chip.dataset.val === currentVal);
    });
  });

  // Modal Telemetry Updates (if open)
  if (spatial.modalOpen) {
    const specSurface = $('#modalSpecSurface');
    if (specSurface) specSurface.textContent = target.environment === 'floor' ? 'Industrial Epoxy Floor' : 'Al-6061 Optical Breadboard';

    const specOrient = $('#modalSpecOrient');
    if (specOrient) specOrient.textContent = `${titleCase(target.orientation)} (${target.orientation === 'rotated' ? '45° Yaw' : target.orientation === 'inverted' ? '180° Inverted' : '0° Upright'})`;

    const specOccl = $('#modalSpecOccl');
    if (specOccl) specOccl.textContent = target.occlusion === 'heavy' ? 'Heavy (20% Sightline · Critical)' : target.occlusion === 'partial' ? 'Partial (58% Sightline · Polycarbonate)' : 'None (100% Sightline · Clear)';

    const specLux = $('#modalSpecLux');
    if (specLux) specLux.textContent = target.lighting === 'bright' ? '1,250 Lux (Overhead Floodlight)' : target.lighting === 'low-light' ? '35 Lux (IR Night Vision Assist)' : '480 Lux (Studio Diffuse)';

    const specCost = $('#modalSpecCost');
    if (specCost) {
      let cost = 1.0;
      if (target.occlusion === 'heavy') cost += 0.35;
      if (target.occlusion === 'partial') cost += 0.15;
      if (target.orientation === 'inverted') cost += 0.30;
      if (target.orientation === 'rotated') cost += 0.10;
      if (target.lighting === 'low-light') cost += 0.20;
      specCost.textContent = `${cost.toFixed(2)}x Multiplier`;
    }
  }
}

function initSpatialVisualizer() {
  const canvas = $('#spatialCanvas');
  const modalCanvas = $('#spatialModalCanvas');
  if (!canvas) return;

  // Initialize Particles
  spatial.particles = [];
  for (let i = 0; i < 30; i++) {
    spatial.particles.push({
      x: (Math.random() - 0.5) * 160,
      y: (Math.random() - 0.5) * 160,
      z: Math.random() * 150 + 10,
      vz: -(Math.random() * 0.4 + 0.2),
      size: Math.random() * 1.5 + 0.8,
      alpha: Math.random() * 0.6 + 0.2
    });
  }

  // Pointer & Touch Controls for 3D Orbiting
  function attachCanvasInteractions(c) {
    if (!c) return;

    let isPinching = false;
    let pinchStartDist = 0;

    const onPointerDown = (e) => {
      spatial.isDragging = true;
      spatial.dragStartX = e.clientX || e.touches?.[0]?.clientX || 0;
      spatial.dragStartY = e.clientY || e.touches?.[0]?.clientY || 0;
      spatial.lastDragX = spatial.dragStartX;
      spatial.lastDragY = spatial.dragStartY;
      spatial.velX = 0;
      spatial.velY = 0;
    };

    const onPointerMove = (e) => {
      // Touch Pinch Zoom Check
      if (e.touches && e.touches.length === 2) {
        isPinching = true;
        const dx = e.touches[0].clientX - e.touches[1].clientX;
        const dy = e.touches[0].clientY - e.touches[1].clientY;
        const dist = Math.hypot(dx, dy);
        if (pinchStartDist > 0) {
          const deltaZoom = (dist - pinchStartDist) * 0.005;
          spatial.targetZoom = Math.min(2.4, Math.max(0.55, spatial.targetZoom + deltaZoom));
        }
        pinchStartDist = dist;
        return;
      }

      if (!spatial.isDragging) return;
      const clientX = e.clientX || e.touches?.[0]?.clientX || 0;
      const clientY = e.clientY || e.touches?.[0]?.clientY || 0;

      const deltaX = clientX - spatial.lastDragX;
      const deltaY = clientY - spatial.lastDragY;

      spatial.velX = deltaX * 0.008;
      spatial.velY = deltaY * 0.008;

      spatial.targetYaw += spatial.velX;
      spatial.targetPitch = Math.min(1.52, Math.max(-0.15, spatial.targetPitch + spatial.velY));

      spatial.lastDragX = clientX;
      spatial.lastDragY = clientY;
    };

    const onPointerUp = () => {
      spatial.isDragging = false;
      isPinching = false;
      pinchStartDist = 0;
    };

    const onWheel = (e) => {
      e.preventDefault();
      const zoomDelta = -e.deltaY * 0.0015;
      spatial.targetZoom = Math.min(2.4, Math.max(0.55, spatial.targetZoom + zoomDelta));
    };

    c.addEventListener('mousedown', onPointerDown);
    c.addEventListener('touchstart', onPointerDown, { passive: true });
    c.addEventListener('wheel', onWheel, { passive: false });
  }

  attachCanvasInteractions(canvas);
  attachCanvasInteractions(modalCanvas);

  window.addEventListener('mousemove', (e) => {
    if (!spatial.isDragging) return;
    const deltaX = e.clientX - spatial.lastDragX;
    const deltaY = e.clientY - spatial.lastDragY;
    spatial.velX = deltaX * 0.008;
    spatial.velY = deltaY * 0.008;
    spatial.targetYaw += spatial.velX;
    spatial.targetPitch = Math.min(1.52, Math.max(-0.15, spatial.targetPitch + spatial.velY));
    spatial.lastDragX = e.clientX;
    spatial.lastDragY = e.clientY;
  });

  window.addEventListener('touchmove', (e) => {
    if (!spatial.isDragging || !e.touches?.[0]) return;
    const clientX = e.touches[0].clientX;
    const clientY = e.touches[0].clientY;
    const deltaX = clientX - spatial.lastDragX;
    const deltaY = clientY - spatial.lastDragY;
    spatial.velX = deltaX * 0.008;
    spatial.velY = deltaY * 0.008;
    spatial.targetYaw += spatial.velX;
    spatial.targetPitch = Math.min(1.52, Math.max(-0.15, spatial.targetPitch + spatial.velY));
    spatial.lastDragX = clientX;
    spatial.lastDragY = clientY;
  }, { passive: true });

  window.addEventListener('mouseup', () => { spatial.isDragging = false; });
  window.addEventListener('touchend', () => { spatial.isDragging = false; });

  // Camera Preset Helpers
  function setCameraPreset(mode) {
    spatial.cameraMode = mode;
    $$('[data-cam], [data-modal-cam]').forEach(btn => {
      btn.classList.toggle('active', btn.dataset.cam === mode || btn.dataset.modalCam === mode);
    });

    if (mode === 'iso') {
      spatial.targetYaw = 0.65;
      spatial.targetPitch = 0.46;
      spatial.targetZoom = 1.0;
    } else if (mode === 'top') {
      spatial.targetYaw = 0.0;
      spatial.targetPitch = 1.54; // ~88.5 degrees top-down
      spatial.targetZoom = 1.05;
    } else if (mode === 'front') {
      spatial.targetYaw = 0.0;
      spatial.targetPitch = 0.04; // ~2 degrees side elevation
      spatial.targetZoom = 1.0;
    } else if (mode === 'tcp') {
      spatial.targetYaw = 0.0;
      spatial.targetPitch = 1.48; // looking down gripper tool center point
      spatial.targetZoom = 1.55;
    }
  }

  // Hook Camera Preset Buttons
  $$('[data-cam]').forEach(btn => {
    btn.addEventListener('click', () => setCameraPreset(btn.dataset.cam));
  });
  $$('[data-modal-cam]').forEach(btn => {
    btn.addEventListener('click', () => setCameraPreset(btn.dataset.modalCam));
  });

  // Grasp Simulation Toggle
  function toggleGraspSim() {
    spatial.isSimGrasp = !spatial.isSimGrasp;
    const isPlaying = spatial.isSimGrasp;
    
    $('#spatialSimGraspBtn')?.classList.toggle('active', isPlaying);
    $('#modalSimGraspBtn')?.classList.toggle('active', isPlaying);
    
    const label = isPlaying ? 'Grasp Sim' : 'Paused';
    if ($('#spatialSimGraspText')) $('#spatialSimGraspText').textContent = label;

    toast(isPlaying ? 'Grasp approach cycle active' : 'Grasp simulation paused');
  }

  $('#spatialSimGraspBtn')?.addEventListener('click', toggleGraspSim);
  $('#modalSimGraspBtn')?.addEventListener('click', toggleGraspSim);

  // Auto-Orbit Toggle
  function toggleAutoOrbit() {
    spatial.autoOrbit = !spatial.autoOrbit;
    const isOrbiting = spatial.autoOrbit;
    $('#spatialAutoOrbitBtn')?.classList.toggle('active', isOrbiting);
    $('#modalAutoOrbitBtn')?.classList.toggle('active', isOrbiting);
    toast(isOrbiting ? 'Auto-orbit turntable enabled' : 'Auto-orbit paused');
  }

  $('#spatialAutoOrbitBtn')?.addEventListener('click', toggleAutoOrbit);
  $('#modalAutoOrbitBtn')?.addEventListener('click', toggleAutoOrbit);

  // Phone Gyroscope Tilt-to-Orbit
  async function toggleGyroOrbit() {
    if (!spatial.gyroActive) {
      if (typeof DeviceOrientationEvent !== 'undefined' && typeof DeviceOrientationEvent.requestPermission === 'function') {
        try {
          const perm = await DeviceOrientationEvent.requestPermission();
          if (perm !== 'granted') {
            toast('Gyroscope permission denied');
            return;
          }
        } catch (err) {
          console.warn('Gyro permission error:', err);
        }
      }

      spatial.gyroActive = true;
      spatial.gyroBaseline = { beta: null, gamma: null };
      $('#spatialGyroBtn')?.classList.add('active');
      $('#modalGyroBtn')?.classList.add('active');
      toast('📱 Gyroscope active · Tilt phone to orbit 3D workcell');
    } else {
      spatial.gyroActive = false;
      spatial.gyroBaseline = { beta: null, gamma: null };
      $('#spatialGyroBtn')?.classList.remove('active');
      $('#modalGyroBtn')?.classList.remove('active');
      toast('Gyroscope orbit disabled');
    }
  }

  function handleDeviceOrientation(e) {
    if (!spatial.gyroActive || spatial.isDragging) return;
    const beta = e.beta;   // front/back tilt (-180 to 180)
    const gamma = e.gamma; // left/right tilt (-90 to 90)

    if (beta === null || gamma === null) return;

    if (spatial.gyroBaseline.beta === null) {
      spatial.gyroBaseline.beta = beta;
      spatial.gyroBaseline.gamma = gamma;
      return;
    }

    const deltaGamma = gamma - spatial.gyroBaseline.gamma;
    const deltaBeta = beta - spatial.gyroBaseline.beta;

    // Smooth gyro update with deadband
    if (Math.abs(deltaGamma) > 0.8) {
      spatial.targetYaw = 0.65 + deltaGamma * 0.032;
    }
    if (Math.abs(deltaBeta) > 0.8) {
      spatial.targetPitch = Math.min(1.52, Math.max(-0.15, 0.46 + deltaBeta * 0.024));
    }
  }

  window.addEventListener('deviceorientation', handleDeviceOrientation, { passive: true });
  $('#spatialGyroBtn')?.addEventListener('click', toggleGyroOrbit);
  $('#modalGyroBtn')?.addEventListener('click', toggleGyroOrbit);

  // View Mode Cycle (Shaded -> Wireframe -> Collision -> Shaded)
  function cycleViewMode() {
    const modes = ['shaded', 'wireframe', 'collision'];
    const nextIdx = (modes.indexOf(spatial.renderMode) + 1) % modes.length;
    spatial.renderMode = modes[nextIdx];

    const label = titleCase(spatial.renderMode);
    if ($('#spatialViewModeText')) $('#spatialViewModeText').textContent = label;
    if ($('#modalRenderModeText')) $('#modalRenderModeText').textContent = label;

    toast(`Render mode: ${label}`);
  }

  $('#spatialViewModeBtn')?.addEventListener('click', cycleViewMode);
  $('#modalRenderModeBtn')?.addEventListener('click', cycleViewMode);

  // Reset Camera
  function resetCamera() {
    setCameraPreset('iso');
    toast('Camera reset to Isometric 3D');
  }

  $('#spatialResetCamBtn')?.addEventListener('click', resetCamera);

  // Quick Override Setup Lab Listeners
  $$('.override-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      const parentGroup = chip.closest('.override-subgroup');
      if (!parentGroup) return;
      const category = parentGroup.dataset.overrideCategory;
      const val = chip.dataset.val;

      if (!spatial.override) {
        spatial.override = {
          environment: recommendation?.environment || 'bench',
          orientation: recommendation?.orientation || 'upright',
          occlusion: recommendation?.occlusion || 'none',
          lighting: recommendation?.lighting || 'normal'
        };
      }

      spatial.override[category] = val;
      updateSpatialHUD();
      toast(`Simulating: ${titleCase(category)} = ${titleCase(val)}`);
    });
  });

  $('#spatialResetOverrideBtn')?.addEventListener('click', () => {
    spatial.override = null;
    updateSpatialHUD();
    toast('Preview reset to active recommended target');
  });

  // Modal Dialog Open / Close
  function openSpatialModal() {
    spatial.modalOpen = true;
    const modal = $('#spatialInspectorModal');
    if (modal) {
      modal.hidden = false;
      document.body.style.overflow = 'hidden';
      updateSpatialHUD();
    }
  }

  function closeSpatialModal() {
    spatial.modalOpen = false;
    const modal = $('#spatialInspectorModal');
    if (modal) {
      modal.hidden = true;
      document.body.style.overflow = '';
    }
  }

  $('#spatialExpandBtn')?.addEventListener('click', openSpatialModal);
  $('#mobilePreviewBtn')?.addEventListener('click', openSpatialModal);
  $('#closeSpatialModal')?.addEventListener('click', closeSpatialModal);
  $('#spatialModalBackdrop')?.addEventListener('click', closeSpatialModal);

  $('#modalRecordBtn')?.addEventListener('click', () => {
    closeSpatialModal();
    beginCapture();
  });

  $('#fullscreenSimulationBtn')?.addEventListener('click', async () => {
    const modal = $('#spatialInspectorModal');
    const button = $('#fullscreenSimulationBtn');
    const entering = !modal?.classList.contains('simulation-focus');
    modal?.classList.toggle('simulation-focus', entering);
    button?.setAttribute('aria-pressed', String(entering));
    const label = button?.querySelector('span');
    if (label) label.textContent = entering ? 'Show details' : 'Full screen';
    try {
      if (entering && modal?.requestFullscreen && !document.fullscreenElement) await modal.requestFullscreen();
      if (!entering && document.fullscreenElement) await document.exitFullscreen();
    } catch (error) {
      console.info('Browser fullscreen unavailable; using in-app fullscreen.', error);
    }
  });

  document.addEventListener('fullscreenchange', () => {
    if (!document.fullscreenElement) {
      $('#spatialInspectorModal')?.classList.remove('simulation-focus');
      $('#fullscreenSimulationBtn')?.setAttribute('aria-pressed', 'false');
      const label = $('#fullscreenSimulationBtn span');
      if (label) label.textContent = 'Full screen';
    }
  });

  // Keyboard Shortcuts
  window.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    if (e.code === 'Space') {
      e.preventDefault();
      toggleGraspSim();
    } else if (e.key === 'r' || e.key === 'R') {
      resetCamera();
    } else if (e.key === 'o' || e.key === 'O') {
      toggleAutoOrbit();
    } else if (e.key === 'm' || e.key === 'M') {
      cycleViewMode();
    } else if (e.key === 'Escape' && spatial.modalOpen) {
      closeSpatialModal();
    }
  });

  // Start Animation Loop
  if (spatial.animId) cancelAnimationFrame(spatial.animId);
  spatialLoop();
}

/* ==========================================================================
   Master 3D Rendering & Animation Loop (60 FPS)
   ========================================================================== */
function spatialLoop(now = performance.now()) {
  spatial.animId = requestAnimationFrame(spatialLoop);

  // FPS calculation
  spatial.fpsCount++;
  if (now - spatial.fpsTimer >= 1000) {
    spatial.fps = spatial.fpsCount;
    spatial.fpsCount = 0;
    spatial.fpsTimer = now;
    const fpsLabel = $('#spatialFpsLabel');
    if (fpsLabel) fpsLabel.textContent = `${spatial.fps} FPS`;
  }

  // Camera smooth interpolation & inertia damping
  if (!spatial.isDragging) {
    spatial.velX *= 0.92;
    spatial.velY *= 0.92;
    spatial.targetYaw += spatial.velX;
    spatial.targetPitch = Math.min(1.52, Math.max(-0.15, spatial.targetPitch + spatial.velY));
  }

  if (spatial.autoOrbit && !spatial.isDragging) {
    spatial.targetYaw += spatial.orbitSpeed;
  }

  spatial.yaw += (spatial.targetYaw - spatial.yaw) * 0.14;
  spatial.pitch += (spatial.targetPitch - spatial.pitch) * 0.14;
  spatial.zoom += (spatial.targetZoom - spatial.zoom) * 0.14;

  // Grasp Simulation Progress Cycle
  if (spatial.isSimGrasp) {
    spatial.graspProgress = (spatial.graspProgress + spatial.graspSpeed) % 1.0;
  }

  // Update particles
  for (const p of spatial.particles) {
    p.z += p.vz;
    if (p.z <= 0) {
      p.z = 160;
      p.x = (Math.random() - 0.5) * 160;
      p.y = (Math.random() - 0.5) * 160;
    }
  }

  // Render Main Canvas
  const canvas = $('#spatialCanvas');
  if (canvas && canvas.offsetParent !== null) {
    render3DWorkcellScene(canvas, false);
  }

  // Render Modal Canvas (if open)
  if (spatial.modalOpen) {
    const modalCanvas = $('#spatialModalCanvas');
    if (modalCanvas) {
      render3DWorkcellScene(modalCanvas, true);
    }
  }
}

function drawSpatialVisualizer() {
  updateSpatialHUD();
  const canvas = $('#spatialCanvas');
  if (canvas) render3DWorkcellScene(canvas, false);
}

/* ==========================================================================
   3D Projection & Polygonal Geometry Engine
   ========================================================================== */
function render3DWorkcellScene(canvas, isModal = false) {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const rect = canvas.getBoundingClientRect();
  const targetW = Math.round(rect.width * dpr);
  const targetH = Math.round(rect.height * dpr);

  if (canvas.width !== targetW || canvas.height !== targetH) {
    canvas.width = targetW;
    canvas.height = targetH;
  }

  ctx.save();
  ctx.scale(dpr, dpr);

  const w = rect.width;
  const h = rect.height;
  ctx.clearRect(0, 0, w, h);

  const cx = w / 2;
  const cy = h / 2 + (isModal ? 25 : 20);
  const zoom = spatial.zoom * (isModal ? 1.25 : 0.95);
  const yaw = spatial.yaw;
  const pitch = spatial.pitch;

  const target = getActiveSpatialConfig();
  const t = spatial.graspProgress;

  // 3D Projection Helper
  function project(x, y, z) {
    // 1. Yaw rotation around Z axis
    const cosY = Math.cos(yaw);
    const sinY = Math.sin(yaw);
    const x1 = x * cosY - y * sinY;
    const y1 = x * sinY + y * cosY;
    const z1 = z;

    // 2. Pitch rotation around camera X axis
    const cosP = Math.cos(pitch);
    const sinP = Math.sin(pitch);
    const x2 = x1;
    const y2 = y1 * cosP - z1 * sinP;
    const z2 = y1 * sinP + z1 * cosP;

    // 3. Perspective Camera
    const cameraDist = 360;
    const focal = 310 * zoom;
    const depth = cameraDist - y2;
    const safeDepth = Math.max(depth, 20);
    const scale = focal / safeDepth;

    return {
      x: cx + x2 * scale,
      y: cy - z2 * scale,
      z: y2, // for depth sorting
      scale: scale,
      visible: depth > 10
    };
  }

  // Directional Lighting Vector
  const lightPos = [70, -110, 190];
  const isWireframe = spatial.renderMode === 'wireframe';
  const isCollision = spatial.renderMode === 'collision';

  // Atmosphere & Light Intensity
  let lightAlpha = 0.65;
  if (target.lighting === 'bright') lightAlpha = 1.0;
  if (target.lighting === 'low-light') lightAlpha = 0.35;

  /* --------------------------------------------------------------------------
     1. Ground Workcell (Bench / Floor)
     -------------------------------------------------------------------------- */
  const isFloor = target.environment === 'floor';
  const groundHalf = 135;

  // Ground Slab Polygon
  const gPts = [
    project(-groundHalf, -groundHalf, 0),
    project(groundHalf, -groundHalf, 0),
    project(groundHalf, groundHalf, 0),
    project(-groundHalf, groundHalf, 0)
  ];

  ctx.beginPath();
  ctx.moveTo(gPts[0].x, gPts[0].y);
  for (let i = 1; i < gPts.length; i++) ctx.lineTo(gPts[i].x, gPts[i].y);
  ctx.closePath();

  if (isFloor) {
    // Dark epoxy factory floor
    ctx.fillStyle = '#080c12';
    ctx.fill();
    ctx.strokeStyle = '#1e293b';
    ctx.lineWidth = 1.2;
    ctx.stroke();

    // Safety Hazard Perimeter Chevron Stripes
    ctx.save();
    ctx.strokeStyle = `rgba(245, 158, 11, ${0.4 * lightAlpha})`;
    ctx.lineWidth = 1.5;
    ctx.setLineDash([8, 8]);
    ctx.beginPath();
    ctx.moveTo(gPts[0].x, gPts[0].y);
    ctx.lineTo(gPts[1].x, gPts[1].y);
    ctx.lineTo(gPts[2].x, gPts[2].y);
    ctx.lineTo(gPts[3].x, gPts[3].y);
    ctx.closePath();
    ctx.stroke();
    ctx.restore();

    // Floor Concentric Safety Radius Circle
    for (const r of [40, 80, 120]) {
      ctx.beginPath();
      ctx.strokeStyle = `rgba(56, 189, 248, ${0.12 * lightAlpha})`;
      ctx.lineWidth = 1;
      for (let a = 0; a <= Math.PI * 2; a += 0.2) {
        const pt = project(Math.cos(a) * r, Math.sin(a) * r, 0);
        if (a === 0) ctx.moveTo(pt.x, pt.y);
        else ctx.lineTo(pt.x, pt.y);
      }
      ctx.closePath();
      ctx.stroke();
    }
  } else {
    // Brushed Anodized Aluminum Bench
    const grad = ctx.createLinearGradient(gPts[0].x, gPts[0].y, gPts[2].x, gPts[2].y);
    grad.addColorStop(0, '#0c131d');
    grad.addColorStop(0.5, '#131e2e');
    grad.addColorStop(1, '#090f17');
    ctx.fillStyle = grad;
    ctx.fill();
    ctx.strokeStyle = `rgba(56, 189, 248, ${0.25 * lightAlpha})`;
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // M6 Threaded Fixture Holes Grid Matrix
    for (let x = -100; x <= 100; x += 25) {
      for (let y = -100; y <= 100; y += 25) {
        const holeCenter = project(x, y, 0);
        const holeRadius = 2.4 * holeCenter.scale;
        ctx.beginPath();
        ctx.fillStyle = '#05070a';
        ctx.arc(holeCenter.x, holeCenter.y, holeRadius, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = `rgba(255, 255, 255, ${0.15 * lightAlpha})`;
        ctx.lineWidth = 0.8;
        ctx.stroke();
      }
    }

    // Centimeter Grid Lines
    ctx.strokeStyle = `rgba(56, 189, 248, ${0.08 * lightAlpha})`;
    ctx.lineWidth = 0.8;
    for (let x = -100; x <= 100; x += 50) {
      const p1 = project(x, -100, 0);
      const p2 = project(x, 100, 0);
      ctx.beginPath();
      ctx.moveTo(p1.x, p1.y);
      ctx.lineTo(p2.x, p2.y);
      ctx.stroke();
    }
    for (let y = -100; y <= 100; y += 50) {
      const p1 = project(-100, y, 0);
      const p2 = project(100, y, 0);
      ctx.beginPath();
      ctx.moveTo(p1.x, p1.y);
      ctx.lineTo(p2.x, p2.y);
      ctx.stroke();
    }
  }

  // 3D Coordinate Origin Axis Triad (Red = +X, Green = +Y, Blue = +Z)
  const o = project(0, 0, 0);
  const axX = project(35, 0, 0);
  const axY = project(0, 35, 0);
  const axZ = project(0, 0, 35);

  // +X (Red)
  ctx.strokeStyle = '#f43f5e';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(o.x, o.y);
  ctx.lineTo(axX.x, axX.y);
  ctx.stroke();

  // +Y (Green)
  ctx.strokeStyle = '#10b981';
  ctx.beginPath();
  ctx.moveTo(o.x, o.y);
  ctx.lineTo(axY.x, axY.y);
  ctx.stroke();

  // +Z (Blue)
  ctx.strokeStyle = '#38bdf8';
  ctx.beginPath();
  ctx.moveTo(o.x, o.y);
  ctx.lineTo(axZ.x, axZ.y);
  ctx.stroke();

  /* --------------------------------------------------------------------------
     2. Grasp Cycle Kinematics State
     -------------------------------------------------------------------------- */
  let gripperZ = 120;
  let gripperGap = 66;
  let objZ = 0;
  let phaseText = 'STANDBY · HOVER 120mm';
  let gripperLedColor = '#38bdf8';

  // 4-Phase Grasp Timeline
  if (t < 0.25) {
    // Phase 0: Standby Hover
    const sub = t / 0.25;
    gripperZ = 120;
    gripperGap = 66;
    objZ = 0;
    phaseText = 'PHASE 1/4 · STANDBY HOVER (120mm)';
    gripperLedColor = '#10b981';
  } else if (t < 0.55) {
    // Phase 1: Descending Approach
    const sub = (t - 0.25) / 0.30;
    const ease = sub * sub * (3 - 2 * sub);
    gripperZ = 120 - ease * 82; // descends to 38mm
    gripperGap = 66;
    objZ = 0;
    phaseText = `PHASE 2/4 · APPROACHING TARGET (${Math.round(gripperZ)}mm)`;
    gripperLedColor = '#f59e0b';
  } else if (t < 0.70) {
    // Phase 2: Contact & Clamping
    const sub = (t - 0.55) / 0.15;
    const ease = sub * sub * (3 - 2 * sub);
    gripperZ = 38;
    gripperGap = 66 - ease * 22; // clamps to 44mm
    objZ = 0;
    phaseText = 'PHASE 3/4 · GRASPING MANIPULAND (18.5 N)';
    gripperLedColor = '#38bdf8';
  } else if (t < 0.88) {
    // Phase 3: Lift & Verify Stability
    const sub = (t - 0.70) / 0.18;
    const ease = Math.sin(sub * Math.PI);
    objZ = ease * 26;
    gripperZ = 38 + objZ;
    gripperGap = 44;
    phaseText = `PHASE 4/4 · LIFT & VERIFY (+${Math.round(objZ)}mm)`;
    gripperLedColor = '#34d399';
  } else {
    // Phase 4: Release & Ascend Reset
    const sub = (t - 0.88) / 0.12;
    objZ = 0;
    gripperZ = 38 + sub * 82;
    gripperGap = 44 + sub * 22;
    phaseText = 'CYCLE COMPLETE · RESETTING HOVER';
    gripperLedColor = '#10b981';
  }

  const phaseBannerText = $('#spatialGraspPhaseText');
  if (phaseBannerText) phaseBannerText.textContent = phaseText;

  // Update Modal Kinematics values
  if (isModal) {
    const vx = $('#modalValX'); if (vx) vx.textContent = '0.0 mm';
    const vy = $('#modalValY'); if (vy) vy.textContent = '0.0 mm';
    const vz = $('#modalValZ'); if (vz) vz.textContent = `${gripperZ.toFixed(1)} mm`;
    const vyaw = $('#modalValYaw'); if (vyaw) vyaw.textContent = `${target.orientation === 'rotated' ? '45.0°' : '0.0°'}`;
    const vpitch = $('#modalValPitch'); if (vpitch) vpitch.textContent = `${target.orientation === 'inverted' ? '180.0°' : '0.0°'}`;
    const vroll = $('#modalValRoll'); if (vroll) vroll.textContent = '0.0°';
  }

  /* --------------------------------------------------------------------------
     3. Drop Shadows (Ground Contact Shading)
     -------------------------------------------------------------------------- */
  // Object Drop Shadow
  const shadowAlpha = (0.55 - (objZ / 60)) * lightAlpha;
  if (shadowAlpha > 0.05) {
    const sRad = (24 + objZ * 0.2) * zoom;
    const sCenter = project(0, 0, 0);
    ctx.save();
    ctx.beginPath();
    ctx.ellipse(sCenter.x, sCenter.y, sRad * 1.1, sRad * 0.55, 0, 0, Math.PI * 2);
    ctx.fillStyle = `rgba(0, 0, 0, ${shadowAlpha})`;
    ctx.fill();
    ctx.restore();
  }

  // Gripper Shadow when approaching
  if (gripperZ < 90) {
    const gShadowAlpha = ((90 - gripperZ) / 90) * 0.35 * lightAlpha;
    const gShadowCenter = project(0, 0, 0);
    ctx.beginPath();
    ctx.ellipse(gShadowCenter.x, gShadowCenter.y, 35 * zoom, 18 * zoom, 0, 0, Math.PI * 2);
    ctx.fillStyle = `rgba(56, 189, 248, ${gShadowAlpha * 0.3})`;
    ctx.fill();
  }

  /* --------------------------------------------------------------------------
     4. Target Manipuland (3D Object Billet Geometry)
     -------------------------------------------------------------------------- */
  // Object Dimensions
  const halfW = 22;
  const halfD = 22;
  const objH = 38;

  // Local object transformation based on orientation
  function transformObjPoint(lx, ly, lz) {
    let px = lx;
    let py = ly;
    let pz = lz;

    if (target.orientation === 'rotated') {
      const rot = Math.PI / 4; // 45 degrees
      const cosR = Math.cos(rot);
      const sinR = Math.sin(rot);
      const rx = px * cosR - py * sinR;
      const ry = px * sinR + py * cosR;
      px = rx;
      py = ry;
    } else if (target.orientation === 'inverted') {
      // 180 degrees flip around X axis
      pz = objH - lz;
    }

    return project(px, py, pz + objZ);
  }

  // 8 Bounding Box Vertices
  const v = [
    transformObjPoint(-halfW, -halfD, 0),     // 0: Bottom-Front-Left
    transformObjPoint(halfW, -halfD, 0),      // 1: Bottom-Front-Right
    transformObjPoint(halfW, halfD, 0),       // 2: Bottom-Back-Right
    transformObjPoint(-halfW, halfD, 0),      // 3: Bottom-Back-Left
    transformObjPoint(-halfW, -halfD, objH),  // 4: Top-Front-Left
    transformObjPoint(halfW, -halfD, objH),   // 5: Top-Front-Right
    transformObjPoint(halfW, halfD, objH),    // 6: Top-Back-Right
    transformObjPoint(-halfW, halfD, objH)    // 7: Top-Back-Left
  ];

  // 6 Quad Faces [v0, v1, v2, v3, normalName, colorBase]
  const faces = [
    { pts: [v[4], v[5], v[6], v[7]], name: 'top', norm: [0, 0, 1] },
    { pts: [v[0], v[1], v[5], v[4]], name: 'front', norm: [0, -1, 0] },
    { pts: [v[1], v[2], v[6], v[5]], name: 'right', norm: [1, 0, 0] },
    { pts: [v[2], v[3], v[7], v[6]], name: 'back', norm: [0, 1, 0] },
    { pts: [v[3], v[0], v[4], v[7]], name: 'left', norm: [-1, 0, 0] },
    { pts: [v[3], v[2], v[1], v[0]], name: 'bottom', norm: [0, 0, -1] }
  ];

  // Render Object Faces with Lambertian Diffuse Shading
  const baseColor = isFloor ? [6, 182, 212] : [16, 185, 129]; // Cyan or Emerald

  for (const face of faces) {
    const p = face.pts;
    // 2D Signed Area Backface Test
    const area = (p[1].x - p[0].x) * (p[2].y - p[0].y) - (p[1].y - p[0].y) * (p[2].x - p[0].x);
    if (area <= 0 && spatial.cameraMode !== 'wireframe') continue;

    ctx.beginPath();
    ctx.moveTo(p[0].x, p[0].y);
    for (let i = 1; i < p.length; i++) ctx.lineTo(p[i].x, p[i].y);
    ctx.closePath();

    if (isWireframe) {
      ctx.strokeStyle = '#38bdf8';
      ctx.lineWidth = 1.2;
      ctx.stroke();
    } else if (isCollision) {
      ctx.fillStyle = 'rgba(16, 185, 129, 0.4)';
      ctx.fill();
      ctx.strokeStyle = '#10b981';
      ctx.lineWidth = 1.5;
      ctx.stroke();
    } else {
      // Calculate lighting intensity
      let diffuse = 0.5;
      if (face.name === 'top') diffuse = 0.95;
      if (face.name === 'front') diffuse = 0.75;
      if (face.name === 'right') diffuse = 0.60;
      if (face.name === 'left') diffuse = 0.45;
      if (face.name === 'back') diffuse = 0.35;

      const r = Math.round(baseColor[0] * diffuse * lightAlpha);
      const g = Math.round(baseColor[1] * diffuse * lightAlpha);
      const b = Math.round(baseColor[2] * diffuse * lightAlpha);

      ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
      ctx.fill();
      ctx.strokeStyle = `rgba(255, 255, 255, ${0.4 * lightAlpha})`;
      ctx.lineWidth = 1.2;
      ctx.stroke();
    }
  }

  // Top Face Fiducial ArUco / AprilTag Marker (when upright)
  if (target.orientation === 'upright' && !isWireframe) {
    const topCenter = transformObjPoint(0, 0, objH + 0.5);
    const tagSize = 10 * topCenter.scale;
    ctx.save();
    ctx.fillStyle = '#080c10';
    ctx.beginPath();
    ctx.rect(topCenter.x - tagSize, topCenter.y - tagSize * 0.6, tagSize * 2, tagSize * 1.2);
    ctx.fill();
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 1;
    ctx.stroke();

    // Center Crosshair
    ctx.fillStyle = '#10b981';
    ctx.beginPath();
    ctx.arc(topCenter.x, topCenter.y, 2.5 * topCenter.scale, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }

  // Rotated Yaw Arc Marker
  if (target.orientation === 'rotated') {
    const arcCenter = project(0, 0, 2);
    ctx.save();
    ctx.strokeStyle = '#fbbf24';
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let a = 0; a <= Math.PI / 4; a += 0.05) {
      const pt = project(Math.cos(a) * 34, Math.sin(a) * 34, 2);
      if (a === 0) ctx.moveTo(pt.x, pt.y);
      else ctx.lineTo(pt.x, pt.y);
    }
    ctx.stroke();
    ctx.restore();
  }

  // Inverted Pose Caution Hash
  if (target.orientation === 'inverted' && !isWireframe) {
    const topCenter = transformObjPoint(0, 0, objH + 0.5);
    ctx.save();
    ctx.fillStyle = 'rgba(244, 63, 94, 0.85)';
    ctx.beginPath();
    ctx.arc(topCenter.x, topCenter.y, 6 * topCenter.scale, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 1.5;
    ctx.stroke();
    ctx.restore();
  }

  /* --------------------------------------------------------------------------
     5. Obstacle & Occlusion Simulation (3D Partition / Fixture)
     -------------------------------------------------------------------------- */
  if (target.occlusion !== 'none') {
    const isHeavy = target.occlusion === 'heavy';
    const ox = isHeavy ? 14 : 22;
    const oy = isHeavy ? -18 : -24;
    const oz = 0;
    const ow = isHeavy ? 36 : 14;
    const od = isHeavy ? 36 : 48;
    const oh = isHeavy ? 74 : 56;

    // Obstacle 8 Vertices
    const obV = [
      project(ox - ow / 2, oy - od / 2, oz),
      project(ox + ow / 2, oy - od / 2, oz),
      project(ox + ow / 2, oy + od / 2, oz),
      project(ox - ow / 2, oy + od / 2, oz),
      project(ox - ow / 2, oy - od / 2, oz + oh),
      project(ox + ow / 2, oy - od / 2, oz + oh),
      project(ox + ow / 2, oy + od / 2, oz + oh),
      project(ox - ow / 2, oy + od / 2, oz + oh)
    ];

    const obFaces = [
      { pts: [obV[4], obV[5], obV[6], obV[7]], name: 'top' },
      { pts: [obV[0], obV[1], obV[5], obV[4]], name: 'front' },
      { pts: [obV[1], obV[2], obV[6], obV[5]], name: 'right' },
      { pts: [obV[2], obV[3], obV[7], obV[6]], name: 'back' },
      { pts: [obV[3], obV[0], obV[4], obV[7]], name: 'left' }
    ];

    for (const f of obFaces) {
      const p = f.pts;
      const area = (p[1].x - p[0].x) * (p[2].y - p[0].y) - (p[1].y - p[0].y) * (p[2].x - p[0].x);
      if (area <= 0 && spatial.renderMode !== 'wireframe') continue;

      ctx.beginPath();
      ctx.moveTo(p[0].x, p[0].y);
      for (let i = 1; i < p.length; i++) ctx.lineTo(p[i].x, p[i].y);
      ctx.closePath();

      if (isHeavy) {
        // Red solid industrial fixture
        ctx.fillStyle = `rgba(244, 63, 94, ${0.7 * lightAlpha})`;
        ctx.fill();
        ctx.strokeStyle = '#fb7185';
        ctx.lineWidth = 1.5;
        ctx.stroke();

        // Pulsating Collision Envelope Wireframe
        ctx.save();
        ctx.strokeStyle = `rgba(244, 63, 94, ${0.8 + Math.sin(Date.now() / 150) * 0.2})`;
        ctx.lineWidth = 2;
        ctx.setLineDash([4, 4]);
        ctx.stroke();
        ctx.restore();
      } else {
        // Tinted translucent polycarbonate screen
        ctx.fillStyle = `rgba(245, 158, 11, ${0.28 * lightAlpha})`;
        ctx.fill();
        ctx.strokeStyle = 'rgba(251, 191, 36, 0.8)';
        ctx.lineWidth = 1.2;
        ctx.stroke();
      }
    }

    // Curved Approach Spline trajectory if Heavy obstacle
    if (isHeavy) {
      ctx.save();
      ctx.strokeStyle = '#fb7185';
      ctx.lineWidth = 2;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      const pStart = project(-30, -30, 120);
      const pMid = project(-15, 0, 80);
      const pEnd = project(0, 0, objH + objZ);
      ctx.moveTo(pStart.x, pStart.y);
      ctx.quadraticCurveTo(pMid.x, pMid.y, pEnd.x, pEnd.y);
      ctx.stroke();
      ctx.restore();
    }
  }

  /* --------------------------------------------------------------------------
     6. Articulated Robotic Gripper (Parallel Jaw End-Effector)
     -------------------------------------------------------------------------- */
  const gripCenter = project(0, 0, gripperZ);
  const jawHalf = gripperGap / 2;

  // Gripper Tool Flange (Top Disc)
  const flangeP1 = project(-18, 0, gripperZ + 36);
  const flangeP2 = project(18, 0, gripperZ + 36);
  ctx.save();
  ctx.strokeStyle = '#64748b';
  ctx.lineWidth = 3 * gripCenter.scale;
  ctx.beginPath();
  ctx.moveTo(flangeP1.x, flangeP1.y);
  ctx.lineTo(flangeP2.x, flangeP2.y);
  ctx.stroke();
  ctx.restore();

  // Gripper Main Housing (Servo Body)
  const b1 = project(-16, -10, gripperZ + 16);
  const b2 = project(16, -10, gripperZ + 16);
  const b3 = project(16, 10, gripperZ + 16);
  const b4 = project(-16, 10, gripperZ + 16);
  const bTop = project(0, 0, gripperZ + 34);

  ctx.save();
  ctx.fillStyle = '#0f172a';
  ctx.beginPath();
  ctx.moveTo(b1.x, b1.y);
  ctx.lineTo(b2.x, b2.y);
  ctx.lineTo(b3.x, b3.y);
  ctx.lineTo(b4.x, b4.y);
  ctx.closePath();
  ctx.fill();
  ctx.strokeStyle = '#38bdf8';
  ctx.lineWidth = 1.5;
  ctx.stroke();

  // Gripper Status LED Beacon
  const ledPt = project(0, -10, gripperZ + 25);
  ctx.fillStyle = gripperLedColor;
  ctx.shadowColor = gripperLedColor;
  ctx.shadowBlur = 10;
  ctx.beginPath();
  ctx.arc(ledPt.x, ledPt.y, 3 * gripCenter.scale, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();

  // Articulated Parallel Jaw Fingers
  const leftFingerBase = project(-jawHalf, 0, gripperZ + 16);
  const leftFingerTip = project(-jawHalf, 0, gripperZ);
  const rightFingerBase = project(jawHalf, 0, gripperZ + 16);
  const rightFingerTip = project(jawHalf, 0, gripperZ);

  ctx.save();
  ctx.strokeStyle = '#94a3b8';
  ctx.lineWidth = 4 * gripCenter.scale;
  ctx.lineCap = 'round';

  // Left Finger
  ctx.beginPath();
  ctx.moveTo(leftFingerBase.x, leftFingerBase.y);
  ctx.lineTo(leftFingerTip.x, leftFingerTip.y);
  ctx.stroke();

  // Right Finger
  ctx.beginPath();
  ctx.moveTo(rightFingerBase.x, rightFingerBase.y);
  ctx.lineTo(rightFingerTip.x, rightFingerTip.y);
  ctx.stroke();

  // Rubber Friction Pads on Inner Finger Faces
  ctx.strokeStyle = '#0284c7';
  ctx.lineWidth = 2.5 * gripCenter.scale;
  ctx.beginPath();
  ctx.moveTo(project(-jawHalf + 2, 0, gripperZ + 12).x, project(-jawHalf + 2, 0, gripperZ + 12).y);
  ctx.lineTo(project(-jawHalf + 2, 0, gripperZ).x, project(-jawHalf + 2, 0, gripperZ).y);
  ctx.moveTo(project(jawHalf - 2, 0, gripperZ + 12).x, project(jawHalf - 2, 0, gripperZ + 12).y);
  ctx.lineTo(project(jawHalf - 2, 0, gripperZ).x, project(jawHalf - 2, 0, gripperZ).y);
  ctx.stroke();
  ctx.restore();

  // Pulsating Laser Rangefinder Line to Workpiece
  const tcpPt = project(0, 0, gripperZ);
  const targetTopPt = project(0, 0, objH + objZ);

  ctx.save();
  ctx.strokeStyle = `rgba(56, 189, 248, ${0.7 * lightAlpha})`;
  ctx.lineWidth = 1.5;
  ctx.setLineDash([3, 3]);
  ctx.beginPath();
  ctx.moveTo(tcpPt.x, tcpPt.y);
  ctx.lineTo(targetTopPt.x, targetTopPt.y);
  ctx.stroke();

  // Laser Landing Crosshair
  ctx.fillStyle = '#38bdf8';
  ctx.beginPath();
  ctx.arc(targetTopPt.x, targetTopPt.y, 3.5, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();

  // Haptic Touch Pulse Rings on Grasp (t in 0.55 to 0.72)
  if (t >= 0.55 && t <= 0.72) {
    const pulseRad = ((t - 0.55) / 0.17) * 18 * zoom;
    const pulseAlpha = 1.0 - ((t - 0.55) / 0.17);
    ctx.save();
    ctx.strokeStyle = `rgba(56, 189, 248, ${pulseAlpha})`;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(targetTopPt.x, targetTopPt.y, pulseRad, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();
  }

  /* --------------------------------------------------------------------------
     7. Volumetric Atmosphere & Particles (Bright Mode)
     -------------------------------------------------------------------------- */
  if (target.lighting === 'bright') {
    // Overhead Spotlight Photon Cone
    const spotOrigin = project(lightPos[0], lightPos[1], lightPos[2]);
    const spotBase = project(0, 0, 0);

    ctx.save();
    const coneGrad = ctx.createRadialGradient(spotOrigin.x, spotOrigin.y, 10, spotBase.x, spotBase.y, 120 * zoom);
    coneGrad.addColorStop(0, 'rgba(255, 255, 255, 0.12)');
    coneGrad.addColorStop(0.6, 'rgba(56, 189, 248, 0.05)');
    coneGrad.addColorStop(1, 'rgba(0, 0, 0, 0)');
    ctx.fillStyle = coneGrad;
    ctx.beginPath();
    ctx.arc(spotBase.x, spotBase.y, 110 * zoom, 0, Math.PI * 2);
    ctx.fill();

    // Floating Dust Photon Particles
    ctx.fillStyle = 'rgba(255, 255, 255, 0.75)';
    for (const p of spatial.particles) {
      const pt = project(p.x, p.y, p.z);
      if (pt.visible) {
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, p.size * pt.scale, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    ctx.restore();
  } else if (target.lighting === 'low-light') {
    // Night-Vision / IR Reticle Overlay
    ctx.save();
    ctx.strokeStyle = 'rgba(16, 185, 129, 0.35)';
    ctx.lineWidth = 1;
    ctx.setLineDash([6, 6]);
    ctx.strokeRect(16, 16, w - 32, h - 32);

    // Corner brackets
    ctx.setLineDash([]);
    ctx.strokeStyle = '#10b981';
    ctx.lineWidth = 2;
    const cSize = 14;
    // Top-Left
    ctx.beginPath(); ctx.moveTo(16, 16 + cSize); ctx.lineTo(16, 16); ctx.lineTo(16 + cSize, 16); ctx.stroke();
    // Top-Right
    ctx.beginPath(); ctx.moveTo(w - 16 - cSize, 16); ctx.lineTo(w - 16, 16); ctx.lineTo(w - 16, 16 + cSize); ctx.stroke();
    // Bottom-Left
    ctx.beginPath(); ctx.moveTo(16, h - 16 - cSize); ctx.lineTo(16, h - 16); ctx.lineTo(16 + cSize, h - 16); ctx.stroke();
    // Bottom-Right
    ctx.beginPath(); ctx.moveTo(w - 16 - cSize, h - 16); ctx.lineTo(w - 16, h - 16); ctx.lineTo(w - 16, h - 16 - cSize); ctx.stroke();

    ctx.restore();
  }

  ctx.restore();
}

/* ==========================================================================
   Capture & Keyframe Processing
   ========================================================================== */
async function beginCapture() {
  showView('captureView');
  capturedFrameCanvases = [];
  frameMetrics = [];
  $('#preview')?.classList.remove('uploaded-source');
  updateKeyframeLiveStrip();
  await enableOptionalImu();

  if (!deviceCapabilities?.camera) {
    toast('Camera capture is unavailable. You can upload a video instead.');
    $('#recState').textContent = 'CAMERA UNAVAILABLE · UPLOAD VIDEO';
    return;
  }
  try {
    const constraints = {
      video: { facingMode: currentFacingMode, width: { ideal: 1280 } },
      audio: true
    };
    try {
      stream = await navigator.mediaDevices.getUserMedia(constraints);
      audioEnabled = stream.getAudioTracks().length > 0;
    } catch (audioError) {
      stream = await navigator.mediaDevices.getUserMedia({ video: constraints.video, audio: false });
      audioEnabled = false;
    }
    $('#preview').srcObject = stream;
    const cameraFps = stream.getVideoTracks()[0]?.getSettings?.().frameRate;
    if ($('#hudFps')) $('#hudFps').textContent = cameraFps ? String(Math.round(cameraFps)) : '—';
    $('#systemStatusText').textContent = visionModel
      ? `Camera + MobileNet Active · ${modelBackend.toUpperCase()}`
      : 'Camera Active · Model Fallback';
    $('#recState').textContent = 'STANDBY · READY';
  } catch (err) {
    console.warn('Camera fallback:', err);
    toast('Camera offline — simulator active');
    $('#recState').textContent = 'SIMULATOR READY';
  }
}

async function enableOptionalImu() {
  if (!deviceCapabilities?.imu) return;
  try {
    if (typeof DeviceMotionEvent.requestPermission === 'function') {
      const permission = await DeviceMotionEvent.requestPermission();
      if (permission !== 'granted') return;
    }
    window.removeEventListener('devicemotion', collectMotionSample);
    window.addEventListener('devicemotion', collectMotionSample, { passive: true });
  } catch (error) {
    console.info('Optional IMU unavailable:', error);
  }
}

function endCamera() {
  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }
  clearInterval(ticker);
  clearInterval(sampleTicker);
  recordingStartedAt = 0;
  seconds = 0;
  $('#timer').textContent = `00:00 / ${formatDuration(taskProfile.maxDurationSeconds)}`;
  $('#recordBtn')?.classList.remove('recording');
  $('#recordingIndicator')?.classList.remove('recording');
  $('#recordBtn')?.setAttribute('aria-label', 'Start recording');
  const shutterLabel = $('.shutter-label');
  if (shutterLabel) shutterLabel.textContent = `Tap to record · stop anytime · ${taskProfile.maxDurationSeconds}s maximum`;
}

function formatDuration(totalSeconds) {
  return formatClock(totalSeconds);
}

function sampleFrame() {
  const video = $('#preview');
  const canvas = $('#sample');
  if (!canvas) return;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  try {
    if (!video || video.videoWidth <= 0) return;
    const aspect = video.videoWidth / video.videoHeight;
    if (aspect >= 1) {
      canvas.width = 160;
      canvas.height = Math.max(64, Math.round(160 / aspect));
    } else {
      canvas.height = 160;
      canvas.width = Math.max(64, Math.round(160 * aspect));
    }
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const snap = document.createElement('canvas');
    snap.width = canvas.width;
    snap.height = canvas.height;
    snap.getContext('2d').drawImage(canvas, 0, 0);
    capturedFrameCanvases.push(snap);
    updateKeyframeLiveStrip();

    // Run the open-source MobileNet backbone on alternating keyframes. The
    // cloned canvas prevents subsequent camera frames from changing inference input.
    if (visionModel && !yoloReady && capturedFrameCanvases.length % 3 === 1) {
      embeddingPromises.push(extractMobileNetEmbedding(snap));
    }
    if (yoloReady && !yoloBusy && (capturedFrameCanvases.length === 1 || capturedFrameCanvases.length % 4 === 0)) {
      const detectorFrame = document.createElement('canvas');
      if (aspect >= 1) {
        detectorFrame.width = 320;
        detectorFrame.height = Math.max(128, Math.round(320 / aspect));
      } else {
        detectorFrame.height = 320;
        detectorFrame.width = Math.max(128, Math.round(320 * aspect));
      }
      detectorFrame.getContext('2d').drawImage(video, 0, 0, detectorFrame.width, detectorFrame.height);
      detectionCanvases.push(detectorFrame);
      const task = runYoloFrame(detectorFrame, capturedFrameCanvases.length - 1);
      detectionPromises.push(task);
    }

    // Compute pixel metrics
    const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
    const gray = [];
    let sum = 0, sqSum = 0, gx = 0, gy = 0;
    let centerSum = 0, centerSqSum = 0, centerCount = 0;
    let topSum = 0, bottomSum = 0;

    for (let y = 0; y < canvas.height; y++) {
      for (let x = 0; x < canvas.width; x++) {
        const idx = (y * canvas.width + x) * 4;
        const val = (imgData[idx] + imgData[idx + 1] + imgData[idx + 2]) / 3;
        gray.push(val);
        sum += val;
        sqSum += val * val;

        if (y < canvas.height / 3) topSum += val;
        if (y >= (canvas.height * 2) / 3) bottomSum += val;

        if (x > canvas.width * .25 && x < canvas.width * .75 && y > canvas.height * .25 && y < canvas.height * .75) {
          centerSum += val;
          centerSqSum += val * val;
          centerCount++;
        }

        if (x > 0) gx += Math.abs(val - gray[gray.length - 2]);
        if (y > 0) gy += Math.abs(val - gray[gray.length - canvas.width - 1]);
      }
    }

    const n = gray.length;
    const mean = sum / n;
    const variance = (sqSum / n) - (mean * mean);
    const centerMean = centerSum / centerCount;
    const centerVariance = (centerSqSum / centerCount) - (centerMean * centerMean);
    const prevPixels = frameMetrics.at(-1)?.pixels;
    const motion = prevPixels ? gray.reduce((a, v, i) => a + Math.abs(v - prevPixels[i]), 0) / n : 0;

    frameMetrics.push({
      mean,
      variance,
      centerVariance,
      gx: gx / n,
      gy: gy / n,
      verticalBias: (bottomSum - topSum) / (n / 3),
      motion,
      pixels: gray
    });

    $('#hudLuminance').textContent = Math.round(mean);
    $('#hudMotion').textContent = motion.toFixed(1);

  } catch (e) {
    console.warn('Sample error:', e);
  }
}

async function runYoloFrame(canvas, frameIndex) {
  yoloBusy = true;
  try {
    const started = performance.now();
    const detections = await detectObjects(canvas);
    yoloLatencyMs = performance.now() - started;
    detectionFrames.push({ frameIndex, at: frameIndex * CAPTURE_CONFIG.keyframeIntervalMs / 1000, detections });
    const overlay = $('#detectionOverlay');
    if (overlay) drawDetections(overlay, detections, canvas.width, canvas.height);
    const status = $('#yoloLiveStatus');
    if (status) status.textContent = `YOLO · ${detections.length} · ${Math.round(yoloLatencyMs)}ms`;
    renderModelHealth();
  } catch (error) {
    console.warn('YOLO frame skipped:', error);
  } finally {
    yoloBusy = false;
  }
}

async function extractMobileNetEmbedding(canvas) {
  let tensor;
  try {
    tensor = visionModel.infer(canvas, true);
    embeddingSamples.push(Array.from(await tensor.data()));
    if (objectLabels.length < 3) {
      const prediction = await visionModel.classify(canvas, 1);
      if (prediction[0]) objectLabels.push(prediction[0].className);
    }
  } catch (error) {
    console.warn('MobileNet keyframe inference skipped:', error);
  } finally {
    tensor?.dispose();
  }
}

function updateKeyframeLiveStrip() {
  const container = $('#keyframeLiveStrip');
  const label = $('#keyframeCountLabel');
  if (!container) return;

  if (!capturedFrameCanvases.length) {
    container.innerHTML = '<div class="filmstrip-placeholder">Keyframes will appear while recording.</div>';
    if (label) label.textContent = '0 Frames';
    if ($('#hudKeyframes')) $('#hudKeyframes').textContent = '0';
    return;
  }

  if (label) label.textContent = `${capturedFrameCanvases.length} Frames Buffer`;
  if ($('#hudKeyframes')) $('#hudKeyframes').textContent = String(capturedFrameCanvases.length);

  container.innerHTML = '';
  capturedFrameCanvases.forEach(c => {
    const slot = document.createElement('div');
    slot.className = 'filmstrip-frame-slot';
    const display = document.createElement('canvas');
    display.width = c.width;
    display.height = c.height;
    display.getContext('2d').drawImage(c, 0, 0);
    slot.appendChild(display);
    container.appendChild(slot);
  });
  container.scrollLeft = container.scrollWidth;
}

function toggleRecording() {
  if (recorder && recorder.state === 'recording') {
    recorder.stop();
    return;
  }

  if (!stream) {
    toast('Camera is unavailable. Use Upload to analyze an existing video.');
    return;
  }

  chunks = [];
  frameMetrics = [];
  capturedFrameCanvases = [];
  embeddingSamples = [];
  embeddingPromises = [];
  objectLabels = [];
  detectionFrames = [];
  detectionPromises = [];
  detectionCanvases = [];
  imuSamples = [];
  recordedDurationSeconds = 0;
  recordingStartedAt = performance.now();
  sampleFrame();
  sampleTicker = setInterval(sampleFrame, CAPTURE_CONFIG.keyframeIntervalMs);

  if (stream) {
    try {
      const mimeType = MediaRecorder.isTypeSupported('video/webm;codecs=vp9') ? 'video/webm;codecs=vp9' : 'video/webm';
      recorder = new MediaRecorder(stream, { mimeType });
      recorder.ondataavailable = e => { if (e.data.size) chunks.push(e.data); };
      recorder.onstop = finishRecording;
      recorder.start();
    } catch (err) {
      console.warn('Recorder start fallback:', err);
    }
  }

  $('#recordBtn')?.classList.add('recording');
  $('#recordingIndicator')?.classList.add('recording');
  $('#recState').textContent = 'RECORDING LIVE';
  $('#recordBtn')?.setAttribute('aria-label', 'Stop recording');
  const shutterLabel = $('.shutter-label');
  if (shutterLabel) shutterLabel.textContent = 'STOP';

  seconds = 0;
  ticker = setInterval(() => {
    seconds++;
    recordedDurationSeconds = (performance.now() - recordingStartedAt) / 1000;
    $('#timer').textContent = `${formatDuration(recordedDurationSeconds)} / ${formatDuration(taskProfile.maxDurationSeconds)}`;
    if (recordedDurationSeconds >= taskProfile.maxDurationSeconds) {
      if (recorder && recorder.state === 'recording') recorder.stop();
      else finishRecording();
    }
  }, 1000);
}

async function finishRecording() {
  recordedDurationSeconds = Math.min(taskProfile.maxDurationSeconds, Math.max(.1, (performance.now() - recordingStartedAt) / 1000));
  clearInterval(sampleTicker);
  clearInterval(ticker);
  sampleFrame();

  recordedBlob = new Blob(chunks, { type: 'video/webm' });
  await Promise.allSettled(detectionPromises);
  if (visionModel && yoloReady && detectionCanvases.length) {
    const selected = [detectionCanvases[0], detectionCanvases[Math.floor(detectionCanvases.length / 2)], detectionCanvases.at(-1)].filter((item, index, all) => item && all.indexOf(item) === index);
    for (const canvas of selected) await extractMobileNetEmbedding(canvas);
    await refineAmbiguousDetections();
  }
  await Promise.allSettled(embeddingPromises);
  endCamera();
  showView('reviewView');
  analyzeFrames();
}

async function analyzeUploadedVideo(file) {
  recordedBlob = file;
  toast(`Analyzing ${file.name}`);
  frameMetrics = [];
  capturedFrameCanvases = [];
  embeddingSamples = [];
  embeddingPromises = [];
  detectionFrames = [];
  detectionPromises = [];
  detectionCanvases = [];
  objectLabels = [];
  const video = $('#preview');
  const url = URL.createObjectURL(file);
  video.srcObject = null;
  video.src = url;
  video.classList.add('uploaded-source');
  await new Promise((resolve, reject) => {
    video.onloadedmetadata = resolve;
    video.onerror = reject;
  });
  const duration = video.duration || 1;
  recordedDurationSeconds = video.duration || 0;
  const samples = Math.min(CAPTURE_CONFIG.uploadedVideoSampleLimit, Math.max(3, Math.ceil(duration)));
  for (let i = 0; i < samples; i++) {
    video.currentTime = samples === 1 ? 0 : (duration * i / (samples - 1));
    await new Promise(resolve => video.onseeked = resolve);
    sampleFrame();
  }
  await Promise.allSettled(detectionPromises);
  if (visionModel && yoloReady && detectionCanvases.length) {
    const selected = [detectionCanvases[0], detectionCanvases[Math.floor(detectionCanvases.length / 2)], detectionCanvases.at(-1)].filter((item, index, all) => item && all.indexOf(item) === index);
    for (const canvas of selected) await extractMobileNetEmbedding(canvas);
    await refineAmbiguousDetections();
  }
  await Promise.allSettled(embeddingPromises);
  URL.revokeObjectURL(url);
  endCamera();
  showView('reviewView');
  analyzeFrames();
}

const IMAGENET_TO_COCO = [
  { pattern: /notebook|laptop|portable computer/i, label: 'laptop' },
  { pattern: /television|monitor|screen|desktop computer/i, label: 'tv' },
  { pattern: /cellular telephone|mobile phone|smartphone/i, label: 'cell phone' },
  { pattern: /book jacket|comic book|bookshop|library/i, label: 'book' },
  { pattern: /coffee mug|cup/i, label: 'cup' },
  { pattern: /water bottle|beer bottle|pop bottle/i, label: 'bottle' },
  { pattern: /computer keyboard|typewriter keyboard/i, label: 'keyboard' },
  { pattern: /computer mouse|mouse/i, label: 'mouse' }
];

function mappedImageNetLabel(className) {
  return IMAGENET_TO_COCO.find(item => item.pattern.test(className))?.label || null;
}

async function refineAmbiguousDetections() {
  if (!visionModel || !detectionFrames.length) return;
  const ambiguous = new Set(['cell phone', 'book', 'laptop', 'tv', 'cup', 'bottle', 'keyboard', 'mouse']);
  const frame = detectionFrames.at(-1);
  const source = detectionCanvases.at(-1);
  if (!source) return;
  const candidates = frame.detections.filter(item => ambiguous.has(item.label)).sort((a, b) => b.confidence - a.confidence).slice(0, 4);
  for (const detection of candidates) {
    const crop = document.createElement('canvas');
    crop.width = Math.max(32, Math.round(detection.width));
    crop.height = Math.max(32, Math.round(detection.height));
    crop.getContext('2d').drawImage(source, detection.x, detection.y, detection.width, detection.height, 0, 0, crop.width, crop.height);
    try {
      const predictions = await visionModel.classify(crop, 5);
      const match = predictions.map(item => ({ ...item, mapped: mappedImageNetLabel(item.className) })).find(item => item.mapped);
      if (match && match.probability >= 0.12) {
        detection.yoloLabel = detection.label;
        detection.label = match.mapped;
        detection.refinedBy = 'MobileNet crop verification';
        detection.refinementConfidence = match.probability;
      }
    } catch (error) {
      console.warn('Crop verification skipped:', error);
    }
  }
  applyTemporalLabelVoting();
}

function applyTemporalLabelVoting() {
  const tracks = [];
  detectionFrames.forEach(frame => frame.detections.forEach(detection => {
    const cx = detection.x + detection.width / 2, cy = detection.y + detection.height / 2;
    let track = tracks.find(item => Math.hypot(item.cx - cx, item.cy - cy) < Math.max(detection.width, detection.height) * .65);
    if (!track) { track = { cx, cy, detections: [] }; tracks.push(track); }
    track.cx = (track.cx + cx) / 2; track.cy = (track.cy + cy) / 2; track.detections.push(detection);
  }));
  tracks.forEach(track => {
    const votes = new Map();
    track.detections.forEach(item => votes.set(item.label, (votes.get(item.label) || 0) + item.confidence * (item.refinedBy ? 5 : 1)));
    const winner = [...votes.entries()].sort((a, b) => b[1] - a[1])[0]?.[0];
    if (winner) track.detections.forEach(item => { item.label = winner; item.temporalConsensus = track.detections.length; });
  });
}

/* ==========================================================================
   Keyframe Review & Verification
   ========================================================================== */
function analyzeFrames() {
  const frames = frameMetrics.length ? frameMetrics : [
    { mean: 110, variance: 900, centerVariance: 700, gx: 10, gy: 12, verticalBias: 0, motion: 0 }
  ];

  const avg = k => frames.reduce((s, f) => s + (f[k] || 0), 0) / frames.length;
  const meanLum = avg('mean');
  const textureRatio = avg('centerVariance') / Math.max(1, avg('variance'));
  const motions = frames.map(f => f.motion || 0);
  const motionMean = motions.reduce((a, b) => a + b, 0) / Math.max(1, motions.length);
  const motionPeaks = motions.slice(1, -1).filter((m, i) => m > motions[i] * 1.35 && m > motions[i + 2] * 1.35 && m > 6).length;

  const allDetections = detectionFrames.flatMap(frame => frame.detections);
  const largestArea = allDetections.reduce((max, d) => Math.max(max, d.areaRatio || 0), 0);
  const detectedOcclusion = largestArea > 0.62 ? 'heavy' : (largestArea > 0.32 ? 'partial' : null);
  proposedTags = {
    occlusion: detectedOcclusion || (textureRatio < 0.32 ? 'heavy' : (textureRatio < 0.68 ? 'partial' : 'none')),
    lighting: meanLum < 70 ? 'low-light' : (meanLum > 175 ? 'bright' : 'normal'),
    orientation: avg('gx') > avg('gy') * 1.18 ? 'rotated' : (avg('gy') > avg('gx') * 1.6 ? 'inverted' : 'upright'),
    environment: avg('verticalBias') > 12 ? 'floor' : 'bench',
    result: motions.at(-1) > Math.max(8, motionMean * 1.35) ? 'failure' : 'success',
    recovery: motionPeaks >= 2 ? 'yes' : 'no'
  };

  const strongestDetection = allDetections.sort((a, b) => b.confidence - a.confidence)[0];
  proposedEvidence = {
    object: {
      measurement: strongestDetection ? `${strongestDetection.label} detected${strongestDetection.refinedBy ? ' and crop-verified' : ''}` : 'No known COCO class detected',
      inference: strongestDetection?.label || 'unknown object',
      confidence: strongestDetection?.refinedBy
        ? strongestDetection.confidence * .55 + strongestDetection.refinementConfidence * .45
        : strongestDetection?.confidence || 0
    },
    occlusion: {
      measurement: allDetections.length ? `Largest detection occupies ${Math.round(largestArea * 100)}% of frame` : `Center texture ratio ${textureRatio.toFixed(2)}`,
      inference: proposedTags.occlusion,
      confidence: Math.min(.96, allDetections.length ? .58 + largestArea * .55 : .52 + Math.abs(textureRatio - .68) * .45)
    },
    lighting: {
      measurement: `Mean luminance ${Math.round(meanLum)} / 255`,
      inference: proposedTags.lighting,
      confidence: Math.min(.97, .62 + Math.abs(meanLum - 122) / 255)
    },
    orientation: {
      measurement: `Edge ratio ${(avg('gx') / Math.max(.1, avg('gy'))).toFixed(2)}`,
      inference: proposedTags.orientation,
      confidence: Math.min(.9, .55 + Math.abs(avg('gx') - avg('gy')) / Math.max(1, avg('gx') + avg('gy')))
    },
    outcome: {
      measurement: `${motionPeaks} temporal motion inflection${motionPeaks === 1 ? '' : 's'}`,
      inference: `likely ${proposedTags.result}`,
      confidence: Math.min(.88, .52 + Math.abs((motions.at(-1) || 0) - motionMean) / Math.max(10, motionMean * 3))
    }
  };
  proposedTask = taskProfile.task;
  proposedObject = proposedEvidence.object.inference;
  calibrationCorrections = [];
  outcomeConfirmed = false;
  if (playbackUrl) URL.revokeObjectURL(playbackUrl);
  playbackUrl = recordedBlob?.size ? URL.createObjectURL(recordedBlob) : null;
  const playback = $('#reviewPlayback');
  if (playback) { playback.src = playbackUrl || ''; playback.hidden = !playbackUrl; }
  $$('[data-confirm-result]').forEach(button => button.classList.remove('selected'));
  if ($('#confirmRecovery')) $('#confirmRecovery').checked = proposedTags.recovery === 'yes';

  $('#featLuminance').textContent = `${Math.round(meanLum)} / 255`;
  $('#featTexture').textContent = textureRatio.toFixed(2);
  $('#featGradient').textContent = `${(avg('gx') / Math.max(0.1, avg('gy'))).toFixed(2)}x`;
  $('#featMotion').textContent = `${motionPeaks} peaks`;

  renderKeyframeGallery();
  renderTagEditor();
  const latestDetections = detectionFrames.at(-1)?.detections || [];
  $('#detectedObjectCount').textContent = `${latestDetections.length} object${latestDetections.length === 1 ? '' : 's'}`;
  renderDetectionEvidence({ meanLum, textureRatio, motionPeaks });

  $('#processBar').style.width = '100%';
  setTimeout(() => {
    $('#processState').textContent = visionModel
      ? `${embeddingSamples.length} MOBILENET EMBEDDINGS · ${modelBackend.toUpperCase()}`
      : `${frames.length} KEYFRAMES · HEURISTIC FALLBACK`;
    $('#saveClip').disabled = !outcomeConfirmed;
  }, 600);
}

function renderDetectionEvidence(metrics) {
  const objectList = $('#detectionEvidenceList');
  const tagList = $('#tagEvidenceList');
  if (!objectList || !tagList) return;
  const bestByLabel = new Map();
  detectionFrames.flatMap(frame => frame.detections).forEach(detection => {
    const current = bestByLabel.get(detection.label);
    if (!current || detection.confidence > current.confidence) bestByLabel.set(detection.label, detection);
  });
  const objects = [...bestByLabel.values()].sort((a, b) => b.confidence - a.confidence).slice(0, 6);
  objectList.innerHTML = objects.length
    ? objects.map(d => `<span class="detected-object-chip"><b>${titleCase(d.label)}</b><small>${Math.round(d.confidence * 100)}%</small></span>`).join('')
    : '<span class="evidence-empty">No known COCO object detected; visual features still produced reviewable tags.</span>';
  const evidence = Object.entries(proposedEvidence);
  tagList.innerHTML = evidence.map(([name, item]) => `<div class="tag-evidence-row confidence-${confidenceLabel(item.confidence)}"><span>${titleCase(name)}</span><b>${titleCase(item.inference)}</b><small>${item.measurement}</small><em title="${confidenceLabel(item.confidence)} confidence">${Math.round(item.confidence * 100)}%</em></div>`).join('');
}

function renderKeyframeGallery() {
  const gallery = $('#sampledKeyframeGallery');
  if (!gallery) return;

  if (!capturedFrameCanvases.length) {
    gallery.innerHTML = '<p class="b-desc">No video keyframes sampled.</p>';
    return;
  }

  gallery.innerHTML = '';
  capturedFrameCanvases.forEach((canvas, idx) => {
    const item = document.createElement('div');
    item.className = 'gallery-item';

    const box = document.createElement('div');
    box.className = 'gallery-canvas-box';
    const c = document.createElement('canvas');
    c.width = canvas.width;
    c.height = canvas.height;
    c.getContext('2d').drawImage(canvas, 0, 0);
    box.appendChild(c);

    const lbl = document.createElement('span');
    lbl.className = 'gallery-time-lbl';
    lbl.textContent = `T+${(idx * 0.5).toFixed(1)}s`;

    item.appendChild(box);
    item.appendChild(lbl);
    gallery.appendChild(item);
  });
}

function renderTagEditor() {
  const editor = $('#tagEditor');
  if (!editor) return;

  editor.innerHTML = `
    <div class="tag-field-group text-attribute-field">
      <label class="tag-field-label" for="taskAttribute">TASK</label>
      <input id="taskAttribute" value="${escapeHtml(proposedTask)}" placeholder="e.g. insertion, inspection, navigation">
    </div>
    <div class="tag-field-group text-attribute-field">
      <label class="tag-field-label" for="objectAttribute">OBJECT</label>
      <input id="objectAttribute" value="${escapeHtml(proposedObject)}" placeholder="Detected or operator-provided object">
    </div>` + Object.entries(SCHEMA).map(([key, options]) => `
    <div class="tag-field-group">
      <span class="tag-field-label">${key.toUpperCase()}</span>
      <div class="tag-options-row">
        ${options.map(val => `
          <button class="tag-toggle-pill ${proposedTags[key] === val ? 'selected' : ''}" data-key="${key}" data-value="${val}">
            ${titleCase(val)}
          </button>
        `).join('')}
      </div>
    </div>
  `).join('');

  $$('.tag-toggle-pill').forEach(btn => {
    btn.onclick = () => {
      proposedTags[btn.dataset.key] = btn.dataset.value;
      const original = proposedEvidence[btn.dataset.key]?.inference;
      if (original && original !== btn.dataset.value) calibrationCorrections.push(calibrationCorrection({ attribute:btn.dataset.key, from:original, to:btn.dataset.value, evidence:proposedEvidence[btn.dataset.key] }));
      renderTagEditor();
    };
  });
  $('#taskAttribute')?.addEventListener('input', event => { proposedTask = event.target.value.trim() || taskProfile.task; });
  $('#objectAttribute')?.addEventListener('input', event => { proposedObject = event.target.value.trim() || 'unknown object'; });
}

async function commitClip() {
  if (!outcomeConfirmed) { toast('Confirm the actual outcome before saving'); return; }
  const beforeReadiness = calculateReadiness().total;
  const beforeImpact = distributionSnapshot();
  const beforeRecommendation = recommendation ? { ...recommendation } : null;
  const notes = $('#operatorNotes')?.value || '';

  const embedding = embeddingSamples.length
    ? embeddingSamples[0].map((_, i) => embeddingSamples.reduce((sum, item) => sum + (item[i] || 0), 0) / embeddingSamples.length)
    : undefined;

  const newRecord = {
    id: `live-${Date.now().toString(36).toUpperCase()}`,
    created: Date.now(),
    source: 'live',
    durationSeconds: Number(recordedDurationSeconds.toFixed(2)),
    sourceDevice: describeSourceDevice(),
    task: proposedTask,
    object: proposedObject,
    ...proposedTags,
    notes,
    evidence: Object.entries(proposedEvidence).map(([attribute, item]) => ({ attribute, ...item })),
    calibrationCorrections: [...calibrationCorrections],
    optionalMetadata: {
      audio: { available: audioEnabled },
      imu: { available: deviceCapabilities?.imu || false, samples: imuSamples },
      gps: { available: deviceCapabilities?.gps || false, collected: false },
      robotTelemetry: null,
      external: {}
    },
    embedding,
    model: {
      name: visionModel ? 'MobileNetV1 alpha-0.25' : 'heuristic-fallback',
      backend: modelBackend,
      keyframes: embeddingSamples.length,
      topLabel: objectLabels[0] || null
    },
    detections: detectionFrames.map(frame => ({
      at: frame.at,
      objects: frame.detections.map(({ label, confidence, x, y, width, height, areaRatio }) => ({ label, confidence, x, y, width, height, areaRatio }))
    })),
    video: recordedBlob
  };

  clips.push(newRecord);
  await saveRecord(newRecord);
  renderAll();

  const afterReadiness = calculateReadiness().total;
  $('#oldCoverage').textContent = `${beforeReadiness}%`;
  $('#newCoverage').textContent = `${afterReadiness}%`;
  renderImpactComparison(beforeImpact, distributionSnapshot());

  if (beforeRecommendation) {
    const contextKeys = ['occlusion', 'lighting', 'orientation', 'environment'];
    const updatedTarget = rankedExperiments.find(item => contextKeys.every(key => item[key] === beforeRecommendation[key]));
    const addedExamples = Math.max(0, (updatedTarget?.count ?? beforeRecommendation.count) - beforeRecommendation.count);
    const gapReduction = Math.max(0, Math.round((beforeRecommendation.gap - (updatedTarget?.gap ?? beforeRecommendation.gap)) * 100));
    if ($('#targetExampleDelta')) $('#targetExampleDelta').textContent = `${beforeRecommendation.count} → ${updatedTarget?.count ?? beforeRecommendation.count}`;
    if ($('#gapReductionDelta')) $('#gapReductionDelta').textContent = addedExamples ? `−${gapReduction} pts` : 'Rebalanced';
    const nextChanged = recommendation && contextKeys.some(key => recommendation[key] !== beforeRecommendation[key]);
    if ($('#nextRecommendationReason')) $('#nextRecommendationReason').innerHTML = `<b>${nextChanged ? 'NEXT PRIORITY CHANGED' : 'PRIORITY RECALCULATED'}</b><br>${escapeHtml(recommendationEvidence().text)}`;
  }

  const recDetail = proposedTags.recovery === 'yes' ? ' with recovery attempt' : '';
  $('#successCopy').textContent = `${formatDuration(recordedDurationSeconds)} demonstration saved: ${titleCase(proposedTags.occlusion)} occlusion · ${titleCase(proposedTags.lighting)} light · ${titleCase(proposedTags.result)}${recDetail}. Rankings were recalculated from the updated dataset.`;

  showView('successView');
}

function distributionSnapshot() {
  const total = Math.max(1, clips.length);
  const values = {};
  Object.entries(SCHEMA).forEach(([attribute, options]) => options.forEach(value => {
    values[`${attribute}:${value}`] = clips.filter(clip => clip[attribute] === value).length / total;
  }));
  return values;
}

function renderImpactComparison(before, after) {
  const container = $('#impactComparison');
  if (!container) return;
  const focus = [
    ['Occlusion', `occlusion:${proposedTags.occlusion}`],
    ['Lighting', `lighting:${proposedTags.lighting}`],
    ['Orientation', `orientation:${proposedTags.orientation}`],
    ['Recovery', `recovery:${proposedTags.recovery}`]
  ];
  container.innerHTML = focus.map(([label, key]) => {
    const from = Math.round((before[key] || 0) * 100), to = Math.round((after[key] || 0) * 100);
    return `<div class="impact-row"><span>${label}</span><div class="impact-track"><i style="width:${from}%"></i><b style="width:${to}%"></b></div><em>${from}% → ${to}%</em></div>`;
  }).join('');
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
        const row = { ...raw, id: raw.id || `import-${crypto.randomUUID?.() || Date.now()}`, created: raw.created || Date.now(), source:'import' };
        clips.push(row); await saveRecord(row); imported++;
      }
    } catch (error) { toast(`${file.name}: invalid TRACE JSON`); }
  }
  if (imported) { renderAll(); toast(`Imported ${imported} demonstrations`); }
  if (videoFiles.length) {
    if (videoFiles.length > 1) toast('Opening the first video; confirm each clip separately');
    await analyzeUploadedVideo(videoFiles[0]);
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
  // Navigation tabs
  $$('[data-view]').forEach(btn => {
    btn.onclick = () => showView(btn.dataset.view);
  });

  // Capture buttons
  $('#quickRecordBtn')?.addEventListener('click', openCaptureBriefing);
  $('#quickRecordMobile')?.addEventListener('click', openCaptureBriefing);
  $('#recordThis')?.addEventListener('click', openCaptureBriefing);
  $('#logFailureBtn')?.addEventListener('click', () => openOnboarding(2, 'failure'));
  $('#onboardingBack')?.addEventListener('click', () => setOnboardingStep(onboardingStep - 1));
  $('#onboardingNext')?.addEventListener('click', async () => {
    if (onboardingStep < 3) setOnboardingStep(onboardingStep + 1);
    else await finishOnboarding();
  });
  $('#onboardingSkip')?.addEventListener('click', () => finishOnboarding({ skip: true }));
  $('#closeBriefing')?.addEventListener('click', () => { $('#captureBriefing').hidden = true; });
  $('#startGuidedCapture')?.addEventListener('click', () => { $('#captureBriefing').hidden = true; beginCapture(); });
  $('#quickSimulateBtn')?.addEventListener('click', injectSyntheticDemo);
  $('#reRecordBtn')?.addEventListener('click', () => { if (playbackUrl) URL.revokeObjectURL(playbackUrl); playbackUrl=null; beginCapture(); });
  $$('[data-confirm-result]').forEach(button => button.addEventListener('click', () => {
    proposedTags.result = button.dataset.confirmResult;
    outcomeConfirmed = true;
    $$('[data-confirm-result]').forEach(item => item.classList.toggle('selected', item === button));
    $('#saveClip').disabled = false;
  }));
  $('#confirmRecovery')?.addEventListener('change', event => { proposedTags.recovery = event.target.checked ? 'yes' : 'no'; });
  $('#datasetImportInput')?.addEventListener('change', event => importDatasetFiles(event.target.files));

  $('#toggleDiagnostics')?.addEventListener('click', () => {
    const wrapper = $('.review-wrapper');
    const button = $('#toggleDiagnostics');
    const expanded = !wrapper?.classList.contains('show-diagnostics');
    wrapper?.classList.toggle('show-diagnostics', expanded);
    button?.setAttribute('aria-expanded', String(expanded));
    if (button) button.textContent = expanded ? 'Hide technical analysis' : 'Show technical analysis';
  });

  // Viewfinder actions
  $('#closeCapture')?.addEventListener('click', () => {
    endCamera();
    showView('homeView');
  });
  $('#recordBtn')?.addEventListener('click', toggleRecording);

  $('#switchCameraBtn')?.addEventListener('click', () => {
    currentFacingMode = currentFacingMode === 'environment' ? 'user' : 'environment';
    endCamera();
    beginCapture();
  });

  $('#videoFileInput')?.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (file) {
      await analyzeUploadedVideo(file);
    }
  });

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

  // Review screen actions
  $('#saveClip')?.addEventListener('click', commitClip);
  $('#discardClip')?.addEventListener('click', () => {
    recordedBlob = null;
    capturedFrameCanvases = [];
    showView('homeView');
    toast('Clip discarded');
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
  loadVisionModel();
  loadYoloModel();
  initEventListeners();
  if (OPTIONAL_SIMULATOR_ENABLED) initSpatialVisualizer();
  loadDataset();
});
