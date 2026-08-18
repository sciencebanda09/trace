/**
 * TRACE · Physical-AI Demonstration Acquisition Engine
 * Dynamic deterministic priority scoring, interactive 2.5D visualizer, and telemetry pipeline.
 */

// Dynamic Schema Definition
const SCHEMA = {
  occlusion: ['none', 'partial', 'heavy'],
  lighting: ['bright', 'normal', 'low-light'],
  orientation: ['upright', 'rotated', 'inverted'],
  environment: ['bench', 'floor'],
  result: ['success', 'failure'],
  recovery: ['no', 'yes']
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
  occlusion: x[0],
  lighting: x[1],
  orientation: x[2],
  environment: x[3],
  result: x[4],
  recovery: x[5],
  notes: 'Automated baseline robot demonstration batch'
}));

const $ = s => document.querySelector(s);
const $$ = s => [...document.querySelectorAll(s)];

// Global State
let clips = [];
let rankedExperiments = [];
let recommendation = null;
let activeRankIndex = 0;

// Dynamic Engine Weights (Configurable by user in Real-Time)
let engineWeights = {
  gap: 0.30,
  failure: 0.50,
  novelty: 0.20,
  costSensitivity: 1.00
};

// Camera & Capture State
let stream = null;
let recorder = null;
let chunks = [];
let seconds = 0;
let ticker = null;
let sampleTicker = null;
let recordedBlob = null;
let proposedTags = {};
let frameMetrics = [];
let capturedFrameCanvases = [];
let currentFacingMode = 'environment';

// UI Navigation & Filters
let activeMatrixFilter = 'all';
let activeVaultFilter = 'all';
let vaultSearchQuery = '';

// Interactive 2.5D Spatial Visualizer State
let spatialAngle = 0.45;
let isDraggingSpatial = false;
let startDragX = 0;
let spatialAnimFrame = null;

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
    const db = await openDB();
    clips = await new Promise((res, rej) => {
      const q = db.transaction('clips', 'readonly').objectStore('clips').getAll();
      q.onsuccess = () => res(q.result);
      q.onerror = () => rej(q.error);
    });

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
}

async function saveRecord(row) {
  try {
    const db = await openDB();
    await new Promise((res, rej) => {
      const tx = db.transaction('clips', 'readwrite');
      const q = tx.objectStore('clips').put(row);
      q.onsuccess = res;
      q.onerror = () => rej(q.error);
    });
  } catch (err) {
    console.warn('Record save error:', err);
  }
}

async function deleteRecord(id) {
  try {
    const db = await openDB();
    await new Promise((res, rej) => {
      const tx = db.transaction('clips', 'readwrite');
      const q = tx.objectStore('clips').delete(id);
      q.onsuccess = res;
      q.onerror = () => rej(q.error);
    });
    clips = clips.filter(c => c.id !== id);
    renderAll();
    toast(`Demonstration ${id} deleted`);
  } catch (err) {
    toast('Failed to delete demonstration');
  }
}

async function resetDataset() {
  try {
    const db = await openDB();
    await new Promise((res, rej) => {
      const tx = db.transaction('clips', 'readwrite');
      const q = tx.objectStore('clips').clear();
      q.onsuccess = res;
      q.onerror = () => rej(q.error);
    });
    clips = [...seed];
    await Promise.all(seed.map(saveRecord));
  } catch {
    clips = [...seed];
  }
  renderAll();
  toast('Seed dataset restored');
}

/* ==========================================================================
   Combinatorics & Dynamic Priority Math
   ========================================================================== */
function combinations() {
  const contextKeys = ['occlusion', 'lighting', 'orientation', 'environment'];
  let results = [{}];

  for (const key of contextKeys) {
    const next = [];
    const values = SCHEMA[key] || [];
    for (const combo of results) {
      for (const val of values) {
        next.push({ ...combo, [key]: val });
      }
    }
    results = next;
  }
  return results;
}

function countExact(combo) {
  return clips.filter(c => 
    Object.keys(combo).every(k => c[k] === combo[k])
  ).length;
}

function countExactFailures(combo) {
  return clips.filter(c => 
    c.result === 'failure' &&
    Object.keys(combo).every(k => c[k] === combo[k])
  ).length;
}

function getContextDifficultyCost(x) {
  // Dynamically compute baseline setup difficulty from attribute levels
  const costs = {
    occlusion: { none: 1.0, partial: 1.15, heavy: 1.45 },
    lighting: { normal: 1.0, bright: 1.05, 'low-light': 1.25 },
    orientation: { upright: 1.0, rotated: 1.15, inverted: 1.40 },
    environment: { bench: 1.0, floor: 1.10 }
  };

  let totalCost = 0;
  let count = 0;
  for (const [dim, val] of Object.entries(x)) {
    if (costs[dim] && costs[dim][val] !== undefined) {
      totalCost += costs[dim][val];
      count++;
    }
  }
  return count > 0 ? (totalCost / count) : 1.0;
}

function scoreAll() {
  const failures = clips.filter(c => c.result === 'failure');
  const allCombos = combinations();
  const maxCount = Math.max(1, ...allCombos.map(x => countExact(x)));

  return allCombos.map(x => {
    const count = countExact(x);
    const failCount = countExactFailures(x);
    const gap = 1 - (count / maxCount);

    // Overlap with logged failures across 4 context dimensions
    const contextKeys = ['occlusion', 'lighting', 'orientation', 'environment'];
    const overlaps = failures.map(f => 
      contextKeys.filter(k => f[k] === x[k]).length / contextKeys.length
    );
    const failureScore = overlaps.length ? overlaps.reduce((a, b) => a + b, 0) / overlaps.length : 0;

    // Novelty estimation
    const noveltyScore = Math.min(1, 0.35 + gap * 0.65);

    // Acquisition setup cost with user sensitivity exponent
    const baseCost = getContextDifficultyCost(x);
    const effectiveCost = Math.pow(baseCost, engineWeights.costSensitivity);

    // Dynamic TRACE Ranking Equation
    const numerator = (engineWeights.gap * gap) + 
                      (engineWeights.failure * failureScore) + 
                      (engineWeights.novelty * noveltyScore);
    const priority = numerator / Math.max(0.1, effectiveCost);

    return {
      ...x,
      count,
      failCount,
      gap,
      failure: failureScore,
      novelty: noveltyScore,
      cost: baseCost,
      priority
    };
  }).sort((a, b) => b.priority - a.priority);
}

function calculateReadiness() {
  const contextKeys = ['occlusion', 'lighting', 'orientation', 'environment'];
  
  const catCov = contextKeys.reduce((sum, key) => 
    sum + (new Set(clips.map(c => c[key])).size / SCHEMA[key].length), 0) / contextKeys.length;
  
  const occupied = new Set(clips.map(c => contextKeys.map(k => c[k]).join('|'))).size;
  const comboCov = occupied / combinations().length;
  
  const outcomeCov = (
    (new Set(clips.map(c => c.result)).size / SCHEMA.result.length) +
    (new Set(clips.map(c => c.recovery)).size / SCHEMA.recovery.length)
  ) / 2;

  const total = Math.round(100 * (0.45 * catCov + 0.35 * comboCov + 0.2 * outcomeCov));
  return {
    total,
    catCov: Math.round(catCov * 100),
    comboCov: Math.round(comboCov * 100)
  };
}

function pct(n) {
  return `${Math.round(n * 100)}%`;
}

function titleCase(str) {
  if (!str) return '';
  return String(str).replace(/-/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

function generateInstruction(x) {
  if (!x) return 'Loading experiment targets...';
  const hasLowRecovery = clips.filter(c => c.recovery === 'yes').length / Math.max(1, clips.length) < 0.15;
  const occl = x.occlusion === 'none' ? 'unobstructed' : `${x.occlusion}-occlusion`;
  const recoveryNote = hasLowRecovery ? ' — if grasp fails, keep recording through the recovery attempt' : '';
  return `Record a ${occl} pick attempt of a ${x.orientation} object in ${x.lighting} lighting on the ${x.environment}${recoveryNote}.`;
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

  $('#rationale').textContent = `${matching} of ${failures.length} logged robot breakdowns overlap this exact operational profile. Dataset contains ${recommendation.count} demonstration${recommendation.count === 1 ? '' : 's'} (${pct(recommendation.gap)} sparsity).`;

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

  // Spatial HUD Label
  const occlLabel = $('#spatialOcclLabel');
  const orientLabel = $('#spatialOrientLabel');
  if (occlLabel) occlLabel.textContent = `Occlusion: ${titleCase(recommendation.occlusion)}`;
  if (orientLabel) orientLabel.textContent = `Orientation: ${titleCase(recommendation.orientation)}`;
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
   Interactive 2.5D Physical Scene Visualizer
   ========================================================================== */
function initSpatialVisualizer() {
  const canvas = $('#spatialCanvas');
  if (!canvas) return;

  const onStart = (e) => {
    isDraggingSpatial = true;
    startDragX = e.clientX || e.touches?.[0]?.clientX || 0;
  };

  const onMove = (e) => {
    if (!isDraggingSpatial) return;
    const clientX = e.clientX || e.touches?.[0]?.clientX || 0;
    const delta = clientX - startDragX;
    spatialAngle += delta * 0.015;
    startDragX = clientX;
    drawSpatialVisualizer();
  };

  const onEnd = () => {
    isDraggingSpatial = false;
  };

  canvas.addEventListener('mousedown', onStart);
  window.addEventListener('mousemove', onMove);
  window.addEventListener('mouseup', onEnd);

  canvas.addEventListener('touchstart', onStart, { passive: true });
  window.addEventListener('touchmove', onMove, { passive: true });
  window.addEventListener('touchend', onEnd);
}

function drawSpatialVisualizer() {
  const canvas = $('#spatialCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;

  ctx.clearRect(0, 0, w, h);

  const cx = w / 2;
  const cy = h / 2 + 30;
  const angle = spatialAngle;

  const target = recommendation || { occlusion: 'none', lighting: 'normal', orientation: 'upright', environment: 'bench' };

  // Atmosphere & Lighting
  let lightAlpha = 0.6;
  if (target.lighting === 'bright') lightAlpha = 0.95;
  if (target.lighting === 'low-light') lightAlpha = 0.25;

  // Draw Ground Surface (Isometric Ellipse Grid)
  ctx.save();
  ctx.translate(cx, cy);

  ctx.strokeStyle = `rgba(56, 189, 248, ${0.15 * lightAlpha})`;
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let r = 20; r <= 110; r += 25) {
    ctx.ellipse(0, 0, r, r * 0.45, 0, 0, Math.PI * 2);
  }
  ctx.stroke();

  // Axis grid lines
  for (let i = 0; i < 6; i++) {
    const a = angle + (i * Math.PI / 3);
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.lineTo(Math.cos(a) * 110, Math.sin(a) * 110 * 0.45);
    ctx.stroke();
  }

  // Draw Target Object
  let objRotation = 0;
  if (target.orientation === 'rotated') objRotation = Math.PI / 4;
  if (target.orientation === 'inverted') objRotation = Math.PI;

  const objX = Math.cos(angle) * 10;
  const objY = Math.sin(angle) * 5 - 25;

  ctx.save();
  ctx.translate(objX, objY);
  ctx.rotate(objRotation);

  // Object Base / Body
  const isFloor = target.environment === 'floor';
  const objColor = isFloor ? '#06b6d4' : '#10b981';

  ctx.fillStyle = objColor;
  ctx.shadowColor = `rgba(16, 185, 129, ${0.4 * lightAlpha})`;
  ctx.shadowBlur = 15;

  // Prism / Cylinder geometry
  ctx.beginPath();
  ctx.roundRect(-22, -22, 44, 44, 6);
  ctx.fill();

  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 1.5;
  ctx.stroke();

  // Object Center Calibration Point
  ctx.fillStyle = '#ffffff';
  ctx.beginPath();
  ctx.arc(0, 0, 4, 0, Math.PI * 2);
  ctx.fill();

  ctx.restore();

  // Draw Occlusion Obstacle Overlay
  if (target.occlusion !== 'none') {
    ctx.save();
    ctx.translate(15, -15);
    const occlAlpha = target.occlusion === 'heavy' ? 0.75 : 0.45;
    ctx.fillStyle = `rgba(244, 63, 94, ${occlAlpha})`;
    ctx.strokeStyle = '#fb7185';
    ctx.lineWidth = 1.5;

    ctx.beginPath();
    ctx.roundRect(-30, -35, 60, 40, 4);
    ctx.fill();
    ctx.stroke();

    // Cross-hatch mesh pattern
    ctx.strokeStyle = 'rgba(255,255,255,0.3)';
    ctx.beginPath();
    for (let x = -25; x <= 25; x += 10) {
      ctx.moveTo(x, -35);
      ctx.lineTo(x + 10, 5);
    }
    ctx.stroke();

    ctx.restore();
  }

  // Draw Robot Gripper End-Effector Above Object
  ctx.save();
  ctx.translate(0, -90);

  ctx.strokeStyle = '#38bdf8';
  ctx.lineWidth = 2.5;

  // Stem
  ctx.beginPath();
  ctx.moveTo(0, -30);
  ctx.lineTo(0, 0);
  ctx.stroke();

  // Crossbar
  ctx.beginPath();
  ctx.moveTo(-24, 0);
  ctx.lineTo(24, 0);
  ctx.stroke();

  // Gripper Fingers
  ctx.beginPath();
  ctx.moveTo(-24, 0);
  ctx.lineTo(-24, 28);
  ctx.moveTo(24, 0);
  ctx.lineTo(24, 28);
  ctx.stroke();

  // Gripper Laser Projection Line
  ctx.strokeStyle = `rgba(56, 189, 248, ${0.4 * lightAlpha})`;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(0, 0);
  ctx.lineTo(0, 70);
  ctx.stroke();

  ctx.restore();

  ctx.restore();
}

/* ==========================================================================
   Capture & Keyframe Processing
   ========================================================================== */
async function beginCapture() {
  showView('captureView');
  capturedFrameCanvases = [];
  frameMetrics = [];
  updateKeyframeLiveStrip();

  try {
    const constraints = {
      video: { facingMode: currentFacingMode, width: { ideal: 1280 } },
      audio: true
    };
    stream = await navigator.mediaDevices.getUserMedia(constraints);
    $('#preview').srcObject = stream;
    $('#systemStatusText').textContent = 'Camera Vision Active';
    $('#recState').textContent = 'STANDBY · READY';
  } catch (err) {
    console.warn('Camera fallback:', err);
    toast('Camera offline — simulator active');
    $('#recState').textContent = 'SIMULATOR READY';
  }
}

function endCamera() {
  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }
  clearInterval(ticker);
  clearInterval(sampleTicker);
  seconds = 0;
  $('#timer').textContent = '00:00';
  $('#recordBtn')?.classList.remove('recording');
  $('#recordingIndicator')?.classList.remove('recording');
}

function sampleFrame() {
  const video = $('#preview');
  const canvas = $('#sample');
  if (!canvas) return;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  canvas.width = 96;
  canvas.height = 72;

  try {
    if (video && video.videoWidth > 0) {
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    } else {
      // Draw test video frame
      ctx.fillStyle = '#0f172a';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = '#38bdf8';
      ctx.beginPath();
      ctx.arc(48 + Math.sin(Date.now() / 200) * 20, 36, 12, 0, Math.PI * 2);
      ctx.fill();
    }

    const snap = document.createElement('canvas');
    snap.width = canvas.width;
    snap.height = canvas.height;
    snap.getContext('2d').drawImage(canvas, 0, 0);
    capturedFrameCanvases.push(snap);
    updateKeyframeLiveStrip();

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

        if (x > 24 && x < 72 && y > 18 && y < 54) {
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

function updateKeyframeLiveStrip() {
  const container = $('#keyframeLiveStrip');
  const label = $('#keyframeCountLabel');
  if (!container) return;

  if (!capturedFrameCanvases.length) {
    container.innerHTML = '<div class="filmstrip-placeholder">Recording keyframes will buffer here every 500ms...</div>';
    if (label) label.textContent = '0 Frames';
    return;
  }

  if (label) label.textContent = `${capturedFrameCanvases.length} Frames Buffer`;

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

  chunks = [];
  frameMetrics = [];
  capturedFrameCanvases = [];
  sampleFrame();
  sampleTicker = setInterval(sampleFrame, 500);

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

  seconds = 0;
  ticker = setInterval(() => {
    seconds++;
    $('#timer').textContent = `00:${String(seconds).padStart(2, '0')}`;
    if (seconds >= 8) {
      if (recorder && recorder.state === 'recording') recorder.stop();
      else finishRecording();
    }
  }, 1000);
}

function finishRecording() {
  clearInterval(sampleTicker);
  clearInterval(ticker);
  sampleFrame();

  recordedBlob = new Blob(chunks, { type: 'video/webm' });
  analyzeFrames();
  endCamera();
  showView('reviewView');
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

  proposedTags = {
    occlusion: textureRatio < 0.32 ? 'heavy' : (textureRatio < 0.68 ? 'partial' : 'none'),
    lighting: meanLum < 70 ? 'low-light' : (meanLum > 175 ? 'bright' : 'normal'),
    orientation: avg('gx') > avg('gy') * 1.18 ? 'rotated' : (avg('gy') > avg('gx') * 1.6 ? 'inverted' : 'upright'),
    environment: avg('verticalBias') > 12 ? 'floor' : 'bench',
    result: motions.at(-1) > Math.max(8, motionMean * 1.35) ? 'failure' : 'success',
    recovery: motionPeaks >= 2 ? 'yes' : 'no'
  };

  $('#featLuminance').textContent = `${Math.round(meanLum)} / 255`;
  $('#featTexture').textContent = textureRatio.toFixed(2);
  $('#featGradient').textContent = `${(avg('gx') / Math.max(0.1, avg('gy'))).toFixed(2)}x`;
  $('#featMotion').textContent = `${motionPeaks} peaks`;

  renderKeyframeGallery();
  renderTagEditor();

  $('#processBar').style.width = '100%';
  setTimeout(() => {
    $('#processState').textContent = `${frames.length} KEYFRAMES ANALYZED · PROPOSALS READY`;
    $('#saveClip').disabled = false;
  }, 600);
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

  editor.innerHTML = Object.entries(SCHEMA).map(([key, options]) => `
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
      renderTagEditor();
    };
  });
}

async function commitClip() {
  const beforeReadiness = calculateReadiness().total;
  const notes = $('#operatorNotes')?.value || '';

  const newRecord = {
    id: `live-${Date.now().toString(36).toUpperCase()}`,
    created: Date.now(),
    source: 'live',
    ...proposedTags,
    notes,
    video: recordedBlob
  };

  clips.push(newRecord);
  await saveRecord(newRecord);
  renderAll();

  const afterReadiness = calculateReadiness().total;
  $('#oldCoverage').textContent = `${beforeReadiness}%`;
  $('#newCoverage').textContent = `${afterReadiness}%`;

  const recDetail = proposedTags.recovery === 'yes' ? ' with recovery attempt' : '';
  $('#successCopy').textContent = `A demonstration with ${proposedTags.occlusion} occlusion, ${proposedTags.orientation} object, and ${proposedTags.lighting} lighting (${proposedTags.result.toUpperCase()}${recDetail}) was ingested into the physical AI repository. Deterministic rankings recomputed across all condition candidates.`;

  showView('successView');
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
  $$('.view').forEach(v => v.classList.toggle('active', v.id === viewId));
  $$('.tab-btn').forEach(t => t.classList.toggle('active', t.dataset.view === viewId));
  $$('.mobile-tab-btn').forEach(b => b.classList.toggle('active', b.dataset.view === viewId));

  const menu = $('#moreMenu');
  if (menu) menu.classList.remove('show');

  window.scrollTo({ top: 0, behavior: 'smooth' });

  if (viewId === 'homeView') {
    drawSpatialVisualizer();
  }
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
  $('#quickRecordBtn')?.addEventListener('click', beginCapture);
  $('#quickRecordMobile')?.addEventListener('click', beginCapture);
  $('#recordThis')?.addEventListener('click', beginCapture);
  $('#quickSimulateBtn')?.addEventListener('click', injectSyntheticDemo);

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

  $('#videoFileInput')?.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
      recordedBlob = file;
      toast(`Loaded ${file.name}`);
      frameMetrics = [
        { mean: 125, variance: 820, centerVariance: 680, gx: 14, gy: 11, verticalBias: 4, motion: 12 }
      ];
      analyzeFrames();
      endCamera();
      showView('reviewView');
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
  initEventListeners();
  initSpatialVisualizer();
  loadDataset();
});
