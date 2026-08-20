import { initYolo, detectObjects, drawDetections, getYoloRuntime } from '../yolo.js';
import { TRACE_SCHEMA as SCHEMA, CAPTURE_CONFIG } from './schema.js';
import { calibrationCorrection, confidenceLabel } from './evidence.js';
import { describeSourceDevice, formatClock } from './capture.js';

export function createCaptureController({ query, queryAll, taskProfile, getClips, getRecommendation, getRankedExperiments, getDeviceCapabilities, saveRecord, renderAll, calculateReadiness, recommendationEvidence, showView, toast, titleCase, escapeHtml }) {
  const $ = query;
  const $$ = queryAll;
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
    ['Storage', getDeviceCapabilities()?.indexedDB ? 'local' : 'memory', Boolean(getDeviceCapabilities()?.indexedDB)],
    ['Latency', yoloLatencyMs ? `${Math.round(yoloLatencyMs)}ms` : 'not measured', !yoloLatencyMs || yoloLatencyMs < 1500]
  ];
  container.innerHTML = items.map(([name,value,ok])=>`<span class="${ok?'healthy':'degraded'}"><i></i>${name}<b>${value}</b></span>`).join('');
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

  if (!getDeviceCapabilities()?.camera) {
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
  if (!getDeviceCapabilities()?.imu) return;
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
  const beforeRecommendation = getRecommendation() ? { ...getRecommendation() } : null;
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
      imu: { available: getDeviceCapabilities()?.imu || false, samples: imuSamples },
      gps: { available: getDeviceCapabilities()?.gps || false, collected: false },
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

  getClips().push(newRecord);
  await saveRecord(newRecord);
  renderAll();

  const afterReadiness = calculateReadiness().total;
  $('#oldCoverage').textContent = `${beforeReadiness}%`;
  $('#newCoverage').textContent = `${afterReadiness}%`;
  renderImpactComparison(beforeImpact, distributionSnapshot());

  if (beforeRecommendation) {
    const contextKeys = ['occlusion', 'lighting', 'orientation', 'environment'];
    const updatedTarget = getRankedExperiments().find(item => contextKeys.every(key => item[key] === beforeRecommendation[key]));
    const addedExamples = Math.max(0, (updatedTarget?.count ?? beforeRecommendation.count) - beforeRecommendation.count);
    const gapReduction = Math.max(0, Math.round((beforeRecommendation.gap - (updatedTarget?.gap ?? beforeRecommendation.gap)) * 100));
    if ($('#targetExampleDelta')) $('#targetExampleDelta').textContent = `${beforeRecommendation.count} → ${updatedTarget?.count ?? beforeRecommendation.count}`;
    if ($('#gapReductionDelta')) $('#gapReductionDelta').textContent = addedExamples ? `−${gapReduction} pts` : 'Rebalanced';
    const nextChanged = getRecommendation() && contextKeys.some(key => getRecommendation()[key] !== beforeRecommendation[key]);
    if ($('#nextRecommendationReason')) $('#nextRecommendationReason').innerHTML = `<b>${nextChanged ? 'NEXT PRIORITY CHANGED' : 'PRIORITY RECALCULATED'}</b><br>${escapeHtml(recommendationEvidence().text)}`;
  }

  const recDetail = proposedTags.recovery === 'yes' ? ' with recovery attempt' : '';
  $('#successCopy').textContent = `${formatDuration(recordedDurationSeconds)} attempt saved: ${titleCase(proposedTags.occlusion)} occlusion · ${titleCase(proposedTags.lighting)} light · ${titleCase(proposedTags.result)}${recDetail}. TRACE checked the updated evidence and recalculated what to record next.`;

  showView('successView');
}

function distributionSnapshot() {
  const total = Math.max(1, getClips().length);
  const values = {};
  Object.entries(SCHEMA).forEach(([attribute, options]) => options.forEach(value => {
    values[`${attribute}:${value}`] = getClips().filter(clip => clip[attribute] === value).length / total;
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


  function bind() {
    query('#reRecordBtn')?.addEventListener('click', () => {
      if (playbackUrl) URL.revokeObjectURL(playbackUrl);
      playbackUrl = null;
      beginCapture();
    });
    queryAll('[data-confirm-result]').forEach(button => button.addEventListener('click', () => {
      proposedTags.result = button.dataset.confirmResult;
      outcomeConfirmed = true;
      queryAll('[data-confirm-result]').forEach(item => item.classList.toggle('selected', item === button));
      query('#saveClip').disabled = false;
    }));
    query('#confirmRecovery')?.addEventListener('change', event => { proposedTags.recovery = event.target.checked ? 'yes' : 'no'; });
    query('#toggleDiagnostics')?.addEventListener('click', () => {
      const wrapper = query('.review-wrapper');
      const button = query('#toggleDiagnostics');
      const expanded = !wrapper?.classList.contains('show-diagnostics');
      wrapper?.classList.toggle('show-diagnostics', expanded);
      button?.setAttribute('aria-expanded', String(expanded));
      if (button) button.textContent = expanded ? 'Hide technical analysis' : 'Show technical analysis';
    });
    query('#closeCapture')?.addEventListener('click', () => { endCamera(); showView('homeView'); });
    query('#recordBtn')?.addEventListener('click', toggleRecording);
    query('#switchCameraBtn')?.addEventListener('click', () => {
      currentFacingMode = currentFacingMode === 'environment' ? 'user' : 'environment';
      endCamera();
      beginCapture();
    });
    query('#videoFileInput')?.addEventListener('change', async event => {
      const file = event.target.files[0];
      if (file) await analyzeUploadedVideo(file);
    });
    query('#saveClip')?.addEventListener('click', commitClip);
    query('#discardClip')?.addEventListener('click', () => {
      recordedBlob = null;
      capturedFrameCanvases = [];
      showView('homeView');
      toast('Clip discarded');
    });
  }

  return { analyzeUpload: analyzeUploadedVideo, begin: beginCapture, bind, commit: commitClip, loadModels: () => { loadVisionModel(); loadYoloModel(); } };
}
