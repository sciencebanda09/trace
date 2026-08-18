const LABELS = ['person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant','bed','dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush'];
let session = null;
let loading = null;

export async function initYolo() {
  if (session) return session;
  if (loading) return loading;
  loading = (async () => {
    if (!globalThis.ort) throw new Error('ONNX Runtime Web did not load');
    ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/';
    const providers = navigator.gpu ? ['webgpu', 'wasm'] : ['wasm'];
    session = await ort.InferenceSession.create('./models/yolov8n.onnx', {
      executionProviders: providers,
      graphOptimizationLevel: 'all'
    });
    return session;
  })();
  return loading;
}

function iou(a, b) {
  const x1 = Math.max(a.x, b.x), y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.width, b.x + b.width), y2 = Math.min(a.y + a.height, b.y + b.height);
  const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  return intersection / Math.max(1, a.width * a.height + b.width * b.height - intersection);
}

function suppress(boxes, threshold = 0.45, limit = 12) {
  const kept = [];
  for (const box of boxes.sort((a, b) => b.confidence - a.confidence)) {
    if (kept.every(other => other.classId !== box.classId || iou(other, box) < threshold)) kept.push(box);
    if (kept.length >= limit) break;
  }
  return kept;
}

export async function detectObjects(source, confidenceThreshold = 0.32) {
  const model = await initYolo();
  const size = 640;
  const canvas = document.createElement('canvas');
  canvas.width = size; canvas.height = size;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(source, 0, 0, size, size);
  const rgba = ctx.getImageData(0, 0, size, size).data;
  const data = new Float32Array(3 * size * size);
  for (let i = 0; i < size * size; i++) {
    data[i] = rgba[i * 4] / 255;
    data[size * size + i] = rgba[i * 4 + 1] / 255;
    data[size * size * 2 + i] = rgba[i * 4 + 2] / 255;
  }
  const input = new ort.Tensor('float32', data, [1, 3, size, size]);
  const output = await model.run({ [model.inputNames[0]]: input });
  const tensor = output[model.outputNames[0]];
  const values = tensor.data;
  const candidates = tensor.dims[1] === 84 ? tensor.dims[2] : tensor.dims[1];
  const channelFirst = tensor.dims[1] === 84;
  const at = (channel, index) => channelFirst ? values[channel * candidates + index] : values[index * 84 + channel];
  const sourceWidth = source.width || source.videoWidth;
  const sourceHeight = source.height || source.videoHeight;
  const boxes = [];
  for (let i = 0; i < candidates; i++) {
    let classId = 0, confidence = 0;
    for (let c = 0; c < 80; c++) {
      const score = at(4 + c, i);
      if (score > confidence) { confidence = score; classId = c; }
    }
    if (confidence < confidenceThreshold) continue;
    const width = at(2, i) / size * sourceWidth;
    const height = at(3, i) / size * sourceHeight;
    const x = at(0, i) / size * sourceWidth - width / 2;
    const y = at(1, i) / size * sourceHeight - height / 2;
    boxes.push({ x, y, width, height, confidence, classId, label: LABELS[classId], relativeDepth: Math.min(1, 1 - Math.sqrt(width * height / (sourceWidth * sourceHeight))) });
  }
  return suppress(boxes);
}

export function drawDetections(canvas, detections, sourceWidth, sourceHeight) {
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.round(rect.width * devicePixelRatio));
  canvas.height = Math.max(1, Math.round(rect.height * devicePixelRatio));
  const ctx = canvas.getContext('2d');
  ctx.scale(devicePixelRatio, devicePixelRatio);
  const sx = rect.width / sourceWidth, sy = rect.height / sourceHeight;
  ctx.lineWidth = 2;
  ctx.font = '600 11px system-ui';
  detections.forEach(d => {
    const x = d.x * sx, y = d.y * sy, w = d.width * sx, h = d.height * sy;
    ctx.strokeStyle = '#34d399'; ctx.fillStyle = '#34d399';
    ctx.strokeRect(x, y, w, h);
    const text = `${d.label} ${Math.round(d.confidence * 100)}%`;
    const tw = ctx.measureText(text).width + 10;
    ctx.fillRect(x, Math.max(0, y - 20), tw, 20);
    ctx.fillStyle = '#04110b'; ctx.fillText(text, x + 5, Math.max(14, y - 6));
  });
}

export function drawScene25D(canvas, detections = []) {
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(320, Math.round(rect.width * devicePixelRatio));
  canvas.height = Math.max(220, Math.round(rect.height * devicePixelRatio));
  const ctx = canvas.getContext('2d');
  const w = canvas.width, h = canvas.height;
  ctx.fillStyle = '#071019'; ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = 'rgba(56,189,248,.18)'; ctx.lineWidth = 1;
  for (let i = 0; i < 9; i++) { const y = h * .55 + i * h * .055; ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke(); }
  for (let i = -8; i <= 8; i++) { ctx.beginPath(); ctx.moveTo(w/2, h*.45); ctx.lineTo(w/2 + i*w*.12, h); ctx.stroke(); }
  detections.forEach((d, index) => {
    const depth = d.relativeDepth;
    const scale = .35 + (1 - depth) * .8;
    const x = (d.x + d.width/2) / 96 * w;
    const y = h * (.48 + depth * .34);
    const bw = Math.max(34, d.width / 96 * w * scale), bh = Math.max(28, d.height / 72 * h * scale);
    const dx = 10 * scale, dy = -8 * scale;
    ctx.fillStyle = `hsla(${155 + index * 31},70%,50%,.18)`; ctx.strokeStyle = '#34d399'; ctx.lineWidth = 2;
    ctx.fillRect(x-bw/2, y-bh, bw, bh); ctx.strokeRect(x-bw/2, y-bh, bw, bh);
    ctx.beginPath(); ctx.moveTo(x-bw/2,y-bh);ctx.lineTo(x-bw/2+dx,y-bh+dy);ctx.lineTo(x+bw/2+dx,y-bh+dy);ctx.lineTo(x+bw/2,y-bh);ctx.stroke();
    ctx.fillStyle='#e2e8f0';ctx.font=`${Math.max(12, w/50)}px system-ui`;ctx.fillText(`${d.label} · z≈${depth.toFixed(2)}`,x-bw/2,y+18);
  });
  if (!detections.length) { ctx.fillStyle='#8194a7';ctx.font=`${Math.max(14,w/45)}px system-ui`;ctx.textAlign='center';ctx.fillText('No YOLO objects detected',w/2,h/2); }
}
