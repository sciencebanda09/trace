const CACHE = 'trace-v14-modular';
const ASSETS = [
  './',
  './index.html',
  './styles.css',
  './app.js',
  './yolo.js',
  './core/schema.js',
  './core/priority-engine.js',
  './core/storage.js',
  './core/evidence.js',
  './core/capture.js',
  './core/simulator.js',
  './core/ui.js',
  './core/workflow.js',
  './core/capture-controller.js',
  './core/spatial-controller.js',
  './core/recommendation-presenter.js',
  './models/yolov8n.onnx',
  './manifest.webmanifest',
  './icon.svg'
];

// Cache the browser-side inference runtimes on first install so a judge can
// reopen the installed app without a network connection after the initial load.
const RUNTIME_ASSETS = [
  'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.min.js',
  'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js',
  'https://cdn.jsdelivr.net/npm/@tensorflow-models/mobilenet@2.1.1/dist/mobilenet.min.js'
];

self.addEventListener('install', e => {
  e.waitUntil(
    caches.open(CACHE)
      .then(async c => {
        await c.addAll(ASSETS);
        await Promise.allSettled(RUNTIME_ASSETS.map(asset => c.add(asset)));
      })
      .then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', e => {
  e.waitUntil(
    caches.keys().then(keys => Promise.all(
      keys.filter(k => k !== CACHE).map(k => caches.delete(k))
    )).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', e => {
  e.respondWith(
    caches.match(e.request).then(r => r || fetch(e.request))
  );
});
