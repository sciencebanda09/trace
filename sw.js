const CACHE = 'trace-v13-aspect-ratio';
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
  './models/yolov8n.onnx',
  './manifest.webmanifest',
  './icon.svg'
];

self.addEventListener('install', e => {
  e.waitUntil(
    caches.open(CACHE).then(c => c.addAll(ASSETS)).then(() => self.skipWaiting())
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
