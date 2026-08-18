import { cp, mkdir, rm } from 'node:fs/promises';
await rm('dist', { recursive: true, force: true });
await mkdir('dist');
for (const file of ['index.html','styles.css','app.js','yolo.js','manifest.webmanifest','sw.js','icon.svg']) await cp(file, `dist/${file}`);
await cp('core', 'dist/core', { recursive: true });
await cp('models', 'dist/models', { recursive: true });
console.log('TRACE built to dist/');
