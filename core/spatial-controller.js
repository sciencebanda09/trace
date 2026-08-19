// Optional presentation controller. It depends on app services through explicit callbacks.
export function createSpatialController({ query, queryAll, getRecommendation, getRankedExperiments, titleCase, toast, beginCapture }) {
  const $ = query;
  const $$ = queryAll;
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
   Interactive 3D Physical Scene Visualizer & Spatial Simulator Engine
   ========================================================================== */

function getActiveSpatialConfig() {
  if (spatial.override) {
    return {
      environment: spatial.override.environment || getRecommendation()?.environment || 'bench',
      orientation: spatial.override.orientation || getRecommendation()?.orientation || 'upright',
      occlusion: spatial.override.occlusion || getRecommendation()?.occlusion || 'none',
      lighting: spatial.override.lighting || getRecommendation()?.lighting || 'normal',
      isOverride: true
    };
  }
  return getRecommendation() || {
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
          environment: getRecommendation()?.environment || 'bench',
          orientation: getRecommendation()?.orientation || 'upright',
          occlusion: getRecommendation()?.occlusion || 'none',
          lighting: getRecommendation()?.lighting || 'normal'
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


  return { init: initSpatialVisualizer, draw: drawSpatialVisualizer, updateHUD: updateSpatialHUD };
}
