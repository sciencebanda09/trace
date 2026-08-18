export const TRACE_SCHEMA = Object.freeze({
  occlusion: ['none', 'partial', 'heavy'], lighting: ['bright', 'normal', 'low-light'],
  orientation: ['upright', 'rotated', 'inverted'], environment: ['bench', 'floor'],
  result: ['success', 'failure'], recovery: ['no', 'yes']
});
export const CONTEXT_KEYS = Object.freeze(['occlusion','lighting','orientation','environment']);
export const DEFAULT_WEIGHTS = Object.freeze({ gap:.30, failure:.50, novelty:.20, costSensitivity:1 });
export const CAPTURE_CONFIG = Object.freeze({ defaultMaxDurationSeconds:30, keyframeIntervalMs:750, uploadedVideoSampleLimit:12 });
