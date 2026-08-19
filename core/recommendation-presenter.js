const titleCase = value => String(value || '').replace(/-/g, ' ').replace(/\b\w/g, character => character.toUpperCase());

export function buildRecommendationViewModel({ clips, target, rank = 0 }) {
  if (!target) return { instruction: 'Checking failures and dataset gaps...', matching: 0, failures: 0, reason: 'TRACE is calculating the strongest evidence.' };
  const failures = clips.filter(clip => clip.result === 'failure');
  const matching = failures.filter(failure =>
    ['occlusion', 'lighting', 'orientation', 'environment'].filter(key => failure[key] === target[key]).length >= 2
  ).length;
  const recoveryIsSparse = clips.filter(clip => clip.recovery === 'yes').length / Math.max(1, clips.length) < 0.15;
  const occlusion = target.occlusion === 'none' ? 'unobstructed' : `${target.occlusion}-occlusion`;
  const exampleWord = target.count === 1 ? 'demonstration' : 'demonstrations';
  return {
    failures: failures.length,
    matching,
    instruction: `Test this grasp: ${titleCase(occlusion)} · ${titleCase(target.orientation)} object · ${titleCase(target.lighting)} light · ${titleCase(target.environment)}.${recoveryIsSparse ? ' Record the retry too.' : ''}`,
    reason: `Ranked #${rank + 1} because ${matching} of ${failures.length} logged failures share these conditions, but the dataset contains only ${target.count} matching ${exampleWord}.`
  };
}
