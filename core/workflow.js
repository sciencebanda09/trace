export function createOnboardingController({ query, queryAll, onFinish }) {
  let step = 1;
  let mode = 'first-run';

  function render() {
    queryAll('.workflow-step').forEach(panel => panel.classList.toggle('active', Number(panel.dataset.onboardingStep) === step));
    const progress = query('#onboardingProgress');
    const back = query('#onboardingBack');
    const next = query('#onboardingNext');
    if (progress) progress.style.width = `${step * 33.333}%`;
    if (back) back.hidden = step === 1;
    if (next) next.textContent = step === 3 ? (mode === 'failure' ? 'LOG FAILURE' : 'SHOW WHAT TO RECORD') : 'Continue';
    if (step === 3 && query('#setupSummary')) {
      const robot = query('#setupRobot')?.value.trim() || 'your robot';
      const task = query('#setupTask')?.value.trim() || 'the manipulation task';
      query('#setupSummary').textContent = `TRACE will analyze ${robot} performing ${task}, connect the reported failure to dataset gaps, and rank the next capture.`;
    }
  }

  function open(nextStep = 1, nextMode = 'first-run') {
    step = Math.max(1, Math.min(3, nextStep));
    mode = nextMode;
    const modal = query('#onboardingModal');
    if (!modal) return;
    modal.hidden = false;
    render();
    setTimeout(() => (step === 1 ? query('#setupRobot') : query('#setupFailureNotes'))?.focus(), 50);
  }

  function close() {
    const modal = query('#onboardingModal');
    if (modal) modal.hidden = true;
  }

  function bind() {
    query('#onboardingBack')?.addEventListener('click', () => { step = Math.max(1, step - 1); render(); });
    query('#onboardingNext')?.addEventListener('click', async () => {
      if (step < 3) { step += 1; render(); return; }
      await onFinish({ skip: false, mode });
      close();
    });
    query('#onboardingSkip')?.addEventListener('click', async () => { await onFinish({ skip: true, mode }); close(); });
  }

  return { bind, close, open };
}

export function createCaptureEntryController({ query, getInstruction, getReason, onStart }) {
  function open() {
    const instruction = getInstruction();
    if (!instruction) return;
    query('#briefingInstruction').textContent = instruction;
    query('#briefingReason').textContent = getReason();
    query('#captureBriefing').hidden = false;
  }
  function close() { query('#captureBriefing').hidden = true; }
  function bind() {
    query('#closeBriefing')?.addEventListener('click', close);
    query('#startGuidedCapture')?.addEventListener('click', () => { close(); onStart(); });
  }
  return { bind, close, open };
}
