/** Owns request ordering and debounce timing for every translation entry point. */
export function createTranslationController({ state, elements, onResult }) {
  const { sourceText, targetResult, loadingIndicator, latencyBadge } = elements;
  let generation = 0;
  let debounceTimer = null;

  function invalidate() {
    generation += 1;
    clearTimeout(debounceTimer);
    loadingIndicator.classList.remove('active');
  }

  function schedule() {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(translate, 550);
  }

  async function translate() {
    clearTimeout(debounceTimer);
    const text = sourceText.value.trim();
    if (!text) return;
    const requestGeneration = ++generation;
    loadingIndicator.classList.add('active');
    latencyBadge.textContent = 'Đang dịch...';

    try {
      const response = await fetch('/api/translate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text, source_lang: state.sourceLang, target_lang: state.targetLang,
          model_backend: state.modelBackend, method: state.method,
          beam_size: state.beamSize, max_length: state.maxLength,
        }),
      });
      if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        throw new Error(typeof error.detail === 'string' ? error.detail : `Lỗi máy chủ (${response.status})`);
      }
      const data = await response.json();
      if (requestGeneration !== generation) return;
      onResult(text, data);
    } catch (error) {
      if (requestGeneration !== generation) return;
      console.error(error);
      targetResult.textContent = `Có lỗi xảy ra khi dịch: ${error.message}`;
      targetResult.classList.remove('placeholder');
      latencyBadge.textContent = 'Lỗi kết nối';
    } finally {
      if (requestGeneration === generation) loadingIndicator.classList.remove('active');
    }
  }

  return { translate, invalidate, schedule };
}
