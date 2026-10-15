function readRecords(key) {
  try {
    const value = JSON.parse(localStorage.getItem(key) || '[]');
    return Array.isArray(value) ? value : [];
  } catch {
    return [];
  }
}

export function createState() {
  return {
    sourceLang: 'auto', // 'auto', 'vi', 'en'
    targetLang: 'en',   // 'en', 'vi'
    detectedLang: 'vi',
    modelBackend: localStorage.getItem('phomt_model_backend') === 'legacy' ? 'legacy' : 'marian_ct2',
    method: 'beam',     // 'beam' or 'greedy'
    beamSize: 4,
    maxLength: 60,
    autoTranslate: true,
    history: readRecords('phomt_history'),
    saved: readRecords('phomt_saved')
  };
}
