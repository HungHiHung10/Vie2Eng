export function initLanguages({ state, elements, triggerTranslation, showToast, updateCharCount }) {
  const { sourceText, targetResult, swapLangBtn, srcAutoBtn, srcViBtn, srcEnBtn, tgtEnBtn, tgtViBtn } = elements;

  // ---------------- LANGUAGE SELECTION ----------------
  function renderLanguageSelection() {
    srcAutoBtn.classList.toggle('active', state.sourceLang === 'auto');
    srcViBtn.classList.toggle('active', state.sourceLang === 'vi');
    srcEnBtn.classList.toggle('active', state.sourceLang === 'en');
    tgtEnBtn.classList.toggle('active', state.targetLang === 'en');
    tgtViBtn.classList.toggle('active', state.targetLang === 'vi');
  }

  function setSourceLang(lang) {
    state.sourceLang = lang;
    if (lang === 'auto') {
      srcAutoBtn.textContent = 'Phát hiện ngôn ngữ';
    } else if (state.targetLang === lang) {
      state.targetLang = lang === 'vi' ? 'en' : 'vi';
    }
    renderLanguageSelection();
    if (sourceText.value.trim()) {
      triggerTranslation();
    }
  }

  function renderDetectedSource() {
    const langName = state.detectedLang === 'vi' ? 'Tiếng Việt' : 'Tiếng Anh';
    srcAutoBtn.textContent = state.sourceLang === 'auto'
      ? `${langName} - Được phát hiện` : 'Phát hiện ngôn ngữ';
  }

  function applyResult(data) {
    state.detectedLang = data.detected_lang;
    state.targetLang = data.target_lang;
    renderLanguageSelection();
    renderDetectedSource();
  }

  function restore(item) {
    state.sourceLang = item.srcLang;
    state.targetLang = item.tgtLang;
    state.detectedLang = item.srcLang === 'auto'
      ? (item.tgtLang === 'vi' ? 'en' : 'vi') : item.srcLang;
    renderLanguageSelection();
    renderDetectedSource();
  }

  function setTargetLang(lang) {
    state.targetLang = lang;
    // If source isn't auto and is same as target, flip source
    if (state.sourceLang !== 'auto' && state.sourceLang === lang) {
      state.sourceLang = lang === 'en' ? 'vi' : 'en';
    }
    renderLanguageSelection();

    if (sourceText.value.trim()) {
      triggerTranslation();
    }
  }

  srcAutoBtn.addEventListener('click', () => setSourceLang('auto'));
  srcViBtn.addEventListener('click', () => setSourceLang('vi'));
  srcEnBtn.addEventListener('click', () => setSourceLang('en'));

  tgtEnBtn.addEventListener('click', () => setTargetLang('en'));
  tgtViBtn.addEventListener('click', () => setTargetLang('vi'));

  // Swap Languages
  swapLangBtn.addEventListener('click', () => {
    swapLangBtn.classList.add('rotating');
    setTimeout(() => swapLangBtn.classList.remove('rotating'), 300);

    const oldSrc = state.sourceLang === 'auto' ? state.detectedLang : state.sourceLang;
    const oldTgt = state.targetLang;
    const currentTranslation = targetResult.textContent.trim();

    // Flip languages
    state.sourceLang = oldTgt;
    state.targetLang = oldSrc;
    renderLanguageSelection();

    // If there was an existing translation, swap the texts
    if (currentTranslation && !targetResult.classList.contains('placeholder')) {
      sourceText.value = currentTranslation;
      updateCharCount();
    }
    if (sourceText.value.trim()) triggerTranslation();
    showToast('Đã hoán đổi ngôn ngữ');
  });

  renderLanguageSelection();
  return { applyResult, restore };
}
