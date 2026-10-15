export function initPreferences({ state, elements, triggerTranslation, showToast }) {
  const {
    sourceText, modelBadge, decodingModeBadge, modeIconWrapper, currentModeText,
    themeToggleBtn, themeIconSun, themeIconMoon, settingModel, settingMethod,
    settingBeamSize, beamSizeVal, settingMaxLen, maxLenVal, settingAutoTranslate,
  } = elements;

  // ---------------- THEME INITIALIZATION ----------------
  const currentTheme = localStorage.getItem('phomt_theme') || 'light';
  document.documentElement.setAttribute('data-theme', currentTheme);
  updateThemeIcons(currentTheme);

  themeToggleBtn.addEventListener('click', () => {
    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    const newTheme = isDark ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', newTheme);
    localStorage.setItem('phomt_theme', newTheme);
    updateThemeIcons(newTheme);
    showToast(`Đã chuyển sang giao diện ${newTheme === 'dark' ? 'tối' : 'sáng'}`);
  });

  function updateThemeIcons(theme) {
    if (theme === 'dark') {
      themeIconSun.style.display = 'none';
      themeIconMoon.style.display = 'block';
    } else {
      themeIconSun.style.display = 'block';
      themeIconMoon.style.display = 'none';
    }
  }

  // Professional vector icons for decoding algorithms
  const SVG_BEAM = `<svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor">
    <path d="M22 11V3h-7v3H9V3H2v8h7V8h2v10h4v3h7v-8h-7v3h-2V8h2v3h7zM4 9V5h3v4H4zm13 8v-4h3v4h-3zm0-8V5h3v4h-3z"/>
  </svg>`;

  const SVG_GREEDY = `<svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor">
    <path d="M20.38 8.57l-1.23 1.85a8 8 0 0 1-.22 7.58H5.07A8 8 0 0 1 15.58 6.85l1.85-1.23A10 10 0 0 0 3.35 19a2 2 0 0 0 1.72 1h13.85a2 2 0 0 0 1.74-1 10 10 0 0 0-.28-10.43zM10.59 15.41a2 2 0 0 0 2.83 0l5.66-8.49-8.49 5.66a2 2 0 0 0 0 2.83z"/>
  </svg>`;

  // ---------------- BADGE & SETTINGS SYNC ----------------
  function updateModeBadge() {
    if (!decodingModeBadge) return;
    if (state.method === 'greedy') {
      if (modeIconWrapper) modeIconWrapper.innerHTML = SVG_GREEDY;
      if (currentModeText) currentModeText.textContent = 'Greedy Search';
      decodingModeBadge.setAttribute('data-mode', 'greedy');
      decodingModeBadge.title = 'Chế độ Greedy Search (Tốc độ cao) - Nhấp để chuyển sang Beam Search';
    } else {
      if (modeIconWrapper) modeIconWrapper.innerHTML = SVG_BEAM;
      if (currentModeText) currentModeText.textContent = `Beam Search (k=${state.beamSize})`;
      decodingModeBadge.setAttribute('data-mode', 'beam');
      decodingModeBadge.title = `Chế độ Beam Search (k=${state.beamSize}, Độ chính xác cao) - Nhấp để chuyển sang Greedy Search`;
    }
  }

  if (decodingModeBadge) {
    decodingModeBadge.addEventListener('click', () => {
    // Quick toggle between greedy and beam
    if (state.method === 'beam') {
      state.method = 'greedy';
      settingMethod.value = 'greedy';
      showToast('Đã chuyển sang giải thuật: Greedy Search (Tốc độ)');
    } else {
      state.method = 'beam';
      settingMethod.value = 'beam';
      showToast(`Đã chuyển sang giải thuật: Beam Search (k=${state.beamSize})`);
    }
    updateModeBadge();
    if (sourceText.value.trim()) {
      triggerTranslation();
    }
    });
  }

  settingModel.value = state.modelBackend;
  if (modelBadge) modelBadge.textContent = state.modelBackend === 'legacy' ? 'PhoMT Transformer' : 'MarianNMT INT8';
  settingModel.addEventListener('change', (e) => {
    state.modelBackend = e.target.value;
    localStorage.setItem('phomt_model_backend', state.modelBackend);
    if (modelBadge) modelBadge.textContent = state.modelBackend === 'legacy' ? 'PhoMT Transformer' : 'MarianNMT INT8';
    if (sourceText.value.trim()) triggerTranslation();
  });

  settingBeamSize.addEventListener('input', (e) => {
    state.beamSize = parseInt(e.target.value);
    beamSizeVal.textContent = state.beamSize;
    updateModeBadge();
  });

  settingMaxLen.addEventListener('input', (e) => {
    state.maxLength = parseInt(e.target.value);
    maxLenVal.textContent = state.maxLength;
  });

  settingMethod.addEventListener('change', (e) => {
    state.method = e.target.value;
    updateModeBadge();
  });

  settingAutoTranslate.addEventListener('change', (e) => {
    state.autoTranslate = e.target.checked;
  });

  updateModeBadge();
}
