/**
 * Google Translate - PhoMT Marian NMT Web Client
 * Paired with vienv2.pt & enviv4.pt
 */

document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements
  const sourceText = document.getElementById('sourceText');
  const targetResult = document.getElementById('targetResult');
  const translateBtn = document.getElementById('translateBtn');
  const clearTextBtn = document.getElementById('clearTextBtn');
  const charCounter = document.getElementById('charCounter');
  const swapLangBtn = document.getElementById('swapLangBtn');
  const loadingIndicator = document.getElementById('loadingIndicator');

  // Language selectors
  const srcAutoBtn = document.getElementById('srcAutoBtn');
  const srcViBtn = document.getElementById('srcViBtn');
  const srcEnBtn = document.getElementById('srcEnBtn');
  const tgtEnBtn = document.getElementById('tgtEnBtn');
  const tgtViBtn = document.getElementById('tgtViBtn');

  // Stats / Badges
  const modelBadge = document.getElementById('modelBadge');
  const latencyBadge = document.getElementById('latencyBadge');
  const decodingModeBadge = document.getElementById('decodingModeBadge');
  const modeIconWrapper = document.getElementById('modeIconWrapper');
  const currentModeText = document.getElementById('currentModeText');

  // Action buttons
  const micBtn = document.getElementById('micBtn');
  const listenSourceBtn = document.getElementById('listenSourceBtn');
  const listenTargetBtn = document.getElementById('listenTargetBtn');
  const copyResultBtn = document.getElementById('copyResultBtn');
  const starBtn = document.getElementById('starBtn');
  const thumbUpBtn = document.getElementById('thumbUpBtn');
  const shareBtn = document.getElementById('shareBtn');
  const fileUploadInput = document.getElementById('fileUploadInput');

  // Theme
  const themeToggleBtn = document.getElementById('themeToggleBtn');
  const themeIconSun = document.getElementById('themeIconSun');
  const themeIconMoon = document.getElementById('themeIconMoon');

  // Modals
  const settingsBtn = document.getElementById('settingsBtn');
  const settingsModal = document.getElementById('settingsModal');
  const historyModalBtn = document.getElementById('historyModalBtn');
  const historyModal = document.getElementById('historyModal');
  const savedModalBtn = document.getElementById('savedModalBtn');
  const savedModal = document.getElementById('savedModal');
  const aboutBtn = document.getElementById('aboutBtn');
  const modelInfoTabBtn = document.getElementById('modelInfoTabBtn');
  const aboutModal = document.getElementById('aboutModal');
  const docTabBtn = document.getElementById('docTabBtn');

  // Settings Controls
  const settingMethod = document.getElementById('settingMethod');
  const settingBeamSize = document.getElementById('settingBeamSize');
  const beamSizeVal = document.getElementById('beamSizeVal');
  const settingMaxLen = document.getElementById('settingMaxLen');
  const maxLenVal = document.getElementById('maxLenVal');
  const settingAutoTranslate = document.getElementById('settingAutoTranslate');

  // History & Saved Lists
  const historyList = document.getElementById('historyList');
  const savedList = document.getElementById('savedList');
  const clearHistoryBtn = document.getElementById('clearHistoryBtn');
  const historyCount = document.getElementById('historyCount');
  const savedCount = document.getElementById('savedCount');

  // Toast
  const toast = document.getElementById('toast');
  const toastMsg = document.getElementById('toastMsg');

  // Application State
  const state = {
    sourceLang: 'auto', // 'auto', 'vi', 'en'
    targetLang: 'en',   // 'en', 'vi'
    detectedLang: 'vi',
    method: 'beam',     // 'beam' or 'greedy'
    beamSize: 4,
    maxLength: 60,
    autoTranslate: true,
    lastTranslatedText: '',
    history: JSON.parse(localStorage.getItem('phomt_history') || '[]'),
    saved: JSON.parse(localStorage.getItem('phomt_saved') || '[]')
  };

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

  // ---------------- LANGUAGE SELECTION ----------------
  function setSourceLang(lang) {
    state.sourceLang = lang;
    [srcAutoBtn, srcViBtn, srcEnBtn].forEach(b => b.classList.remove('active'));
    
    if (lang === 'auto') {
      srcAutoBtn.classList.add('active');
      srcAutoBtn.textContent = 'Phát hiện ngôn ngữ';
    } else if (lang === 'vi') {
      srcViBtn.classList.add('active');
      // If source is vi, target should default to en
      if (state.targetLang === 'vi') {
        setTargetLang('en');
      }
    } else if (lang === 'en') {
      srcEnBtn.classList.add('active');
      // If source is en, target should default to vi
      if (state.targetLang === 'en') {
        setTargetLang('vi');
      }
    }
    if (sourceText.value.trim()) {
      triggerTranslation();
    }
  }

  function setTargetLang(lang) {
    state.targetLang = lang;
    [tgtEnBtn, tgtViBtn].forEach(b => b.classList.remove('active'));
    
    if (lang === 'en') {
      tgtEnBtn.classList.add('active');
    } else {
      tgtViBtn.classList.add('active');
    }
    
    // If source isn't auto and is same as target, flip source
    if (state.sourceLang !== 'auto' && state.sourceLang === lang) {
      setSourceLang(lang === 'en' ? 'vi' : 'en');
    }

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
    setSourceLang(oldTgt);
    setTargetLang(oldSrc);

    // If there was an existing translation, swap the texts
    if (currentTranslation && !targetResult.classList.contains('placeholder')) {
      sourceText.value = currentTranslation;
      updateCharCount();
      triggerTranslation();
    }
    showToast('Đã hoán đổi ngôn ngữ');
  });

  // ---------------- TRANSLATION LOGIC ----------------
  let debounceTimeout = null;

  function updateCharCount() {
    const len = sourceText.value.length;
    charCounter.textContent = `${len.toLocaleString()} / 5,000`;
    if (len > 0) {
      clearTextBtn.classList.add('visible');
    } else {
      clearTextBtn.classList.remove('visible');
      targetResult.textContent = 'Bản dịch';
      targetResult.classList.add('placeholder');
      latencyBadge.textContent = 'Sẵn sàng';
    }
  }

  sourceText.addEventListener('input', () => {
    updateCharCount();
    if (state.autoTranslate) {
      clearTimeout(debounceTimeout);
      debounceTimeout = setTimeout(() => {
        if (sourceText.value.trim()) {
          triggerTranslation();
        }
      }, 550);
    }
  });

  translateBtn.addEventListener('click', () => {
    triggerTranslation();
  });

  // Keyboard Shortcuts
  document.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault();
      triggerTranslation();
    }
    if (e.key === 'Escape' && document.activeElement === sourceText) {
      clearText();
    }
    if ((e.ctrlKey || e.metaKey) && e.shiftKey && (e.key === 'S' || e.key === 's')) {
      e.preventDefault();
      swapLangBtn.click();
    }
  });

  function clearText() {
    sourceText.value = '';
    updateCharCount();
    sourceText.focus();
  }
  clearTextBtn.addEventListener('click', clearText);

  async function triggerTranslation() {
    const text = sourceText.value.trim();
    if (!text) return;

    loadingIndicator.classList.add('active');
    latencyBadge.textContent = 'Đang dịch...';

    try {
      const response = await fetch('/api/translate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: text,
          source_lang: state.sourceLang,
          target_lang: state.targetLang,
          method: state.method,
          beam_size: state.beamSize,
          max_length: state.maxLength
        })
      });

      if (!response.ok) {
        throw new Error(`Lỗi máy chủ (${response.status})`);
      }

      const data = await response.json();
      state.detectedLang = data.detected_lang;

      // Update UI with translation
      targetResult.textContent = data.translated_text || ' ';
      targetResult.classList.remove('placeholder');

      // Update source auto badge label if auto was active
      if (state.sourceLang === 'auto') {
        const langName = data.detected_lang === 'vi' ? 'Tiếng Việt' : 'Tiếng Anh';
        srcAutoBtn.textContent = `${langName} - Được phát hiện`;
      }

      // Update meta info
      if (modelBadge && data.model_name) modelBadge.textContent = data.model_name;
      latencyBadge.textContent = `${data.latency_ms} ms ${data.from_cache ? '(Cache)' : ''}`;

      // Save to history
      addToHistory(text, data.translated_text, data.source_lang, data.target_lang, data.model_name);

    } catch (err) {
      console.error(err);
      targetResult.textContent = `Có lỗi xảy ra khi dịch: ${err.message}`;
      targetResult.classList.remove('placeholder');
      latencyBadge.textContent = 'Lỗi kết nối';
    } finally {
      loadingIndicator.classList.remove('active');
    }
  }

  // Sample Chips Click
  document.querySelectorAll('.chip-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      sourceText.value = btn.getAttribute('data-text');
      updateCharCount();
      triggerTranslation();
    });
  });

  // ---------------- SPEECH & AUDIO (TTS & RECOGNITION) ----------------
  // Text-To-Speech
  function speakText(text, lang) {
    if (!text || !window.speechSynthesis) {
      showToast('Trình duyệt không hỗ trợ phát âm');
      return;
    }
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = (lang === 'vi') ? 'vi-VN' : 'en-US';
    utterance.rate = 0.95;
    window.speechSynthesis.speak(utterance);
    showToast(`Đang phát âm (${utterance.lang})...`);
  }

  listenSourceBtn.addEventListener('click', () => {
    const text = sourceText.value.trim();
    if (!text) return;
    const lang = state.sourceLang === 'auto' ? state.detectedLang : state.sourceLang;
    speakText(text, lang);
  });

  listenTargetBtn.addEventListener('click', () => {
    const text = targetResult.textContent.trim();
    if (!text || targetResult.classList.contains('placeholder')) return;
    speakText(text, state.targetLang);
  });

  // Voice Input (Microphone)
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  let recognition = null;
  let isRecording = false;

  if (SpeechRecognition) {
    recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = false;

    recognition.onstart = () => {
      isRecording = true;
      micBtn.classList.add('active');
      showToast('Đang lắng nghe giọng nói...');
    };

    recognition.onresult = (event) => {
      const transcript = event.results[0][0].transcript;
      sourceText.value = (sourceText.value ? sourceText.value + ' ' : '') + transcript;
      updateCharCount();
      triggerTranslation();
    };

    recognition.onerror = (event) => {
      console.warn('Speech recognition error:', event.error);
      showToast('Lỗi nhận diện giọng nói: ' + event.error);
      micBtn.classList.remove('active');
      isRecording = false;
    };

    recognition.onend = () => {
      micBtn.classList.remove('active');
      isRecording = false;
    };

    micBtn.addEventListener('click', () => {
      if (isRecording) {
        recognition.stop();
      } else {
        const lang = state.sourceLang === 'en' ? 'en-US' : 'vi-VN';
        recognition.lang = lang;
        recognition.start();
      }
    });
  } else {
    micBtn.addEventListener('click', () => {
      showToast('Trình duyệt không hỗ trợ Web Speech API');
    });
  }

  // ---------------- FILE UPLOAD (.TXT) ----------------
  docTabBtn.addEventListener('click', () => fileUploadInput.click());

  fileUploadInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (event) => {
      sourceText.value = event.target.result;
      updateCharCount();
      triggerTranslation();
      showToast(`Đã nạp tệp: ${file.name}`);
    };
    reader.readAsText(file);
    e.target.value = '';
  });

  // ---------------- COPY, RATE & SHARE ----------------
  copyResultBtn.addEventListener('click', async () => {
    const text = targetResult.textContent.trim();
    if (!text || targetResult.classList.contains('placeholder')) return;
    try {
      await navigator.clipboard.writeText(text);
      showToast('Đã sao chép vào bộ nhớ tạm');
    } catch (err) {
      showToast('Không thể sao chép');
    }
  });

  if (thumbUpBtn) {
    thumbUpBtn.addEventListener('click', () => {
      thumbUpBtn.classList.toggle('active');
      showToast('Cảm ơn bạn đã đánh giá bản dịch!');
    });
  }

  if (shareBtn) {
    shareBtn.addEventListener('click', async () => {
      const text = targetResult.textContent.trim();
      if (!text || targetResult.classList.contains('placeholder')) return;
      if (navigator.share) {
        try {
          await navigator.share({
            title: 'Google Dịch',
            text: `${sourceText.value}\n➔ ${text}`
          });
        } catch (e) {}
      } else {
        navigator.clipboard.writeText(`${sourceText.value}\n➔ ${text}`);
        showToast('Đã sao chép nội dung chia sẻ');
      }
    });
  }

  starBtn.addEventListener('click', () => {
    const src = sourceText.value.trim();
    const tgt = targetResult.textContent.trim();
    if (!src || !tgt || targetResult.classList.contains('placeholder')) return;

    const existingIndex = state.saved.findIndex(s => s.src === src && s.tgt === tgt);
    if (existingIndex >= 0) {
      state.saved.splice(existingIndex, 1);
      starBtn.classList.remove('active');
      showToast('Đã xóa khỏi mục Đã lưu');
    } else {
      state.saved.unshift({
        src, tgt,
        srcLang: state.sourceLang,
        tgtLang: state.targetLang,
        time: new Date().toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit' })
      });
      starBtn.classList.add('active');
      showToast('Đã lưu vào danh sách Yêu thích');
    }
    localStorage.setItem('phomt_saved', JSON.stringify(state.saved));
    updateSavedCount();
  });

  // ---------------- HISTORY & SAVED MANAGERS ----------------
  function addToHistory(src, tgt, srcLang, tgtLang, modelName) {
    if (!src || !tgt) return;
    // Don't duplicate top entry
    if (state.history.length > 0 && state.history[0].src === src && state.history[0].tgt === tgt) {
      return;
    }
    state.history.unshift({
      src, tgt, srcLang, tgtLang, modelName,
      time: new Date().toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit' })
    });
    if (state.history.length > 50) {
      state.history.pop();
    }
    localStorage.setItem('phomt_history', JSON.stringify(state.history));
    updateHistoryCount();
  }

  function updateHistoryCount() {
    historyCount.textContent = state.history.length;
  }
  function updateSavedCount() {
    savedCount.textContent = state.saved.length;
  }
  updateHistoryCount();
  updateSavedCount();

  function renderHistory() {
    if (state.history.length === 0) {
      historyList.innerHTML = '<div style="text-align: center; color: var(--text-muted); padding: 24px;">Chưa có lịch sử dịch</div>';
      return;
    }
    historyList.innerHTML = state.history.map((item, idx) => `
      <div class="history-item" data-idx="${idx}">
        <div class="history-meta">
          <span>${item.srcLang.toUpperCase()} ➔ ${item.tgtLang.toUpperCase()} (${item.time})</span>
        </div>
        <div class="history-src">${escapeHtml(item.src)}</div>
        <div class="history-tgt">${escapeHtml(item.tgt)}</div>
      </div>
    `).join('');

    historyList.querySelectorAll('.history-item').forEach(el => {
      el.addEventListener('click', () => {
        const item = state.history[parseInt(el.getAttribute('data-idx'))];
        sourceText.value = item.src;
        targetResult.textContent = item.tgt;
        targetResult.classList.remove('placeholder');
        updateCharCount();
        closeAllModals();
      });
    });
  }

  function renderSaved() {
    if (state.saved.length === 0) {
      savedList.innerHTML = '<div style="text-align: center; color: var(--text-muted); padding: 24px;">Chưa có bản dịch nào được đánh dấu sao</div>';
      return;
    }
    savedList.innerHTML = state.saved.map((item, idx) => `
      <div class="history-item" data-idx="${idx}">
        <div class="history-meta">
          <span>⭐ ${item.srcLang.toUpperCase()} ➔ ${item.tgtLang.toUpperCase()} (${item.time})</span>
        </div>
        <div class="history-src">${escapeHtml(item.src)}</div>
        <div class="history-tgt">${escapeHtml(item.tgt)}</div>
      </div>
    `).join('');

    savedList.querySelectorAll('.history-item').forEach(el => {
      el.addEventListener('click', () => {
        const item = state.saved[parseInt(el.getAttribute('data-idx'))];
        sourceText.value = item.src;
        targetResult.textContent = item.tgt;
        targetResult.classList.remove('placeholder');
        updateCharCount();
        closeAllModals();
      });
    });
  }

  clearHistoryBtn.addEventListener('click', () => {
    state.history = [];
    localStorage.removeItem('phomt_history');
    renderHistory();
    updateHistoryCount();
    showToast('Đã xóa toàn bộ lịch sử dịch');
  });

  // ---------------- MODAL CONTROLS ----------------
  function openModal(modal) {
    modal.classList.add('active');
  }
  function closeAllModals() {
    document.querySelectorAll('.modal-overlay').forEach(m => m.classList.remove('active'));
  }

  settingsBtn.addEventListener('click', () => openModal(settingsModal));
  historyModalBtn.addEventListener('click', () => {
    renderHistory();
    openModal(historyModal);
  });
  savedModalBtn.addEventListener('click', () => {
    renderSaved();
    openModal(savedModal);
  });
  if (aboutBtn && aboutModal) aboutBtn.addEventListener('click', () => openModal(aboutModal));
  if (modelInfoTabBtn && aboutModal) modelInfoTabBtn.addEventListener('click', () => openModal(aboutModal));

  document.querySelectorAll('.close-modal-btn').forEach(btn => {
    btn.addEventListener('click', closeAllModals);
  });
  document.querySelectorAll('.modal-overlay').forEach(overlay => {
    overlay.addEventListener('click', (e) => {
      if (e.target === overlay) closeAllModals();
    });
  });

  // ---------------- TOAST ----------------
  let toastTimer = null;
  function showToast(msg) {
    toastMsg.textContent = msg;
    toast.classList.add('show');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => {
      toast.classList.remove('show');
    }, 2500);
  }

  function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  // Initial update
  updateModeBadge();
});
