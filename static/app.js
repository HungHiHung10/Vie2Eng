import { getElements } from './js/dom.js';
import { createState } from './js/state.js';
import { initLanguages } from './js/languages.js';
import { initPreferences } from './js/preferences.js';
import { initMedia } from './js/media.js';
import { initRecords } from './js/records.js';
import { createTranslationController } from './js/translation.js';

document.addEventListener('DOMContentLoaded', () => {
  const elements = getElements();
  const state = createState();
  const {
    sourceText, targetResult, clearTextBtn, charCounter, swapLangBtn,
    modelBadge, latencyBadge, copyResultBtn, thumbUpBtn, shareBtn,
    settingsBtn, settingsModal, historyModalBtn, historyModal,
    savedModalBtn, savedModal, aboutBtn, modelInfoTabBtn, aboutModal, toast, toastMsg,
  } = elements;

  const translation = createTranslationController({ state, elements, onResult: displayTranslation });
  const shared = { state, elements, triggerTranslation, showToast, updateCharCount };
  const languages = initLanguages(shared);
  initPreferences(shared);
  const records = initRecords({ state, elements, showToast, restoreTranslation });
  initMedia(shared);

  function triggerTranslation() {
    return translation.translate();
  }

  function displayTranslation(text, data) {
    languages.applyResult(data);
    targetResult.textContent = data.translated_text || ' ';
    targetResult.classList.remove('placeholder');
    if (modelBadge && data.model_name) {
      modelBadge.textContent = data.model_name.includes('ct2-int8') ? 'MarianNMT INT8' : 'PhoMT Transformer';
      modelBadge.title = data.model_name;
    }
    latencyBadge.textContent = `${data.latency_ms} ms ${data.from_cache ? '(Cache)' : ''}`;
    records.addToHistory(text, data.translated_text, data.source_lang, data.target_lang, data.model_name);
  }

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
    translation.invalidate();
    updateCharCount();
    if (state.autoTranslate) translation.schedule();
  });
  elements.translateBtn.addEventListener('click', triggerTranslation);

  function clearText() {
    translation.invalidate();
    sourceText.value = '';
    updateCharCount();
    sourceText.focus();
  }
  elements.clearTextBtn.addEventListener('click', clearText);

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

  // Sample Chips Click
  document.querySelectorAll('.chip-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      sourceText.value = btn.getAttribute('data-text');
      updateCharCount();
      triggerTranslation();
    });
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

  function restoreTranslation(item) {
    // Restoring a record must also invalidate any pending translation.
    translation.invalidate();
    languages.restore(item);
    sourceText.value = item.src;
    targetResult.textContent = item.tgt;
    targetResult.classList.remove('placeholder');
    updateCharCount();
    latencyBadge.textContent = 'Đã khôi phục';
    closeAllModals();
  }

  // ---------------- MODAL CONTROLS ----------------
  function openModal(modal) {
    modal.classList.add('active');
  }
  function closeAllModals() {
    document.querySelectorAll('.modal-overlay').forEach(m => m.classList.remove('active'));
  }

  settingsBtn.addEventListener('click', () => openModal(settingsModal));
  historyModalBtn.addEventListener('click', () => {
    records.renderHistory();
    openModal(historyModal);
  });
  savedModalBtn.addEventListener('click', () => {
    records.renderSaved();
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

});
