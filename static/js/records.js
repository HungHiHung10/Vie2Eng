export function initRecords({ state, elements, showToast, restoreTranslation }) {
  const {
    sourceText, targetResult, starBtn, historyList, savedList, clearHistoryBtn,
    historyCount, savedCount,
  } = elements;

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
    renderCollection(state.history, historyList, 'Chưa có lịch sử dịch');
  }

  function renderSaved() {
    renderCollection(state.saved, savedList, 'Chưa có bản dịch nào được đánh dấu sao', '⭐ ');
  }

  function renderCollection(items, container, emptyMessage, prefix = '') {
    if (items.length === 0) {
      container.innerHTML = `<div style="text-align: center; color: var(--text-muted); padding: 24px;">${emptyMessage}</div>`;
      return;
    }
    container.innerHTML = items.map((item, idx) => `
      <div class="history-item" data-idx="${idx}">
        <div class="history-meta">
          <span>${prefix}${escapeHtml(item.srcLang.toUpperCase())} ➔ ${escapeHtml(item.tgtLang.toUpperCase())} (${escapeHtml(item.time)})</span>
        </div>
        <div class="history-src">${escapeHtml(item.src)}</div>
        <div class="history-tgt">${escapeHtml(item.tgt)}</div>
      </div>
    `).join('');

    container.querySelectorAll('.history-item').forEach(el => {
      el.addEventListener('click', () => {
        restoreTranslation(items[parseInt(el.getAttribute('data-idx'), 10)]);
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

  function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  return { addToHistory, renderHistory, renderSaved };
}
