export function initMedia({ state, elements, triggerTranslation, showToast, updateCharCount }) {
  const { sourceText, targetResult, micBtn, listenSourceBtn, listenTargetBtn, fileUploadInput, docTabBtn } = elements;

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

}
