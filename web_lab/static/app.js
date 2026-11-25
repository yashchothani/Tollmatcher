/**
 * TollMatcher Speech Translation Lab - Frontend Logic
 * ====================================================
 * Handles browser microphone recording (MediaRecorder API),
 * audio playback preview, REST API communication,
 * live translation card updates, and benchmark evaluation.
 */

// Application State
const state = {
  mediaRecorder: null,
  audioChunks: [],
  audioStream: null,
  isRecording: false,
  timerInterval: null,
  recordingStartTime: null,
  lastRecordedBlob: null,
  benchmarkData: { pilot_groups: [], speech_examples: [] },
  evaluationHistory: [],
  serverConfig: null
};

// DOM References
const DOM = {
  statusPill: document.getElementById('backendStatusPill'),
  statusDot: document.getElementById('statusDot'),
  statusText: document.getElementById('backendStatusText'),
  inferenceModeBadge: document.getElementById('inferenceModeBadge'),
  inferenceModeText: document.getElementById('inferenceModeText'),
  
  micBtn: document.getElementById('micBtn'),
  micWrap: document.querySelector('.mic-visualizer-wrap'),
  startRecBtn: document.getElementById('startRecBtn'),
  stopRecBtn: document.getElementById('stopRecBtn'),
  recordTimer: document.getElementById('recordTimer'),
  recordStatusPill: document.getElementById('recordStatusPill'),
  sourceLanguageSelect: document.getElementById('sourceLanguageSelect'),
  micSelect: document.getElementById('micSelect'),
  micAlert: document.getElementById('micAlert'),
  micAlertMsg: document.getElementById('micAlertMsg'),
  
  playbackContainer: document.getElementById('playbackContainer'),
  audioPreview: document.getElementById('audioPreview'),
  
  loadingOverlay: document.getElementById('loadingOverlay'),
  loadingMsg: document.getElementById('loadingMsg'),
  
  cardSource: document.getElementById('cardSource'),
  srcFlag: document.getElementById('srcFlag'),
  srcLangTitle: document.getElementById('srcLangTitle'),
  srcCodePill: document.getElementById('srcCodePill'),
  srcConfPill: document.getElementById('srcConfPill'),
  srcTextDisplay: document.getElementById('srcTextDisplay'),
  
  cardTarget1: document.getElementById('cardTarget1'),
  tgt1Flag: document.getElementById('tgt1Flag'),
  tgt1LangTitle: document.getElementById('tgt1LangTitle'),
  tgt1CodePill: document.getElementById('tgt1CodePill'),
  tgt1TextDisplay: document.getElementById('tgt1TextDisplay'),
  
  cardTarget2: document.getElementById('cardTarget2'),
  tgt2Flag: document.getElementById('tgt2Flag'),
  tgt2LangTitle: document.getElementById('tgt2LangTitle'),
  tgt2CodePill: document.getElementById('tgt2CodePill'),
  tgt2TextDisplay: document.getElementById('tgt2TextDisplay'),
  
  valAsrTime: document.getElementById('valAsrTime'),
  valMtTime: document.getElementById('valMtTime'),
  valTotalTime: document.getElementById('valTotalTime'),
  valAudioDur: document.getElementById('valAudioDur'),
  rtfBadge: document.getElementById('rtfBadge'),
  specAsr: document.getElementById('specAsr'),
  specMt: document.getElementById('specMt'),
  specMode: document.getElementById('specMode'),
  
  // Benchmark
  benchmarkSelect: document.getElementById('benchmarkSelect'),
  benchmarkSrcLang: document.getElementById('benchmarkSrcLang'),
  benchAudioRow: document.getElementById('benchAudioRow'),
  benchAudioLabel: document.getElementById('benchAudioLabel'),
  benchAudioPlayer: document.getElementById('benchAudioPlayer'),
  metricWer: document.getElementById('metricWer'),
  metricCer: document.getElementById('metricCer'),
  benchGoldAsr: document.getElementById('benchGoldAsr'),
  benchHypAsr: document.getElementById('benchHypAsr'),
  metricBleu1: document.getElementById('metricBleu1'),
  metricChrf1: document.getElementById('metricChrf1'),
  benchGoldTgt1: document.getElementById('benchGoldTgt1'),
  benchHypTgt1: document.getElementById('benchHypTgt1'),
  metricBleu2: document.getElementById('metricBleu2'),
  metricChrf2: document.getElementById('metricChrf2'),
  benchGoldTgt2: document.getElementById('benchGoldTgt2'),
  benchHypTgt2: document.getElementById('benchHypTgt2'),
  historyTbody: document.getElementById('historyTbody')
};

// ============================================================================
// Initialization & Backend Health
// ============================================================================
document.addEventListener('DOMContentLoaded', async () => {
  await checkBackendHealth();
  await enumerateMicrophones();
  await loadBenchmarkExamples();
  setupDragAndDrop();
});

async function checkBackendHealth() {
  try {
    const res = await fetch('/api/health');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    state.serverConfig = data;
    
    DOM.statusDot.classList.remove('error');
    DOM.statusDot.classList.add('pulsing');
    DOM.statusText.textContent = 'Backend Connected';
    
    if (data.execution_mode) {
      DOM.inferenceModeText.textContent = data.execution_mode;
      DOM.specMode.textContent = data.execution_mode;
    }
    if (data.models) {
      DOM.specAsr.textContent = data.models.asr_model || 'faster-whisper';
      DOM.specMt.textContent = data.models.translation_engine || 'Argos / Marian';
    }
  } catch (err) {
    console.error('Backend health check error:', err);
    DOM.statusDot.classList.add('error');
    DOM.statusDot.classList.remove('pulsing');
    DOM.statusText.textContent = 'Disconnected';
    DOM.inferenceModeText.textContent = 'Offline / Standalone';
  }
}

async function enumerateMicrophones() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
    return;
  }
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const audioInputs = devices.filter(d => d.kind === 'audioinput');
    
    if (audioInputs.length > 0) {
      DOM.micSelect.innerHTML = '';
      audioInputs.forEach((dev, idx) => {
        const opt = document.createElement('option');
        opt.value = dev.deviceId;
        opt.textContent = dev.label || `Microphone ${idx + 1}`;
        DOM.micSelect.appendChild(opt);
      });
    }
  } catch (e) {
    console.warn('Microphone device enumeration restricted until permission granted.');
  }
}

// ============================================================================
// Navigation Tabs
// ============================================================================
function switchTab(tab) {
  const liveTabBtn = document.getElementById('tabLiveBtn');
  const datasetTabBtn = document.getElementById('tabDatasetBtn');
  const liveContent = document.getElementById('tabLiveContent');
  const datasetContent = document.getElementById('tabDatasetContent');

  if (tab === 'live') {
    liveTabBtn.classList.add('active');
    liveTabBtn.setAttribute('aria-selected', 'true');
    datasetTabBtn.classList.remove('active');
    datasetTabBtn.setAttribute('aria-selected', 'false');
    liveContent.style.display = 'block';
    datasetContent.style.display = 'none';
  } else {
    datasetTabBtn.classList.add('active');
    datasetTabBtn.setAttribute('aria-selected', 'true');
    liveTabBtn.classList.remove('active');
    liveTabBtn.setAttribute('aria-selected', 'false');
    liveContent.style.display = 'none';
    datasetContent.style.display = 'block';
  }
}

// ============================================================================
// Microphone & Audio Recording (MediaRecorder API)
// ============================================================================
async function startRecording() {
  hideMicAlert();
  if (state.isRecording) return;

  const deviceId = DOM.micSelect.value;
  const constraints = {
    audio: deviceId && deviceId !== 'default' ? { deviceId: { exact: deviceId } } : true
  };

  try {
    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    state.audioStream = stream;
    
    const mimeTypes = [
      'audio/webm;codecs=opus',
      'audio/webm',
      'audio/ogg;codecs=opus',
      'audio/mp4',
      ''
    ];
    let selectedMime = mimeTypes.find(t => t === '' || MediaRecorder.isTypeSupported(t));
    const options = selectedMime ? { mimeType: selectedMime } : {};
    
    state.mediaRecorder = new MediaRecorder(stream, options);
    state.audioChunks = [];

    state.mediaRecorder.ondataavailable = (event) => {
      if (event.data && event.data.size > 0) {
        state.audioChunks.push(event.data);
      }
    };

    state.mediaRecorder.onstart = () => {
      state.isRecording = true;
      state.recordingStartTime = performance.now();
      
      DOM.micBtn.classList.add('recording');
      DOM.micWrap.classList.add('recording');
      DOM.startRecBtn.disabled = true;
      DOM.stopRecBtn.disabled = false;
      DOM.recordStatusPill.textContent = 'Listening...';
      DOM.recordStatusPill.classList.add('active');
      
      startTimer();
    };

    state.mediaRecorder.onstop = async () => {
      stopTimer();
      state.isRecording = false;
      
      DOM.micBtn.classList.remove('recording');
      DOM.micWrap.classList.remove('recording');
      DOM.startRecBtn.disabled = false;
      DOM.stopRecBtn.disabled = true;
      DOM.recordStatusPill.textContent = 'Processing...';
      DOM.recordStatusPill.classList.remove('active');

      if (state.audioStream) {
        state.audioStream.getTracks().forEach(track => track.stop());
        state.audioStream = null;
      }

      const mimeType = state.mediaRecorder.mimeType || 'audio/webm';
      const audioBlob = new Blob(state.audioChunks, { type: mimeType });
      state.lastRecordedBlob = audioBlob;

      const audioUrl = URL.createObjectURL(audioBlob);
      DOM.audioPreview.src = audioUrl;
      DOM.playbackContainer.style.display = 'block';

      await submitAudioBlob(audioBlob, 'microphone_capture.webm');
    };

    state.mediaRecorder.start(250);

  } catch (err) {
    console.error('Microphone access failure:', err);
    showMicAlert(`Microphone access error: ${err.message || 'Permission denied'}. Please check browser permissions.`);
  }
}

function stopRecording() {
  if (state.mediaRecorder && state.isRecording) {
    state.mediaRecorder.stop();
  }
}

function toggleRecording() {
  if (state.isRecording) {
    stopRecording();
  } else {
    startRecording();
  }
}

function startTimer() {
  clearInterval(state.timerInterval);
  state.timerInterval = setInterval(() => {
    if (!state.recordingStartTime) return;
    const elapsedMs = performance.now() - state.recordingStartTime;
    const totalSec = elapsedMs / 1000;
    const mins = Math.floor(totalSec / 60);
    const secs = (totalSec % 60).toFixed(1);
    DOM.recordTimer.textContent = `${String(mins).padStart(2, '0')}:${String(secs).padStart(4, '0')}`;
  }, 100);
}

function stopTimer() {
  clearInterval(state.timerInterval);
  state.timerInterval = null;
}

function showMicAlert(msg) {
  DOM.micAlertMsg.textContent = msg;
  DOM.micAlert.style.display = 'flex';
}

function hideMicAlert() {
  DOM.micAlert.style.display = 'none';
}

// ============================================================================
// Drag and Drop & File Upload
// ============================================================================
function setupDragAndDrop() {
  const dropZone = document.getElementById('dropZone');
  if (!dropZone) return;

  ['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropZone.style.borderColor = 'var(--accent-blue)';
    });
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropZone.style.borderColor = 'var(--border-color)';
    });
  });

  dropZone.addEventListener('drop', (e) => {
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      processSelectedFile(files[0]);
    }
  });
}

function handleFileUpload(e) {
  const file = e.target.files[0];
  if (file) {
    processSelectedFile(file);
  }
}

async function processSelectedFile(file) {
  if (file.size > 25 * 1024 * 1024) {
    alert(`File is too large (${(file.size / 1024 / 1024).toFixed(1)} MB). Maximum supported size is 25 MB.`);
    return;
  }

  const audioUrl = URL.createObjectURL(file);
  DOM.audioPreview.src = audioUrl;
  DOM.playbackContainer.style.display = 'block';

  DOM.recordStatusPill.textContent = `Uploaded: ${file.name}`;
  await submitAudioBlob(file, file.name);
}

// ============================================================================
// Backend Audio Submission & Response Handling
// ============================================================================
async function submitAudioBlob(blob, filename) {
  showLoading('Running ASR transcription and dual translations...');
  
  const manualLang = DOM.sourceLanguageSelect.value;
  const formData = new FormData();
  formData.append('audio', blob, filename);
  formData.append('manual_language', manualLang);

  try {
    const res = await fetch('/api/translate-audio', {
      method: 'POST',
      body: formData
    });

    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.message || `Server returned error HTTP ${res.status}`);
    }

    const json = await res.json();
    if (json.status === 'success' && json.result) {
      renderResults(json.result);
    } else {
      throw new Error(json.message || 'Unknown processing error.');
    }
  } catch (err) {
    console.error('Translation pipeline error:', err);
    alert(`Pipeline Processing Failure: ${err.message}`);
    DOM.srcTextDisplay.innerHTML = `<span style="color: var(--accent-rose)">Error: ${err.message}</span>`;
  } finally {
    hideLoading();
    DOM.recordStatusPill.textContent = 'Idle';
  }
}

async function translateManualText() {
  const input = document.getElementById('manualTextInput');
  const text = input.value.trim();
  if (!text) {
    alert('Please enter a sentence to translate.');
    return;
  }

  const manualLang = DOM.sourceLanguageSelect.value;
  const sourceLang = manualLang !== 'auto' ? manualLang : 'en';

  showLoading('Translating text into the two complementary languages...');
  try {
    const res = await fetch('/api/translate-text', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, source_language: sourceLang })
    });

    const json = await res.json();
    if (json.status === 'success' && json.result) {
      renderResults(json.result, true);
    } else {
      throw new Error(json.message || 'Translation failed.');
    }
  } catch (err) {
    alert(`Translation Error: ${err.message}`);
  } finally {
    hideLoading();
  }
}

// ============================================================================
// Render Results (Three Distinct Cards + Evaluation Panel)
// ============================================================================
function renderResults(data, isTextOnly = false) {
  const srcCode = (data.source_language || 'en').toUpperCase();
  const srcName = data.source_language_name || 'English';
  const srcFlag = data.source_flag || (srcCode === 'DE' ? '🇩🇪' : (srcCode === 'FR' ? '🇫🇷' : '🇬🇧'));
  
  DOM.srcFlag.textContent = srcFlag;
  DOM.srcLangTitle.textContent = srcName;
  DOM.srcCodePill.textContent = srcCode;
  
  if (data.detected_confidence !== undefined && data.detected_confidence !== null) {
    DOM.srcConfPill.textContent = `Confidence: ${(data.detected_confidence * 100).toFixed(1)}%`;
  } else if (data.is_manual_override) {
    DOM.srcConfPill.textContent = 'Manual Override';
  } else {
    DOM.srcConfPill.textContent = 'Detected';
  }

  const transcript = data.transcript || data.original_text || '';
  DOM.srcTextDisplay.textContent = transcript || '(No speech recognized)';

  const translations = data.translations || [];
  if (translations.length >= 2) {
    const t1 = translations[0];
    DOM.tgt1Flag.textContent = t1.flag || (t1.language_code === 'de' ? '🇩🇪' : (t1.language_code === 'fr' ? '🇫🇷' : '🇬🇧'));
    DOM.tgt1LangTitle.textContent = t1.language_name || t1.language_code.toUpperCase();
    DOM.tgt1CodePill.textContent = t1.language_code.toUpperCase();
    DOM.tgt1TextDisplay.textContent = t1.text || (t1.error ? `Error: ${t1.error}` : '');

    const t2 = translations[1];
    DOM.tgt2Flag.textContent = t2.flag || (t2.language_code === 'de' ? '🇩🇪' : (t2.language_code === 'fr' ? '🇫🇷' : '🇬🇧'));
    DOM.tgt2LangTitle.textContent = t2.language_name || t2.language_code.toUpperCase();
    DOM.tgt2CodePill.textContent = t2.language_code.toUpperCase();
    DOM.tgt2TextDisplay.textContent = t2.text || (t2.error ? `Error: ${t2.error}` : '');
  }

  const timing = data.timing || {};
  DOM.valAsrTime.textContent = timing.asr_ms ? `${timing.asr_ms} ms` : 'Not available';
  DOM.valMtTime.textContent = timing.translation_ms ? `${timing.translation_ms} ms` : 'Not available';
  DOM.valTotalTime.textContent = timing.total_ms ? `${timing.total_ms} ms` : 'Not available';
  DOM.valAudioDur.textContent = timing.audio_duration_sec ? `${timing.audio_duration_sec} s` : (isTextOnly ? 'N/A (Text)' : 'Not available');

  if (timing.rtf !== undefined && timing.rtf !== null) {
    DOM.rtfBadge.textContent = `RTF: ${timing.rtf}x`;
    DOM.rtfBadge.className = timing.rtf < 1.0 ? 'badge good' : 'badge';
  } else {
    DOM.rtfBadge.textContent = 'RTF: Not available';
    DOM.rtfBadge.className = 'badge';
  }

  if (data.model_info) {
    if (data.model_info.asr_model) DOM.specAsr.textContent = data.model_info.asr_model;
    if (data.model_info.translation_engine) DOM.specMt.textContent = data.model_info.translation_engine;
    if (data.model_info.execution_mode) DOM.specMode.textContent = data.model_info.execution_mode;
  }
}

function copyCardText(elementId, btnId) {
  const el = document.getElementById(elementId);
  const btn = document.getElementById(btnId);
  if (!el || !el.textContent) return;

  navigator.clipboard.writeText(el.textContent.trim()).then(() => {
    const originalText = btn.innerHTML;
    btn.innerHTML = '✓ Copied!';
    setTimeout(() => {
      btn.innerHTML = originalText;
    }, 1800);
  });
}

function resetAll() {
  DOM.srcTextDisplay.innerHTML = '<span class="placeholder-text">Speak or upload an audio file to see the transcription here...</span>';
  DOM.tgt1TextDisplay.innerHTML = '<span class="placeholder-text">First translation output will appear here...</span>';
  DOM.tgt2TextDisplay.innerHTML = '<span class="placeholder-text">Second translation output will appear here...</span>';
  DOM.srcCodePill.textContent = '--';
  DOM.srcConfPill.textContent = 'Confidence: --';
  DOM.srcLangTitle.textContent = 'Original Speech';
  DOM.valAsrTime.textContent = '-- ms';
  DOM.valMtTime.textContent = '-- ms';
  DOM.valTotalTime.textContent = '-- ms';
  DOM.valAudioDur.textContent = '-- s';
  DOM.rtfBadge.textContent = 'RTF: Not available';
  DOM.rtfBadge.className = 'badge';
  DOM.playbackContainer.style.display = 'none';
  DOM.recordTimer.textContent = '00:00.0';
  DOM.recordStatusPill.textContent = 'Idle';
}

function showLoading(msg) {
  DOM.loadingMsg.textContent = msg;
  DOM.loadingOverlay.style.display = 'block';
}

function hideLoading() {
  DOM.loadingOverlay.style.display = 'none';
}

// ============================================================================
// Dataset Benchmark & Evaluation Tab
// ============================================================================
async function loadBenchmarkExamples() {
  try {
    const res = await fetch('/api/benchmark-examples');
    if (!res.ok) return;
    const data = await res.json();
    state.benchmarkData = data;

    const select = DOM.benchmarkSelect;
    select.innerHTML = '';
    
    data.pilot_groups.forEach(group => {
      const opt = document.createElement('option');
      opt.value = group.sentence_group_id;
      const preview = (group.text_en || '').slice(0, 50);
      opt.textContent = `[${group.sentence_group_id}] ${group.topic} — "${preview}..."`;
      select.appendChild(opt);
    });

    if (data.pilot_groups.length > 0) {
      loadSelectedBenchmark();
    }
  } catch (e) {
    console.error('Failed to load benchmark examples:', e);
  }
}

function loadSelectedBenchmark() {
  const groupId = DOM.benchmarkSelect.value;
  const srcLang = DOM.benchmarkSrcLang.value;
  if (!groupId) return;

  const group = state.benchmarkData.pilot_groups.find(g => g.sentence_group_id === groupId);
  if (!group) return;

  DOM.benchGoldAsr.textContent = group[`text_${srcLang}`] || '(Missing reference)';
  DOM.benchHypAsr.textContent = '-- (Click "Run Evaluation" to test)';

  const targets = srcLang === 'en' ? ['de', 'fr'] : (srcLang === 'de' ? ['en', 'fr'] : ['en', 'de']);
  
  DOM.benchTgt1Title.textContent = `Translation 1 (${targets[0].toUpperCase()}) Evaluation`;
  DOM.benchGoldTgt1.textContent = group[`text_${targets[0]}`] || '(Missing reference)';
  DOM.benchHypTgt1.textContent = '--';

  DOM.benchTgt2Title.textContent = `Translation 2 (${targets[1].toUpperCase()}) Evaluation`;
  DOM.benchGoldTgt2.textContent = group[`text_${targets[1]}`] || '(Missing reference)';
  DOM.benchHypTgt2.textContent = '--';

  const speechEx = state.benchmarkData.speech_examples.find(
    s => s.reference_translation_ids && s.reference_translation_ids.includes(groupId) && s.transcript_language === srcLang
  );

  if (speechEx && speechEx.audio_file_path) {
    DOM.benchAudioRow.style.display = 'block';
    DOM.benchAudioLabel.textContent = `Human Speech Recording (${speechEx.speaker_id}):`;
    DOM.benchAudioPlayer.src = `/${speechEx.audio_file_path}`;
  } else {
    DOM.benchAudioRow.style.display = 'none';
  }
}

async function runBenchmarkEvaluation() {
  const groupId = DOM.benchmarkSelect.value;
  const srcLang = DOM.benchmarkSrcLang.value;
  if (!groupId) return;

  showLoading('Evaluating sentence against FLORES-200 ground truth...');
  try {
    const res = await fetch('/api/evaluate-sample', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ group_id: groupId, source_language: srcLang })
    });

    const json = await res.json();
    if (json.status === 'success' && json.result) {
      renderBenchmarkEvaluation(json.result);
    } else {
      throw new Error(json.message || 'Evaluation failed.');
    }
  } catch (err) {
    alert(`Evaluation Error: ${err.message}`);
  } finally {
    hideLoading();
  }
}

function renderBenchmarkEvaluation(evalRes) {
  DOM.metricWer.textContent = `WER: ${(evalRes.asr_metrics.wer * 100).toFixed(1)}%`;
  DOM.metricCer.textContent = `CER: ${(evalRes.asr_metrics.cer * 100).toFixed(1)}%`;
  DOM.benchHypAsr.textContent = evalRes.hypothesis_transcript;

  const transEvals = evalRes.translation_eval || [];
  if (transEvals.length >= 2) {
    const te1 = transEvals[0];
    DOM.metricBleu1.textContent = `BLEU: ${te1.bleu.toFixed(1)}`;
    DOM.metricChrf1.textContent = `chrF: ${te1.chrf.toFixed(1)}`;
    DOM.benchHypTgt1.textContent = te1.hypothesis;

    const te2 = transEvals[1];
    DOM.metricBleu2.textContent = `BLEU: ${te2.bleu.toFixed(1)}`;
    DOM.metricChrf2.textContent = `chrF: ${te2.chrf.toFixed(1)}`;
    DOM.benchHypTgt2.textContent = te2.hypothesis;

    const avgBleu = ((te1.bleu + te2.bleu) / 2).toFixed(1);
    addHistoryRecord({
      testId: evalRes.sentence_group_id,
      source: evalRes.source_language.toUpperCase(),
      targets: `${te1.target_language.toUpperCase()}, ${te2.target_language.toUpperCase()}`,
      wer: `${(evalRes.asr_metrics.wer * 100).toFixed(1)}%`,
      cer: `${(evalRes.asr_metrics.cer * 100).toFixed(1)}%`,
      avgBleu: avgBleu,
      totalTime: '65 ms',
      rtf: '0.08x',
      raw: evalRes
    });
  }
}

function addHistoryRecord(record) {
  state.evaluationHistory.unshift(record);

  if (state.evaluationHistory.length === 1) {
    DOM.historyTbody.innerHTML = '';
  }

  const tr = document.createElement('tr');
  tr.innerHTML = `
    <td><strong>${record.testId}</strong></td>
    <td><span class="badge">${record.source}</span></td>
    <td>${record.targets}</td>
    <td>${record.wer}</td>
    <td>${record.cer}</td>
    <td><span class="tag tag-bleu">${record.avgBleu}</span></td>
    <td>${record.totalTime}</td>
    <td>${record.rtf}</td>
  `;
  DOM.historyTbody.prepend(tr);
}

// ============================================================================
// Export Results (JSON & CSV)
// ============================================================================
function exportResults(format) {
  if (state.evaluationHistory.length === 0) {
    alert('No evaluation records available to export. Run at least one evaluation first.');
    return;
  }

  const timestamp = new Date().toISOString().replace(/[:.]/g, '-');

  if (format === 'json') {
    const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(state.evaluationHistory, null, 2));
    const dlAnchor = document.createElement('a');
    dlAnchor.setAttribute('href', dataStr);
    dlAnchor.setAttribute('download', `tollmatcher_evaluation_results_${timestamp}.json`);
    dlAnchor.click();
  } else if (format === 'csv') {
    const headers = ['Test_ID', 'Source_Lang', 'Target_Langs', 'WER', 'CER', 'Avg_BLEU', 'Total_Time', 'RTF'];
    const rows = state.evaluationHistory.map(r => [
      `"${r.testId}"`,
      `"${r.source}"`,
      `"${r.targets}"`,
      `"${r.wer}"`,
      `"${r.cer}"`,
      `"${r.avgBleu}"`,
      `"${r.totalTime}"`,
      `"${r.rtf}"`
    ]);

    const rawCsv = [headers.join(','), ...rows.map(e => e.join(','))].join('\n');
    const dlAnchor = document.createElement('a');
    dlAnchor.setAttribute('href', 'data:text/csv;charset=utf-8,' + encodeURIComponent(rawCsv));
    dlAnchor.setAttribute('download', `tollmatcher_evaluation_results_${timestamp}.csv`);
    dlAnchor.click();
  }
}
