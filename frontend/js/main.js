/**
 * frontend/js/main.js
 * Application bootstrapper, Control Center event bindings, and SSE real-time stream client.
 */

(function () {
  'use strict';

  /** @type {Record<string, any>} */
  const RadMon = /** @type {any} */ (window).RadMon;
  const {
    state,
    DEFAULT_SOLUTION,
    setText,
    getFormControl,
    getButton,
    apiPost,
    setConnectionStatus
  } = RadMon;

  /**
   * @param {Record<string, any>} frame
   */
  RadMon.applyFramePayload = function (frame) {
    if (!frame) return;
    state.running = Boolean(frame.running);
    state.batchRunning = Boolean(frame.batch_running);
    state.trainingRunning = Boolean(frame.training_running);
    state.facilityKey = frame.facility_key || state.facilityKey;
    state.trueSource = frame.true_source || state.trueSource;
    state.trueZone = frame.true_zone || state.trueZone;
    if (typeof frame.source_activity_kbq === 'number') {
      state.sourceActivityKbq = frame.source_activity_kbq;
    }
    if (typeof frame.ambient_background_cps === 'number') {
      state.ambientBackgroundCps = frame.ambient_background_cps;
    }
    if (typeof frame.mc_histories === 'number') {
      state.mcHistories = frame.mc_histories;
    }
    if (typeof frame.mc_elapsed_ms === 'number') {
      state.mcElapsedMs = frame.mc_elapsed_ms;
    }
    state.mcEngineName = frame.mc_engine || state.mcEngineName;
    if (typeof frame.frame_count === 'number') {
      state.frameCount = frame.frame_count;
    }
    state.hasRun = Boolean(frame.has_run) && state.frameCount > 0;
    state.solution = frame.solution || { ...DEFAULT_SOLUTION };
    state.aiSolution = frame.ai_solution || state.solution;
    state.sensorReadings = Array.isArray(frame.sensor_readings)
      ? frame.sensor_readings
      : state.sensorReadings;

    if (frame.ai_model_status) {
      state.aiModelStatus = frame.ai_model_status;
    }
    if (frame.pathway_telemetry) {
      state.pathwayTelemetry = frame.pathway_telemetry;
    }

    if (frame.log_entry) {
      if (
        state.runLogs.length === 0 ||
        state.runLogs[0].run_id !== frame.log_entry.run_id
      ) {
        state.runLogs.unshift(frame.log_entry);
        if (state.runLogs.length > 200) state.runLogs.pop();
      }
    }

    RadMon.syncControlsUI();
    RadMon.updateKpiCards();
    RadMon.updateSensorTable();
    RadMon.renderFacilityCanvas();
    RadMon.renderNeuralPathwayCanvas();
    RadMon.updateRewardCurveChart();
    RadMon.updateFluxChart();
    RadMon.updateRunLogTable();
  };

  /**
   * @param {Record<string, any>} full
   */
  RadMon.applyFullState = function (full) {
    state.facilities = full.facilities || state.facilities;
    state.running = Boolean(full.running);
    state.facilityKey = full.facility_key || state.facilityKey;
    state.aiModelStatus = full.ai_model_status || state.aiModelStatus;
    state.pathwayTelemetry = full.pathway_telemetry || state.pathwayTelemetry;
    state.validationErrorsAi = full.validation_errors_ai || [];
    state.runLogs = full.recent_logs || [];
    state.trainingSamples = full.training_samples || [];
    state.capturesHistory = Array.isArray(full.captures_history)
      ? full.captures_history
      : [];

    if (state.capturesHistory.length > 0) {
      setText(
        'lastSnapshotBanner',
        `LAST CAPTURE: ${state.capturesHistory[0].folder_path}/`
      );
    }

    RadMon.applyFramePayload(full.latest_frame);
    RadMon.updateTrainingDataTable();
    RadMon.updateCdfChart();
    RadMon.renderNeuralPathwayCanvas();
    RadMon.updateRewardCurveChart();
    RadMon.updateSnapshotArchive();
    RadMon.updateTabBadges();
  };

  function bindControlEvents() {
    function toggleRunningState() {
      apiPost('/api/control', { running: !state.running });
    }

    const statusBadgeEl = document.getElementById('statusBadge');
    if (statusBadgeEl) statusBadgeEl.addEventListener('click', toggleRunningState);

    const btnToggleEl = getButton('btnToggleStream');
    if (btnToggleEl) btnToggleEl.addEventListener('click', toggleRunningState);

    const chkTraj = document.getElementById('chkShowTrajectories');
    if (chkTraj instanceof HTMLInputElement) {
      chkTraj.addEventListener('change', () => {
        state.showTrainingTrajectories = chkTraj.checked;
        RadMon.renderFacilityCanvas();
      });
    }

    const facSelectEl = getFormControl('facilitySelect');
    if (facSelectEl) {
      facSelectEl.addEventListener('change', () => {
        apiPost('/api/control', { facility_key: facSelectEl.value, trigger_run: false });
      });
    }

    const activitySlider = getFormControl('activitySlider');
    if (activitySlider) {
      activitySlider.addEventListener('input', () => {
        setText('activityValue', `${activitySlider.value} kBq`);
      });
      activitySlider.addEventListener('change', () => {
        apiPost('/api/control', {
          source_activity_kbq: Number(activitySlider.value),
          trigger_run: false
        });
      });
    }

    const backgroundSlider = getFormControl('backgroundSlider');
    if (backgroundSlider) {
      backgroundSlider.addEventListener('input', () => {
        setText('backgroundValue', `${backgroundSlider.value} cps`);
      });
      backgroundSlider.addEventListener('change', () => {
        apiPost('/api/control', {
          ambient_background_cps: Number(backgroundSlider.value),
          trigger_run: false
        });
      });
    }

    const btnSimEvent = getButton('btnSimulateEvent');
    if (btnSimEvent) {
      btnSimEvent.addEventListener('click', () => {
        apiPost('/api/simulate-once', { randomize_position: true });
      });
    }

    const btnResample = getButton('btnResampleNoise');
    if (btnResample) {
      btnResample.addEventListener('click', () => {
        apiPost('/api/simulate-once', { randomize_position: false });
      });
    }

    const btnRunVal = getButton('btnRunValidation');
    if (btnRunVal) {
      btnRunVal.addEventListener('click', () => {
        apiPost('/api/run-batch', { n_runs: 100 });
      });
    }

    const btnCap1 = getButton('btnCaptureSnapshot');
    if (btnCap1) btnCap1.addEventListener('click', RadMon.captureSnapshotBundle);

    const btnCap2 = getButton('btnCaptureHeader');
    if (btnCap2) btnCap2.addEventListener('click', RadMon.captureSnapshotBundle);

    const btnTrainOnLogs = getButton('btnTrainOnLogs');
    if (btnTrainOnLogs) {
      btnTrainOnLogs.addEventListener('click', async () => {
        btnTrainOnLogs.disabled = true;
        const res = await apiPost('/api/train-ai', {
          train_source: 'logged_runs',
          epochs: 50
        });
        if (!res || !res.ok) {
          btnTrainOnLogs.disabled = false;
        }
      });
    }

    const btnTrainOpenMC = getButton('btnTrainAiOpenMC');
    if (btnTrainOpenMC) {
      btnTrainOpenMC.addEventListener('click', async () => {
        btnTrainOpenMC.disabled = true;
        const res = await apiPost('/api/train-ai', {
          train_source: 'openmc_batch',
          n_samples: 64,
          epochs: 50
        });
        if (!res || !res.ok) {
          btnTrainOpenMC.disabled = false;
        }
      });
    }

    const btnScrubAi = getButton('btnScrubAi');
    if (btnScrubAi) {
      btnScrubAi.addEventListener('click', () => {
        apiPost('/api/ai/scrub', {});
      });
    }

    const btnScrubAll = getButton('btnScrubAll');
    if (btnScrubAll) {
      btnScrubAll.addEventListener('click', () => {
        apiPost('/api/scrub-all', {});
      });
    }

    const btnClearLogs = getButton('btnClearLogs');
    if (btnClearLogs) {
      btnClearLogs.addEventListener('click', () => {
        apiPost('/api/logs/clear', {});
      });
    }
  }

  async function connectBackend() {
    try {
      const res = await fetch('/api/state');
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const full = await res.json();

      setConnectionStatus(true, 'BACKEND: OPENMC STREAM ONLINE');
      RadMon.applyFullState(full);

      const evtSource = new EventSource('/api/stream');

      evtSource.addEventListener('frame', (e) => {
        const frame = JSON.parse(e.data);
        RadMon.applyFramePayload(frame);
      });

      evtSource.addEventListener('sector_changed', (e) => {
        const fullState = JSON.parse(e.data);
        RadMon.applyFullState(fullState);
      });

      evtSource.addEventListener('status', (e) => {
        const st = JSON.parse(e.data);
        state.running = Boolean(st.running);
        if (st.facility_key) state.facilityKey = st.facility_key;
        if (st.true_source) state.trueSource = st.true_source;
        if (typeof st.source_activity_kbq === 'number') {
          state.sourceActivityKbq = st.source_activity_kbq;
        }
        if (typeof st.ambient_background_cps === 'number') {
          state.ambientBackgroundCps = st.ambient_background_cps;
        }
        if (st.ai_model_status) state.aiModelStatus = st.ai_model_status;
        if (st.pathway_telemetry) state.pathwayTelemetry = st.pathway_telemetry;
        RadMon.syncControlsUI();
        RadMon.updateKpiCards();
        RadMon.renderFacilityCanvas();
        RadMon.renderNeuralPathwayCanvas();
        RadMon.updateRewardCurveChart();
      });

      evtSource.addEventListener('batch_progress', (e) => {
        const prog = JSON.parse(e.data);
        state.batchRunning = true;
        state.validationErrorsAi = prog.validation_errors_ai || state.validationErrorsAi;
        setText('btnRunValidation', `RUNNING BATCH (${prog.completed}/${prog.total})...`);
        RadMon.updateCdfChart();
      });

      evtSource.addEventListener('batch_complete', (e) => {
        const done = JSON.parse(e.data);
        state.batchRunning = false;
        state.validationErrorsAi = done.validation_errors_ai || state.validationErrorsAi;
        setText('btnRunValidation', 'RUN 100-RUN VALIDATION');
        RadMon.syncControlsUI();
        RadMon.updateCdfChart();
      });

      evtSource.addEventListener('ai_training_progress', (e) => {
        const prog = JSON.parse(e.data);
        state.trainingRunning = true;
        if (prog.ai_model_status) state.aiModelStatus = prog.ai_model_status;
        RadMon.syncControlsUI();
      });

      evtSource.addEventListener('ai_training_complete', (e) => {
        const done = JSON.parse(e.data);
        state.trainingRunning = false;
        if (done.ai_model_status) state.aiModelStatus = done.ai_model_status;
        if (done.pathway_telemetry) state.pathwayTelemetry = done.pathway_telemetry;
        if (done.training_samples) state.trainingSamples = done.training_samples;
        if (done.latest_frame) {
          RadMon.applyFramePayload(done.latest_frame);
        }
        const b1 = getButton('btnTrainOnLogs');
        const b2 = getButton('btnTrainAiOpenMC');
        if (b1) b1.disabled = false;
        if (b2) b2.disabled = false;
        RadMon.syncControlsUI();
        RadMon.updateKpiCards();
        RadMon.updateTrainingDataTable();
        RadMon.renderFacilityCanvas();
        RadMon.renderNeuralPathwayCanvas();
        RadMon.updateRewardCurveChart();
      });

      evtSource.addEventListener('snapshot_saved', (e) => {
        const cap = JSON.parse(e.data);
        if (cap && cap.folder_path) {
          setText('lastSnapshotBanner', `SAVED: ${cap.folder_path}/`);
          state.capturesHistory.unshift(cap);
          RadMon.updateSnapshotArchive();
        }
      });

      evtSource.addEventListener('ai_scrubbed', (e) => {
        const d = JSON.parse(e.data);
        if (d.ai_model_status) state.aiModelStatus = d.ai_model_status;
        if (d.pathway_telemetry) state.pathwayTelemetry = d.pathway_telemetry;
        state.trainingSamples = [];
        state.validationErrorsAi = [];
        RadMon.syncControlsUI();
        RadMon.updateKpiCards();
        RadMon.updateTrainingDataTable();
        RadMon.updateCdfChart();
        RadMon.renderFacilityCanvas();
        RadMon.renderNeuralPathwayCanvas();
        RadMon.updateRewardCurveChart();
      });

      evtSource.addEventListener('all_scrubbed', (e) => {
        const fullState = JSON.parse(e.data);
        RadMon.applyFullState(fullState);
      });

      evtSource.addEventListener('logs_cleared', () => {
        state.runLogs = [];
        state.frameCount = 0;
        state.hasRun = false;
        RadMon.updateRunLogTable();
        RadMon.updateKpiCards();
        RadMon.renderFacilityCanvas();
        RadMon.updateTabBadges();
      });

      evtSource.onerror = () => {
        setConnectionStatus(false, 'BACKEND: STREAM RECONNECTING...');
      };
    } catch (err) {
      console.error('Failed to connect to backend:', err);
      setConnectionStatus(false, 'BACKEND: OFFLINE (RUN python3 server.py)');
    }
  }

  function updateUtcClock() {
    const iso = new Date().toISOString();
    setText('utcClock', iso.slice(0, 10) + ' ' + iso.slice(11, 21) + ' UTC');
  }

  // Bootstrap Application
  RadMon.initTabs();
  RadMon.initMapInteractions();
  RadMon.initCharts();
  bindControlEvents();
  window.setInterval(updateUtcClock, 200);
  updateUtcClock();
  connectBackend();
})();

