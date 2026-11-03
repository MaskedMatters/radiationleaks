/**
 * RAD-MON-04 // Frontend Telemetry Control Center
 * Streams real-time OpenMC photon transport & Pure AI Neural Network localization
 * telemetry from the Python backend (`server.py`).
 *
 * All AI weights, training samples, and run logs are persisted in `data/` across
 * server restarts. Includes one-click snapshot capture of the 2D map and all charts
 * into timestamped folders under `captures/`.
 */

(function () {
  'use strict';

  const DOMAIN_SIZE = 20.0;
  const CANVAS_SIZE = 660;
  const PAD = 38;

  const DEFAULT_SOLUTION = {
    solver: 'STANDBY',
    is_trained: false,
    trained_epochs: 0,
    x: 12.4,
    y: 8.15,
    activity: 0.0,
    error_meters: 0.0,
    ci95_meters: 0.0,
    latency_ms: 0.0,
    iterations: 0,
    converged: false
  };

  /** @type {Record<string, any>} */
  const state = {
    connected: false,
    hasRun: false,
    running: false,
    batchRunning: false,
    trainingRunning: false,
    facilityKey: 'sector-a',
    facilities: {},
    trueSource: { x: 12.4, y: 8.15 },
    trueZone: 'AUX-BAY-2',
    sourceActivityKbq: 850,
    ambientBackgroundCps: 18,
    mcHistories: 25000,
    mcElapsedMs: 0.0,
    mcEngineName: 'openmc-photon-mc-kernel',
    frameCount: 0,
    sensorReadings: [],
    solution: { ...DEFAULT_SOLUTION },
    aiSolution: { ...DEFAULT_SOLUTION },
    aiModelStatus: {
      model_version: 'RADNET-v3.1-MLP',
      is_trained: false,
      knowledge_state: 'UNTRAINED (0 KNOWLEDGE)',
      trained_epochs: 0,
      samples_seen: 0,
      total_samples: 0,
      last_loss: 0.0,
      val_mae_meters: 0.0,
      weight_norms: { W1_l2: 0.0, W2_l2: 0.0, W3_l2: 0.0 }
    },
    validationErrorsAi: [],
    runLogs: [],
    trainingSamples: [],
    hoveredSensorId: '',
    draggingSource: false
  };

  // --- DOM Helpers ---
  /**
   * @param {string} id
   * @param {string} text
   */
  function setText(id, text) {
    const el = document.getElementById(id);
    if (el) {
      el.textContent = text;
    }
  }

  /**
   * @param {string} id
   * @returns {HTMLInputElement | HTMLSelectElement | null}
   */
  function getFormControl(id) {
    const el = document.getElementById(id);
    if (el instanceof HTMLInputElement || el instanceof HTMLSelectElement) {
      return el;
    }
    return null;
  }

  /**
   * @param {string} id
   * @returns {HTMLButtonElement | null}
   */
  function getButton(id) {
    const el = document.getElementById(id);
    return el instanceof HTMLButtonElement ? el : null;
  }

  // --- REST API Helper ---
  /**
   * @param {string} path
   * @param {Record<string, any>} payload
   */
  async function apiPost(path, payload) {
    try {
      const res = await fetch(path, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload || {})
      });
      const data = await res.json();
      if (!res.ok) {
        if (data && data.error) {
          window.alert(String(data.error));
        }
        return data;
      }
      return data;
    } catch (err) {
      console.error(`API POST ${path} failed:`, err);
      setConnectionStatus(false, '');
      return null;
    }
  }

  /**
   * @param {boolean} ok
   * @param {string} customText
   */
  function setConnectionStatus(ok, customText) {
    state.connected = ok;
    const pill = document.getElementById('backendConnPill');
    if (!pill) return;
    if (ok) {
      pill.className = 'meta-pill conn-ok';
      pill.textContent = customText || 'BACKEND: OPENMC STREAM ONLINE';
    } else {
      pill.className = 'meta-pill conn-err';
      pill.textContent = customText || 'BACKEND: OFFLINE (RUN python3 server.py)';
    }
  }

  // --- Apply State & Frame Payloads from Backend ---
  /**
   * @param {Record<string, any>} frame
   */
  function applyFramePayload(frame) {
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

    if (frame.log_entry) {
      if (
        state.runLogs.length === 0 ||
        state.runLogs[0].run_id !== frame.log_entry.run_id
      ) {
        state.runLogs.unshift(frame.log_entry);
        if (state.runLogs.length > 200) state.runLogs.pop();
      }
    }

    syncControlsUI();
    updateKpiCards();
    updateSensorTable();
    renderFacilityCanvas();
    updateFluxChart();
    updateRunLogTable();
  }

  function syncControlsUI() {
    const statusBadge = document.getElementById('statusBadge');
    const btnToggleStream = getButton('btnToggleStream');

    if (state.running) {
      if (statusBadge) {
        statusBadge.classList.remove('standby');
        statusBadge.classList.add('running');
      }
      setText('statusText', 'RUNNING');
      if (btnToggleStream) {
        btnToggleStream.textContent = 'STOP CONTINUOUS RUN';
        btnToggleStream.classList.remove('btn-primary');
        btnToggleStream.classList.add('btn-run-active');
      }
    } else {
      if (statusBadge) {
        statusBadge.classList.remove('running');
        statusBadge.classList.add('standby');
      }
      setText('statusText', 'STANDBY');
      if (btnToggleStream) {
        btnToggleStream.textContent = 'START CONTINUOUS RUN';
        btnToggleStream.classList.remove('btn-run-active');
        btnToggleStream.classList.add('btn-primary');
      }
    }

    const facSelect = getFormControl('facilitySelect');
    if (facSelect) facSelect.value = String(state.facilityKey);

    const actSlider = getFormControl('activitySlider');
    if (actSlider && document.activeElement !== actSlider) {
      actSlider.value = String(state.sourceActivityKbq);
      setText('activityValue', `${state.sourceActivityKbq} kBq`);
    }

    const bkgSlider = getFormControl('backgroundSlider');
    if (bkgSlider && document.activeElement !== bkgSlider) {
      bkgSlider.value = String(state.ambientBackgroundCps);
      setText('backgroundValue', `${state.ambientBackgroundCps} cps`);
    }

    const runStateStr = state.batchRunning
      ? 'BATCH VALIDATION ACTIVE'
      : state.running
      ? 'CONTINUOUS 1Hz STREAM'
      : 'STANDBY (PERSISTED)';

    setText(
      'solverStatusInline',
      `STATE: ${runStateStr} | ENGINE: ${String(state.mcEngineName).toUpperCase()}`
    );

    const aiStat = state.aiModelStatus;
    const aiPill = document.getElementById('aiKnowledgePill');
    if (aiStat) {
      if (aiPill) {
        if (aiStat.is_trained) {
          aiPill.className = 'meta-pill conn-ok';
          aiPill.textContent = `AI STATE: TRAINED (${aiStat.trained_epochs} EP / ${aiStat.samples_seen} SAMPLES)`;
        } else {
          aiPill.className = 'meta-pill conn-err';
          aiPill.textContent = 'AI STATE: UNTRAINED (0 KNOWLEDGE)';
        }
      }

      const maeText =
        aiStat.is_trained && typeof aiStat.val_mae_meters === 'number'
          ? `${Number(aiStat.val_mae_meters).toFixed(3)} m`
          : 'N/A (UNTRAINED)';
      const lossText =
        aiStat.is_trained && typeof aiStat.last_loss === 'number'
          ? Number(aiStat.last_loss).toFixed(5)
          : 'N/A';
      const wn = aiStat.weight_norms || { W1_l2: 0, W2_l2: 0, W3_l2: 0 };

      setText(
        'aiModelStatusInline',
        `AI: ${aiStat.knowledge_state} | EPOCHS: ${aiStat.trained_epochs} | SAMPLES: ${aiStat.samples_seen} | MAE: ${maeText} | LOSS: ${lossText} | ||W||: [${wn.W1_l2}, ${wn.W2_l2}, ${wn.W3_l2}]`
      );
    }
  }

  // --- KPI & Table Renderers ---
  function updateKpiCards() {
    const tr = state.trueSource;
    const aiSol = state.aiSolution;
    const aiStat = state.aiModelStatus;

    setText('kpiTrueLoc', `${Number(tr.x).toFixed(2)}, ${Number(tr.y).toFixed(2)} m`);
    setText('kpiTrueZone', `ZONE: ${state.trueZone}`);

    if (!state.hasRun) {
      setText('kpiEstLoc', 'NO RUN YET');
      setText('kpiConfidence', 'STANDBY (0 RUNS)');
      setText('kpiDeltaError', '-- m');
      setText('kpiBatchStats', 'BATCH P50: -- m');
      setText('kpiActivityComp', `-- / ${Math.round(state.sourceActivityKbq)} kBq`);
      setText('kpiActivityErr', `BKG: ${state.ambientBackgroundCps} cps`);
    } else {
      setText('kpiEstLoc', `${Number(aiSol.x).toFixed(2)}, ${Number(aiSol.y).toFixed(2)} m`);
      setText(
        'kpiConfidence',
        aiSol.is_trained
          ? `TRAINED (${aiSol.trained_epochs} EPOCHS) | ${Number(aiSol.latency_ms).toFixed(2)}ms`
          : 'UNTRAINED WEIGHTS (0 EPOCHS)'
      );
      setText('kpiDeltaError', `${Number(aiSol.error_meters).toFixed(3)} m`);

      const errsAi = state.validationErrorsAi || [];
      if (errsAi.length > 0) {
        const p50 = Number(errsAi[Math.floor(errsAi.length * 0.5)] || 0);
        const p90 = Number(errsAi[Math.floor(errsAi.length * 0.9)] || 0);
        setText('kpiBatchStats', `BATCH P50: ${p50.toFixed(2)}m | P90: ${p90.toFixed(2)}m`);
      } else {
        setText('kpiBatchStats', `SINGLE RUN Δ: ${Number(aiSol.error_meters).toFixed(2)}m`);
      }

      const actErrPct =
        ((Number(aiSol.activity) - state.sourceActivityKbq) /
          Math.max(1, state.sourceActivityKbq)) *
        100;
      const sign = actErrPct >= 0 ? '+' : '';
      setText(
        'kpiActivityComp',
        `${Math.round(Number(aiSol.activity))} / ${Math.round(state.sourceActivityKbq)} kBq`
      );
      setText(
        'kpiActivityErr',
        `ΔA: ${sign}${actErrPct.toFixed(1)}% | BKG: ${state.ambientBackgroundCps} cps`
      );
    }

    if (aiStat) {
      setText(
        'kpiLatency',
        aiStat.is_trained ? `${aiStat.trained_epochs} EPOCHS` : '0 EPOCHS (RAW)'
      );
      setText('kpiIterations', `TRAIN SAMPLES: ${aiStat.samples_seen}`);
    }

    setText(
      'frameCounter',
      state.frameCount > 0
        ? `RUN #${String(state.frameCount).padStart(4, '0')} // MC ${(state.mcHistories / 1000).toFixed(0)}k HIST`
        : 'RUN #0000 // STANDBY'
    );
  }

  function updateSensorTable() {
    const tbody = document.getElementById('sensorTableBody');
    if (!tbody || !Array.isArray(state.sensorReadings) || state.sensorReadings.length === 0) {
      return;
    }

    let totalGross = 0;
    let peakSensor = state.sensorReadings[0];
    let rowsHtml = '';

    for (const r of state.sensorReadings) {
      totalGross += Number(r.raw);
      if (Number(r.raw) > Number(peakSensor.raw)) peakSensor = r;

      const badgeClass =
        r.status === 'ALARM'
          ? 'status-alarm'
          : r.status === 'ELEVATED'
          ? 'status-elevated'
          : 'status-nominal';
      const activeClass = state.hoveredSensorId === r.id ? 'active-row' : '';

      rowsHtml += `
        <tr data-sensor-id="${r.id}" class="${activeClass}">
          <td class="ch-tag">${r.id}</td>
          <td>(${Number(r.x).toFixed(1)}, ${Number(r.y).toFixed(1)})</td>
          <td class="num">${r.raw}</td>
          <td class="num">${Number(r.net).toFixed(0)}</td>
          <td class="num">${Number(r.modelFit).toFixed(1)}</td>
          <td class="num">${Number(r.distEst).toFixed(2)}</td>
          <td><span class="row-status ${badgeClass}">${r.status}</span></td>
        </tr>
      `;
    }

    tbody.innerHTML = rowsHtml;
    setText(
      'arrayTotalCounts',
      totalGross > 0
        ? `TOTAL GROSS: ${totalGross} cps | PEAK: ${peakSensor.id} (${peakSensor.raw} cps)`
        : 'STANDBY — 0 COUNTS RECORDED'
    );

    tbody.querySelectorAll('tr').forEach((tr) => {
      tr.addEventListener('mouseenter', () => {
        state.hoveredSensorId = tr.getAttribute('data-sensor-id') || '';
        renderFacilityCanvas();
      });
      tr.addEventListener('mouseleave', () => {
        state.hoveredSensorId = '';
        renderFacilityCanvas();
      });
    });
  }

  function updateTrainingDataTable() {
    const tbody = document.getElementById('trainingDataTableBody');
    if (!tbody) return;
    const samples = Array.isArray(state.trainingSamples) ? state.trainingSamples : [];

    if (samples.length === 0) {
      setText(
        'trainDataCountLabel',
        '(0 TRAINING SAMPLES — AI MODEL IS UNTRAINED)'
      );
      tbody.innerHTML = `
        <tr>
          <td colspan="12" style="text-align:center;color:#64748b;padding:14px;">
            NO TRAINING DATA YET — Click "TRAIN AI ON LOGGED RUNS" or "GENERATE &amp; TRAIN (64 MC SAMPLES)" to train the AI and inspect every sample here.
          </td>
        </tr>
      `;
      return;
    }

    setText(
      'trainDataCountLabel',
      `(${samples.length} TRAINING SAMPLES PERSISTED)`
    );
    let html = '';
    for (const s of samples.slice(0, 100)) {
      const countsStr = Array.isArray(s.raw_counts) ? s.raw_counts.join(', ') : '';
      const improved = Number(s.post_err_m) < Number(s.pre_err_m);
      html += `
        <tr>
          <td class="ch-tag">${s.sample_id}</td>
          <td>${s.timestamp_utc}</td>
          <td><span class="row-status status-nominal">${s.source_tag}</span></td>
          <td>${String(s.facility).toUpperCase()}</td>
          <td class="num">${s.epochs_applied}</td>
          <td style="color:#dc2626;font-weight:600;">(${Number(s.true_x).toFixed(2)}, ${Number(s.true_y).toFixed(2)})</td>
          <td class="num">${Number(s.true_activity_kbq).toFixed(0)}</td>
          <td class="num">${Number(s.background_cps).toFixed(0)}</td>
          <td style="font-size:10px;color:#334155;">[${countsStr}]</td>
          <td class="num" style="color:#b91c1c;">${Number(s.pre_err_m).toFixed(3)} m</td>
          <td style="color:#0284c7;font-weight:600;">(${Number(s.post_pred_x).toFixed(2)}, ${Number(s.post_pred_y).toFixed(2)})</td>
          <td class="num" style="font-weight:700;color:${improved ? '#15803d' : '#0f172a'};">${Number(s.post_err_m).toFixed(3)} m</td>
        </tr>
      `;
    }
    tbody.innerHTML = html;
  }

  function updateRunLogTable() {
    const tbody = document.getElementById('runLogTableBody');
    if (!tbody) return;
    setText('logCountLabel', `(${state.runLogs.length} RUNS PERSISTED)`);

    if (state.runLogs.length === 0) {
      tbody.innerHTML = `
        <tr>
          <td colspan="12" style="text-align:center;color:#64748b;padding:14px;">
            0 RUNS LOGGED. Click "STEP 1 MC FRAME", "SIMULATE NEW EVENT", or "START CONTINUOUS RUN" to record OpenMC runs.
          </td>
        </tr>
      `;
      return;
    }

    let html = '';
    const slice = state.runLogs.slice(0, 80);
    for (const row of slice) {
      const lowError = Number(row.ai_error_m) < 1.0;
      const aiStateBadge = row.ai_is_trained
        ? `<span class="row-status status-nominal">TRAINED (${row.ai_epochs} EP)</span>`
        : `<span class="row-status status-alarm">UNTRAINED (0 EP)</span>`;
      const countsStr = Array.isArray(row.raw_counts) ? row.raw_counts.join(', ') : '';

      html += `
        <tr>
          <td class="ch-tag">${row.run_id}</td>
          <td>${row.timestamp_utc}</td>
          <td><span class="row-status status-nominal">${row.mode}</span></td>
          <td>${String(row.facility).toUpperCase()}</td>
          <td>${aiStateBadge}</td>
          <td style="color:#dc2626;font-weight:600;">(${Number(row.true_x).toFixed(2)}, ${Number(row.true_y).toFixed(2)})</td>
          <td class="num">${Number(row.true_activity_kbq).toFixed(0)}</td>
          <td style="color:#0284c7;font-weight:600;">(${Number(row.ai_x).toFixed(2)}, ${Number(row.ai_y).toFixed(2)})</td>
          <td class="num">${Number(row.ai_activity_kbq).toFixed(0)}</td>
          <td class="num" style="font-weight:700;color:${lowError ? '#15803d' : '#b91c1c'};">${Number(row.ai_error_m).toFixed(3)} m</td>
          <td class="num">${Number(row.ai_latency_ms).toFixed(2)} ms</td>
          <td style="font-size:10px;color:#334155;">[${countsStr}]</td>
        </tr>
      `;
    }
    tbody.innerHTML = html;
  }

  // --- 2D Facility Canvas Renderer ---
  const canvasEl = document.getElementById('facilityCanvas');
  const canvas = canvasEl instanceof HTMLCanvasElement ? canvasEl : null;
  const ctx = canvas ? canvas.getContext('2d') : null;

  /**
   * @param {number} wx
   * @param {number} wy
   * @param {number} width
   * @param {number} height
   */
  function worldToCanvas(wx, wy, width, height) {
    const plotW = width - PAD * 2;
    const plotH = height - PAD * 2;
    return {
      cx: PAD + (wx / DOMAIN_SIZE) * plotW,
      cy: PAD + (wy / DOMAIN_SIZE) * plotH
    };
  }

  /**
   * @param {number} cx
   * @param {number} cy
   * @param {number} width
   * @param {number} height
   */
  function canvasToWorld(cx, cy, width, height) {
    const plotW = width - PAD * 2;
    const plotH = height - PAD * 2;
    const wx = ((cx - PAD) / plotW) * DOMAIN_SIZE;
    const wy = ((cy - PAD) / plotH) * DOMAIN_SIZE;
    return {
      wx: Math.max(0.2, Math.min(DOMAIN_SIZE - 0.2, wx)),
      wy: Math.max(0.2, Math.min(DOMAIN_SIZE - 0.2, wy))
    };
  }

  function renderFacilityCanvas() {
    const facility = state.facilities[state.facilityKey];
    if (!facility || !canvas || !ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const W = CANVAS_SIZE;
    const H = CANVAS_SIZE;
    if (canvas.width !== W * dpr || canvas.height !== H * dpr) {
      canvas.width = W * dpr;
      canvas.height = H * dpr;
    }

    ctx.save();
    ctx.scale(dpr, dpr);

    const plotW = W - PAD * 2;
    const plotH = H - PAD * 2;
    const pxPerMeter = plotW / DOMAIN_SIZE;

    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, W, H);

    ctx.fillStyle = '#fcfcfd';
    ctx.fillRect(PAD, PAD, plotW, plotH);

    ctx.lineWidth = 1;
    for (let m = 0; m <= DOMAIN_SIZE; m += 0.5) {
      const pos = PAD + m * pxPerMeter;
      const isMajor = Math.abs(m % 2.0) < 1e-6;
      ctx.strokeStyle = isMajor ? '#e2e8f0' : '#f1f5f9';
      ctx.beginPath();
      ctx.moveTo(pos, PAD);
      ctx.lineTo(pos, PAD + plotH);
      ctx.moveTo(PAD, pos);
      ctx.lineTo(PAD + plotW, pos);
      ctx.stroke();
    }

    ctx.font = '600 10px "JetBrains Mono", monospace';
    ctx.fillStyle = '#94a3b8';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    for (const z of facility.zones) {
      const tl = worldToCanvas(Number(z.x), Number(z.y), W, H);
      ctx.fillText(String(z.label), tl.cx + 8, tl.cy + 7);
    }

    ctx.strokeStyle = '#94a3b8';
    ctx.lineWidth = 2.5;
    for (const w of facility.walls) {
      const p1 = worldToCanvas(Number(w.x1), Number(w.y1), W, H);
      const p2 = worldToCanvas(Number(w.x2), Number(w.y2), W, H);
      ctx.beginPath();
      ctx.moveTo(p1.cx, p1.cy);
      ctx.lineTo(p2.cx, p2.cy);
      ctx.stroke();
    }

    ctx.strokeStyle = '#334155';
    ctx.lineWidth = 1.5;
    ctx.strokeRect(PAD, PAD, plotW, plotH);

    ctx.fillStyle = '#64748b';
    ctx.font = '500 10px "JetBrains Mono", monospace';
    for (let m = 0; m <= DOMAIN_SIZE; m += 2) {
      const pos = PAD + m * pxPerMeter;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'bottom';
      ctx.fillText(`${m}m`, pos, PAD - 6);
      ctx.textBaseline = 'top';
      ctx.fillText(`${m}m`, pos, PAD + plotH + 6);
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      ctx.fillText(`${m}m`, PAD - 6, pos);
    }

    const aiSol = state.aiSolution;
    const truePt = worldToCanvas(
      Number(state.trueSource.x),
      Number(state.trueSource.y),
      W,
      H
    );

    ctx.save();
    ctx.beginPath();
    ctx.rect(PAD, PAD, plotW, plotH);
    ctx.clip();

    const rawValues = state.sensorReadings.map(
      /** @param {Record<string, any>} s */ (s) => Number(s.raw)
    );
    const maxRaw = Math.max(...rawValues, state.ambientBackgroundCps + 10);

    for (const s of state.sensorReadings) {
      const pt = worldToCanvas(Number(s.x), Number(s.y), W, H);
      const rawVal = Number(s.raw);
      const netRatio =
        rawVal > 0
          ? Math.max(
              0,
              (rawVal - state.ambientBackgroundCps) /
                Math.max(1, maxRaw - state.ambientBackgroundCps)
            )
          : 0;
      const bubbleRadius = rawVal > 0 ? 10 + Math.sqrt(netRatio) * 32 : 8;
      const isAlarm = s.status === 'ALARM';
      const isElevated = s.status === 'ELEVATED';

      ctx.beginPath();
      ctx.arc(pt.cx, pt.cy, bubbleRadius, 0, Math.PI * 2);
      if (isAlarm) {
        ctx.fillStyle = 'rgba(220, 38, 38, 0.11)';
        ctx.strokeStyle = 'rgba(220, 38, 38, 0.40)';
      } else if (isElevated) {
        ctx.fillStyle = 'rgba(217, 119, 6, 0.11)';
        ctx.strokeStyle = 'rgba(217, 119, 6, 0.40)';
      } else {
        ctx.fillStyle = 'rgba(100, 116, 139, 0.05)';
        ctx.strokeStyle = 'rgba(100, 116, 139, 0.20)';
      }
      ctx.lineWidth = 1;
      ctx.fill();
      ctx.stroke();

      if (state.hoveredSensorId === s.id) {
        ctx.beginPath();
        ctx.arc(pt.cx, pt.cy, bubbleRadius + 4, 0, Math.PI * 2);
        ctx.strokeStyle = '#0f172a';
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }

      ctx.beginPath();
      ctx.arc(pt.cx, pt.cy, 5, 0, Math.PI * 2);
      ctx.fillStyle = '#ffffff';
      ctx.fill();
      ctx.lineWidth = 1.75;
      ctx.strokeStyle = isAlarm ? '#dc2626' : isElevated ? '#d97706' : '#334155';
      ctx.stroke();

      ctx.beginPath();
      ctx.arc(pt.cx, pt.cy, 1.8, 0, Math.PI * 2);
      ctx.fillStyle = isAlarm ? '#dc2626' : isElevated ? '#d97706' : '#334155';
      ctx.fill();

      ctx.font = '700 10px "JetBrains Mono", monospace';
      ctx.fillStyle = '#0f172a';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'bottom';
      ctx.fillText(String(s.id), pt.cx, pt.cy - 7);

      ctx.font = '500 9.5px "JetBrains Mono", monospace';
      ctx.fillStyle = isAlarm ? '#b91c1c' : '#475569';
      ctx.textBaseline = 'top';
      ctx.fillText(`${rawVal} cps`, pt.cx, pt.cy + 7);
    }

    if (state.hasRun) {
      const aiPt = worldToCanvas(Number(aiSol.x), Number(aiSol.y), W, H);

      // Distance Delta Vector from True Source to AI Estimate
      ctx.beginPath();
      ctx.setLineDash([4, 3]);
      ctx.moveTo(truePt.cx, truePt.cy);
      ctx.lineTo(aiPt.cx, aiPt.cy);
      ctx.strokeStyle = '#dc2626';
      ctx.lineWidth = 1.5;
      ctx.stroke();
      ctx.setLineDash([]);

      // AI Crosshair
      const chSize = 11;
      ctx.strokeStyle = '#0284c7';
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      ctx.moveTo(aiPt.cx - chSize, aiPt.cy);
      ctx.lineTo(aiPt.cx + chSize, aiPt.cy);
      ctx.moveTo(aiPt.cx, aiPt.cy - chSize);
      ctx.lineTo(aiPt.cx, aiPt.cy + chSize);
      ctx.stroke();

      ctx.beginPath();
      ctx.arc(aiPt.cx, aiPt.cy, 4.5, 0, Math.PI * 2);
      ctx.strokeStyle = '#0284c7';
      ctx.lineWidth = 1.5;
      ctx.stroke();

      const tag = aiSol.is_trained ? 'AI-TRAINED' : 'AI-UNTRAINED';
      drawCanvasCallout(
        ctx,
        aiPt.cx + 10,
        aiPt.cy + 8,
        `${tag} (${Number(aiSol.x).toFixed(2)}, ${Number(aiSol.y).toFixed(2)}) | Δ=${Number(aiSol.error_meters).toFixed(2)}m`,
        '#0284c7'
      );
    }

    ctx.save();
    ctx.translate(truePt.cx, truePt.cy);
    ctx.rotate(Math.PI / 4);
    ctx.fillStyle = '#dc2626';
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.rect(-5.5, -5.5, 11, 11);
    ctx.fill();
    ctx.stroke();
    ctx.restore();

    drawCanvasCallout(
      ctx,
      truePt.cx + 10,
      truePt.cy - 18,
      `TRUE (${Number(state.trueSource.x).toFixed(2)}, ${Number(state.trueSource.y).toFixed(2)})`,
      '#dc2626'
    );

    ctx.restore();
    ctx.restore();
  }

  /**
   * @param {CanvasRenderingContext2D} context
   * @param {number} x
   * @param {number} y
   * @param {string} text
   * @param {string} borderColor
   */
  function drawCanvasCallout(context, x, y, text, borderColor) {
    context.save();
    context.font = '600 10px "JetBrains Mono", monospace';
    const padX = 5;
    const boxH = 16;
    const textW = context.measureText(text).width;
    const boxW = textW + padX * 2;

    const clampedX = Math.max(PAD + 4, Math.min(CANVAS_SIZE - PAD - boxW - 4, x));
    const clampedY = Math.max(PAD + 4, Math.min(CANVAS_SIZE - PAD - boxH - 4, y));

    context.fillStyle = 'rgba(255, 255, 255, 0.94)';
    context.strokeStyle = borderColor;
    context.lineWidth = 1;
    context.beginPath();
    context.rect(clampedX, clampedY, boxW, boxH);
    context.fill();
    context.stroke();

    context.fillStyle = '#0f172a';
    context.textAlign = 'left';
    context.textBaseline = 'middle';
    context.fillText(text, clampedX + padX, clampedY + boxH / 2 + 0.5);
    context.restore();
  }

  // --- Charts (Chart.js) ---
  /** @type {Record<string, any>} */
  const chartStore = {
    cdf: null,
    flux: null
  };

  function initCharts() {
    const ChartLib = /** @type {any} */ (window)['Chart'];
    if (!ChartLib) return;

    // White background plugin so exported chart PNGs are never transparent
    const whiteBgPlugin = {
      id: 'customCanvasBackgroundColor',
      /** @param {any} chart */
      beforeDraw: (chart) => {
        const g = chart.canvas.getContext('2d');
        g.save();
        g.globalCompositeOperation = 'destination-over';
        g.fillStyle = '#ffffff';
        g.fillRect(0, 0, chart.width, chart.height);
        g.restore();
      }
    };

    ChartLib.defaults.font.family = '"JetBrains Mono", "IBM Plex Mono", monospace';
    ChartLib.defaults.font.size = 10;
    ChartLib.defaults.color = '#475569';

    const cdfCanvas = document.getElementById('cdfChart');
    if (cdfCanvas instanceof HTMLCanvasElement) {
      const cdfCtx = cdfCanvas.getContext('2d');
      chartStore.cdf = new ChartLib(cdfCtx, {
        type: 'line',
        plugins: [whiteBgPlugin],
        data: {
          labels: [],
          datasets: [
            {
              label: 'AI Neural Network Localization CDF (%)',
              data: [],
              borderColor: '#0284c7',
              backgroundColor: 'rgba(2, 132, 199, 0.10)',
              borderWidth: 2,
              pointRadius: 0,
              fill: true,
              tension: 0.15
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          animation: { duration: 120 },
          plugins: {
            legend: {
              display: true,
              position: 'top',
              align: 'end',
              labels: { boxWidth: 10, boxHeight: 10, padding: 10, font: { size: 10 } }
            }
          },
          scales: {
            x: {
              title: {
                display: true,
                text: 'AI Localization Error Δ (meters)',
                font: { size: 10, weight: '600' },
                color: '#475569'
              },
              grid: { color: '#f1f5f9' },
              ticks: { maxTicksLimit: 8 }
            },
            y: {
              min: 0,
              max: 100,
              title: {
                display: true,
                text: 'Cumulative %',
                font: { size: 10, weight: '600' },
                color: '#475569'
              },
              grid: { color: '#f1f5f9' },
              ticks: { stepSize: 25 }
            }
          }
        }
      });
    }

    const fluxCanvas = document.getElementById('fluxChart');
    if (fluxCanvas instanceof HTMLCanvasElement) {
      const fluxCtx = fluxCanvas.getContext('2d');
      chartStore.flux = new ChartLib(fluxCtx, {
        type: 'bar',
        plugins: [whiteBgPlugin],
        data: {
          labels: [],
          datasets: [
            {
              label: 'OpenMC Tally Counts (cps)',
              data: [],
              backgroundColor: '#334155',
              borderColor: '#0f172a',
              borderWidth: 1,
              borderRadius: 1
            },
            {
              label: 'AI Predicted Flux E[λ] (cps)',
              data: [],
              backgroundColor: 'rgba(2, 132, 199, 0.28)',
              borderColor: '#0284c7',
              borderWidth: 1,
              borderRadius: 1
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          animation: { duration: 100 },
          plugins: {
            legend: {
              display: true,
              position: 'top',
              align: 'end',
              labels: { boxWidth: 10, boxHeight: 10, padding: 10, font: { size: 10 } }
            }
          },
          scales: {
            x: { grid: { display: false } },
            y: {
              beginAtZero: true,
              title: {
                display: true,
                text: 'Count Rate (cps)',
                font: { size: 10, weight: '600' },
                color: '#475569'
              },
              grid: { color: '#f1f5f9' }
            }
          }
        }
      });
    }
  }

  function updateCdfChart() {
    const errsAi = Array.isArray(state.validationErrorsAi) ? state.validationErrorsAi : [];

    if (errsAi.length === 0) {
      setText(
        'cdfSummaryLabel',
        '0 VALIDATION RUNS — CLICK "RUN 100-RUN VALIDATION"'
      );
      if (chartStore.cdf) {
        chartStore.cdf.data.labels = [];
        chartStore.cdf.data.datasets[0].data = [];
        chartStore.cdf.update();
      }
      return;
    }

    const aiP50 = Number(errsAi[Math.floor(errsAi.length * 0.5)] || 0);
    const aiP90 = Number(errsAi[Math.floor(errsAi.length * 0.9)] || 0);
    const maxErrVal = Number(errsAi[errsAi.length - 1] || 0.5);

    setText(
      'cdfSummaryLabel',
      `N=${errsAi.length} | AI P50: ${aiP50.toFixed(2)}m | P90: ${aiP90.toFixed(2)}m | MAX: ${maxErrVal.toFixed(2)}m`
    );

    const maxErr = Math.max(0.3, maxErrVal);
    const upper = maxErr * 1.05;
    const nBins = 25;
    const labels = [];
    const cdfAi = [];

    for (let i = 0; i <= nBins; i++) {
      const thresh = (i / nBins) * upper;
      labels.push(thresh.toFixed(2));
      const cAi = errsAi.filter(/** @param {number} e */ (e) => e <= thresh).length;
      cdfAi.push(Number(((cAi / errsAi.length) * 100).toFixed(1)));
    }

    if (chartStore.cdf) {
      chartStore.cdf.data.labels = labels;
      chartStore.cdf.data.datasets[0].data = cdfAi;
      chartStore.cdf.update();
    }
  }

  function updateFluxChart() {
    if (!chartStore.flux || !Array.isArray(state.sensorReadings)) return;
    chartStore.flux.data.labels = state.sensorReadings.map(
      /** @param {Record<string, any>} r */ (r) => r.id
    );
    chartStore.flux.data.datasets[0].data = state.sensorReadings.map(
      /** @param {Record<string, any>} r */ (r) => r.raw
    );
    chartStore.flux.data.datasets[1].data = state.sensorReadings.map(
      /** @param {Record<string, any>} r */ (r) => r.modelFit
    );
    chartStore.flux.update();
  }

  // --- Snapshot Screenshot Capture (Map + Graphs + Composite Overview) ---
  async function captureSnapshotBundle() {
    const cdfCanvas = document.getElementById('cdfChart');
    const fluxCanvas = document.getElementById('fluxChart');
    if (!canvas) return;

    const mapDataUrl = canvas.toDataURL('image/png');
    const cdfDataUrl =
      cdfCanvas instanceof HTMLCanvasElement ? cdfCanvas.toDataURL('image/png') : '';
    const fluxDataUrl =
      fluxCanvas instanceof HTMLCanvasElement ? fluxCanvas.toDataURL('image/png') : '';

    // Build a high-res composite board (1320 x 740) containing Header KPIs + Map + Both Charts
    const compCanvas = document.createElement('canvas');
    compCanvas.width = 1320;
    compCanvas.height = 740;
    const g = compCanvas.getContext('2d');
    let compositeDataUrl = '';

    if (g) {
      g.fillStyle = '#f8f9fa';
      g.fillRect(0, 0, compCanvas.width, compCanvas.height);

      // Header Banner
      g.fillStyle = '#ffffff';
      g.fillRect(16, 14, 1288, 56);
      g.strokeStyle = '#cbd5e1';
      g.strokeRect(16, 14, 1288, 56);

      g.fillStyle = '#0f172a';
      g.font = '700 14px "JetBrains Mono", monospace';
      const nowIso = new Date().toISOString().replace('T', ' ').slice(0, 19) + ' UTC';
      g.fillText(
        `RAD-MON-04 // TELEMETRY SNAPSHOT | ${state.facilityKey.toUpperCase()} | RUN #${String(state.frameCount).padStart(4, '0')} | ${nowIso}`,
        30,
        36
      );

      const aiSol = state.aiSolution;
      const aiStat = state.aiModelStatus;
      g.fillStyle = '#334155';
      g.font = '500 12px "JetBrains Mono", monospace';
      const subSummary = state.hasRun
        ? `TRUE: (${Number(state.trueSource.x).toFixed(2)}, ${Number(state.trueSource.y).toFixed(2)})m, ${state.sourceActivityKbq}kBq | AI EST: (${Number(aiSol.x).toFixed(2)}, ${Number(aiSol.y).toFixed(2)})m, ${Math.round(Number(aiSol.activity))}kBq | Δ ERR: ${Number(aiSol.error_meters).toFixed(3)}m | AI EPOCHS: ${aiStat.trained_epochs} (${aiStat.samples_seen} samples)`
        : `STATE: STANDBY | TRUE: (${Number(state.trueSource.x).toFixed(2)}, ${Number(state.trueSource.y).toFixed(2)})m | AI EPOCHS: ${aiStat.trained_epochs}`;
      g.fillText(subSummary, 30, 56);

      // Left Box: 2D Facility Map (640x640)
      g.fillStyle = '#ffffff';
      g.fillRect(16, 84, 640, 640);
      g.strokeStyle = '#cbd5e1';
      g.strokeRect(16, 84, 640, 640);
      g.drawImage(canvas, 20, 88, 632, 632);

      // Right Top Box: CDF Error Distribution Chart
      g.fillStyle = '#ffffff';
      g.fillRect(670, 84, 634, 312);
      g.strokeStyle = '#cbd5e1';
      g.strokeRect(670, 84, 634, 312);
      g.fillStyle = '#334155';
      g.font = '700 11px "JetBrains Mono", monospace';
      g.fillText('CDF ERROR DISTRIBUTION (AI NEURAL NETWORK)', 684, 104);
      if (cdfCanvas instanceof HTMLCanvasElement) {
        g.drawImage(cdfCanvas, 680, 114, 614, 272);
      }

      // Right Bottom Box: Sensor Flux Profile Chart
      g.fillStyle = '#ffffff';
      g.fillRect(670, 412, 634, 312);
      g.strokeStyle = '#cbd5e1';
      g.strokeRect(670, 412, 634, 312);
      g.fillStyle = '#334155';
      g.font = '700 11px "JetBrains Mono", monospace';
      g.fillText('SENSOR FLUX PROFILE (16-CH RECORDED vs. AI PREDICTION)', 684, 432);
      if (fluxCanvas instanceof HTMLCanvasElement) {
        g.drawImage(fluxCanvas, 680, 442, 614, 272);
      }

      compositeDataUrl = compCanvas.toDataURL('image/png');
    }

    const btn1 = getButton('btnCaptureSnapshot');
    const btn2 = getButton('btnCaptureHeader');
    if (btn1) btn1.textContent = 'SAVING SNAPSHOT...';
    if (btn2) btn2.textContent = 'SAVING SNAPSHOT...';

    const res = await apiPost('/api/capture-snapshot', {
      map_png: mapDataUrl,
      cdf_png: cdfDataUrl,
      flux_png: fluxDataUrl,
      composite_png: compositeDataUrl
    });

    if (res && res.ok && res.capture) {
      const folderPath = res.capture.folder_path;
      setText('lastSnapshotBanner', `SAVED: ${folderPath}/`);
      if (btn1) btn1.textContent = `SAVED TO ${folderPath}`;
      if (btn2) btn2.textContent = 'SNAPSHOT SAVED!';
      window.setTimeout(() => {
        if (btn1) btn1.textContent = 'SCREENSHOT MAP & GRAPHS';
        if (btn2) btn2.textContent = 'CAPTURE SNAPSHOT (MAP + CHARTS)';
      }, 2500);
    } else {
      if (btn1) btn1.textContent = 'SCREENSHOT MAP & GRAPHS';
      if (btn2) btn2.textContent = 'CAPTURE SNAPSHOT (MAP + CHARTS)';
    }
  }

  // --- Interactive Canvas Source Relocation ---
  /**
   * @param {MouseEvent} evt
   */
  function getCanvasRelativeCoords(evt) {
    if (!canvas) return { cx: 0, cy: 0 };
    const rect = canvas.getBoundingClientRect();
    const scaleX = CANVAS_SIZE / rect.width;
    const scaleY = CANVAS_SIZE / rect.height;
    return {
      cx: (evt.clientX - rect.left) * scaleX,
      cy: (evt.clientY - rect.top) * scaleY
    };
  }

  let dragThrottleTimer = 0;
  /**
   * @param {number} wx
   * @param {number} wy
   * @param {string} modeLabel
   */
  function dispatchSourcePositionToBackend(wx, wy, modeLabel) {
    state.trueSource.x = Number(wx.toFixed(2));
    state.trueSource.y = Number(wy.toFixed(2));
    renderFacilityCanvas();

    if (dragThrottleTimer !== 0) return;
    dragThrottleTimer = window.setTimeout(() => {
      dragThrottleTimer = 0;
      apiPost('/api/control', {
        true_source: { x: state.trueSource.x, y: state.trueSource.y },
        trigger_run: true,
        mode: modeLabel
      });
    }, 65);
  }

  if (canvas) {
    canvas.addEventListener('mousedown', (evt) => {
      const { cx, cy } = getCanvasRelativeCoords(evt);
      const { wx, wy } = canvasToWorld(cx, cy, CANVAS_SIZE, CANVAS_SIZE);
      state.draggingSource = true;
      dispatchSourcePositionToBackend(wx, wy, 'CANVAS-RELOC');
    });
  }

  window.addEventListener('mousemove', (evt) => {
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    if (
      evt.clientX >= rect.left &&
      evt.clientX <= rect.right &&
      evt.clientY >= rect.top &&
      evt.clientY <= rect.bottom
    ) {
      const { cx, cy } = getCanvasRelativeCoords(evt);
      const { wx, wy } = canvasToWorld(cx, cy, CANVAS_SIZE, CANVAS_SIZE);
      setText(
        'canvasCoordReadout',
        `CURSOR: ${wx.toFixed(2)}, ${wy.toFixed(2)} m | DOMAIN: 20.0m × 20.0m`
      );

      if (state.draggingSource) {
        dispatchSourcePositionToBackend(wx, wy, 'CANVAS-DRAG');
      }
    }
  });

  window.addEventListener('mouseup', () => {
    state.draggingSource = false;
  });

  // --- Control Center Event Listeners ---
  function toggleRunningState() {
    apiPost('/api/control', { running: !state.running });
  }

  const statusBadgeEl = document.getElementById('statusBadge');
  if (statusBadgeEl) statusBadgeEl.addEventListener('click', toggleRunningState);

  const btnToggleEl = getButton('btnToggleStream');
  if (btnToggleEl) btnToggleEl.addEventListener('click', toggleRunningState);

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
  if (btnCap1) btnCap1.addEventListener('click', captureSnapshotBundle);

  const btnCap2 = getButton('btnCaptureHeader');
  if (btnCap2) btnCap2.addEventListener('click', captureSnapshotBundle);

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

  // --- UTC Clock ---
  function updateUtcClock() {
    const iso = new Date().toISOString();
    setText('utcClock', iso.slice(0, 10) + ' ' + iso.slice(11, 21) + ' UTC');
  }
  window.setInterval(updateUtcClock, 200);
  updateUtcClock();

  // --- Bootstrap State & Connect SSE Stream ---
  /**
   * @param {Record<string, any>} full
   */
  function applyFullState(full) {
    state.facilities = full.facilities || state.facilities;
    state.running = Boolean(full.running);
    state.facilityKey = full.facility_key || state.facilityKey;
    state.aiModelStatus = full.ai_model_status || state.aiModelStatus;
    state.validationErrorsAi = full.validation_errors_ai || [];
    state.runLogs = full.recent_logs || [];
    state.trainingSamples = full.training_samples || [];

    if (Array.isArray(full.captures_history) && full.captures_history.length > 0) {
      setText('lastSnapshotBanner', `LAST CAPTURE: ${full.captures_history[0].folder_path}/`);
    }

    applyFramePayload(full.latest_frame);
    updateTrainingDataTable();
    updateCdfChart();
  }

  async function connectBackend() {
    try {
      const res = await fetch('/api/state');
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const full = await res.json();

      setConnectionStatus(true, 'BACKEND: OPENMC STREAM ONLINE');
      applyFullState(full);

      const evtSource = new EventSource('/api/stream');

      evtSource.addEventListener('frame', (e) => {
        const frame = JSON.parse(e.data);
        applyFramePayload(frame);
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
        syncControlsUI();
        updateKpiCards();
        renderFacilityCanvas();
      });

      evtSource.addEventListener('batch_progress', (e) => {
        const prog = JSON.parse(e.data);
        state.batchRunning = true;
        state.validationErrorsAi = prog.validation_errors_ai || state.validationErrorsAi;
        setText('btnRunValidation', `RUNNING BATCH (${prog.completed}/${prog.total})...`);
        updateCdfChart();
      });

      evtSource.addEventListener('batch_complete', (e) => {
        const done = JSON.parse(e.data);
        state.batchRunning = false;
        state.validationErrorsAi = done.validation_errors_ai || state.validationErrorsAi;
        setText('btnRunValidation', 'RUN 100-RUN VALIDATION');
        syncControlsUI();
        updateCdfChart();
      });

      evtSource.addEventListener('ai_training_progress', (e) => {
        const prog = JSON.parse(e.data);
        state.trainingRunning = true;
        if (prog.ai_model_status) state.aiModelStatus = prog.ai_model_status;
        syncControlsUI();
      });

      evtSource.addEventListener('ai_training_complete', (e) => {
        const done = JSON.parse(e.data);
        state.trainingRunning = false;
        if (done.ai_model_status) state.aiModelStatus = done.ai_model_status;
        if (done.training_samples) state.trainingSamples = done.training_samples;
        if (btnTrainOnLogs) btnTrainOnLogs.disabled = false;
        if (btnTrainOpenMC) btnTrainOpenMC.disabled = false;
        syncControlsUI();
        updateKpiCards();
        updateTrainingDataTable();
      });

      evtSource.addEventListener('snapshot_saved', (e) => {
        const cap = JSON.parse(e.data);
        if (cap && cap.folder_path) {
          setText('lastSnapshotBanner', `SAVED: ${cap.folder_path}/`);
        }
      });

      evtSource.addEventListener('ai_scrubbed', (e) => {
        const d = JSON.parse(e.data);
        if (d.ai_model_status) state.aiModelStatus = d.ai_model_status;
        state.trainingSamples = [];
        state.validationErrorsAi = [];
        syncControlsUI();
        updateKpiCards();
        updateTrainingDataTable();
        updateCdfChart();
      });

      evtSource.addEventListener('all_scrubbed', (e) => {
        const fullState = JSON.parse(e.data);
        applyFullState(fullState);
      });

      evtSource.addEventListener('logs_cleared', () => {
        state.runLogs = [];
        state.frameCount = 0;
        state.hasRun = false;
        updateRunLogTable();
        updateKpiCards();
        renderFacilityCanvas();
      });

      evtSource.onerror = () => {
        setConnectionStatus(false, 'BACKEND: STREAM RECONNECTING...');
      };
    } catch (err) {
      console.error('Failed to connect to backend:', err);
      setConnectionStatus(false, 'BACKEND: OFFLINE (RUN python3 server.py)');
    }
  }

  initCharts();
  connectBackend();
})();
