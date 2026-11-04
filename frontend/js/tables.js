/**
 * frontend/js/tables.js
 * Renders the KPI cards, 16-Channel Sensor Array table (including Concrete Wall Shielding),
 * AI Training Dataset & Reward Inspector table, OpenMC Run Log table, and Snapshot Archive.
 */

(function () {
  'use strict';

  /** @type {Record<string, any>} */
  const RadMon = /** @type {any} */ (window).RadMon;
  const { state, setText, getFormControl, getButton } = RadMon;

  RadMon.syncControlsUI = function () {
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
      `STATE: ${runStateStr} | ENGINE: ${String(state.mcEngineName).toUpperCase()} (22cm CONCRETE WALL SHIELDING)`
    );

    const sectorUpper = String(state.facilityKey).toUpperCase();
    const aiStat = state.aiModelStatus;
    const aiPill = document.getElementById('aiKnowledgePill');
    if (aiStat) {
      if (aiPill) {
        if (aiStat.is_trained) {
          aiPill.className = 'meta-pill conn-ok';
          aiPill.textContent = `AI [${sectorUpper}]: TRAINED (${aiStat.trained_epochs} EP / ${aiStat.samples_seen} SAMPLES)`;
        } else {
          aiPill.className = 'meta-pill conn-err';
          aiPill.textContent = `AI [${sectorUpper}]: UNTRAINED (0 KNOWLEDGE)`;
        }
      }

      setText(
        'aiSectorBoundBanner',
        `SECTOR-ISOLATED AI MODEL // BOUND TO ${sectorUpper} ENVIRONMENT ONLY (NOT CROSS-SECTOR)`
      );

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
        `ENV: ${sectorUpper} ONLY | STATE: ${aiStat.knowledge_state} | EPOCHS: ${aiStat.trained_epochs} | SAMPLES: ${aiStat.samples_seen} | MAE: ${maeText} | LOSS: ${lossText} | ||W||: [${wn.W1_l2}, ${wn.W2_l2}, ${wn.W3_l2}]`
      );
    }

    if (typeof RadMon.updateTabBadges === 'function') {
      RadMon.updateTabBadges();
    }
  };

  RadMon.updateKpiCards = function () {
    const tr = state.trueSource;
    const aiSol = state.aiSolution;
    const aiStat = state.aiModelStatus;
    const sectorUpper = String(state.facilityKey).toUpperCase();

    setText('kpiTrueLoc', `${Number(tr.x).toFixed(2)}, ${Number(tr.y).toFixed(2)} m`);
    setText('kpiTrueZone', `ZONE: ${state.trueZone}`);

    if (!state.hasRun) {
      setText('kpiEstLoc', 'NO RUN YET');
      setText('kpiConfidence', `STANDBY (${sectorUpper})`);
      setText('kpiDeltaError', '-- m');
      setText('kpiBatchStats', 'BATCH P50: -- m');
      setText('kpiActivityComp', `-- / ${Math.round(state.sourceActivityKbq)} kBq`);
      setText('kpiActivityErr', `BKG: ${state.ambientBackgroundCps} cps`);
    } else {
      setText('kpiEstLoc', `${Number(aiSol.x).toFixed(2)}, ${Number(aiSol.y).toFixed(2)} m`);
      setText(
        'kpiConfidence',
        aiSol.is_trained
          ? `TRAINED [${sectorUpper}] (${aiSol.trained_epochs} EP) | ${Number(aiSol.latency_ms).toFixed(2)}ms`
          : `UNTRAINED [${sectorUpper}] (0 EPOCHS)`
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
      setText('kpiIterations', `${sectorUpper} SAMPLES: ${aiStat.samples_seen}`);
    }

    setText(
      'frameCounter',
      state.frameCount > 0
        ? `RUN #${String(state.frameCount).padStart(4, '0')} // MC ${(state.mcHistories / 1000).toFixed(0)}k HIST`
        : 'RUN #0000 // STANDBY'
    );
  };

  RadMon.updateSensorTable = function () {
    const tbody = document.getElementById('sensorTableBody');
    if (!tbody || !Array.isArray(state.sensorReadings) || state.sensorReadings.length === 0) {
      return;
    }

    let totalGross = 0;
    let shieldedCount = 0;
    let peakSensor = state.sensorReadings[0];
    let rowsHtml = '';

    for (const r of state.sensorReadings) {
      totalGross += Number(r.raw);
      if (Number(r.raw) > Number(peakSensor.raw)) peakSensor = r;

      const nWalls = Number(r.wallCrossings || 0);
      if (nWalls > 0) shieldedCount++;
      const wallTrans =
        typeof r.wallTransmission === 'number' ? Number(r.wallTransmission) : 1.0;
      const transPct = Math.round(wallTrans * 100);

      const shieldBadge =
        nWalls === 0
          ? `<span class="row-status status-rewarded">LOS (100%)</span>`
          : `<span class="row-status status-elevated">${nWalls} WALL${nWalls > 1 ? 'S' : ''} (${transPct}%)</span>`;

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
          <td>${shieldBadge}</td>
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
        ? `TOTAL: ${totalGross} cps | PEAK: ${peakSensor.id} (${peakSensor.raw} cps) | SHIELDED: ${shieldedCount}/16 CH`
        : `STANDBY — ${shieldedCount}/16 CH BEHIND CONCRETE WALLS`
    );

    tbody.querySelectorAll('tr').forEach((tr) => {
      tr.addEventListener('mouseenter', () => {
        state.hoveredSensorId = tr.getAttribute('data-sensor-id') || '';
        if (typeof RadMon.renderFacilityCanvas === 'function') {
          RadMon.renderFacilityCanvas();
        }
      });
      tr.addEventListener('mouseleave', () => {
        state.hoveredSensorId = '';
        if (typeof RadMon.renderFacilityCanvas === 'function') {
          RadMon.renderFacilityCanvas();
        }
      });
    });
  };

  RadMon.updateTrainingDataTable = function () {
    const tbody = document.getElementById('trainingDataTableBody');
    if (!tbody) return;
    const sectorUpper = String(state.facilityKey).toUpperCase();
    const samples = Array.isArray(state.trainingSamples) ? state.trainingSamples : [];

    if (samples.length === 0) {
      setText(
        'trainDataCountLabel',
        `(0 TRAINING SAMPLES IN ${sectorUpper} — ${sectorUpper} AI MODEL IS UNTRAINED)`
      );
      tbody.innerHTML = `
        <tr>
          <td colspan="12" style="text-align:center;color:#64748b;padding:24px;">
            NO TRAINING DATA FOR ${sectorUpper} YET — Open Tab "02 // AI NEURAL PATHWAYS &amp; CONVERGENCE LAB" and click "TRAIN SECTOR AI ON LOGGED RUNS" or "GENERATE &amp; TRAIN SECTOR AI (64 MC SAMPLES)".
          </td>
        </tr>
      `;
      return;
    }

    let rewardedCount = 0;
    for (const s of samples) {
      const preE = Number(s.pre_err_m || 0);
      const postE = Number(s.post_err_m || 0);
      if (postE <= preE) rewardedCount++;
    }

    setText(
      'trainDataCountLabel',
      `(${samples.length} SAMPLES IN ${sectorUpper} | ${rewardedCount} REWARDED / ${samples.length - rewardedCount} PENALIZED)`
    );

    let html = '';
    for (const s of samples.slice(0, 120)) {
      const countsStr = Array.isArray(s.raw_counts) ? s.raw_counts.join(', ') : '';
      const preErr = Number(s.pre_err_m || 0);
      const postErr = Number(s.post_err_m || 0);
      const rewardDelta =
        typeof s.reward_delta_m === 'number'
          ? Number(s.reward_delta_m)
          : preErr - postErr;
      const isRewarded = s.outcome
        ? s.outcome === 'REWARDED'
        : postErr <= preErr;

      const outcomeBadge = isRewarded
        ? `<span class="row-status status-rewarded">REWARDED (+Δ)</span>`
        : `<span class="row-status status-alarm">PENALIZED (-Δ)</span>`;

      const preGuessX =
        typeof s.pre_pred_x === 'number' ? Number(s.pre_pred_x).toFixed(2) : '--';
      const preGuessY =
        typeof s.pre_pred_y === 'number' ? Number(s.pre_pred_y).toFixed(2) : '--';

      const shiftSign = rewardDelta >= 0 ? '+' : '';
      const shiftColor = rewardDelta >= 0 ? '#15803d' : '#dc2626';

      html += `
        <tr>
          <td class="ch-tag">${s.sample_id}</td>
          <td>${s.timestamp_utc || '--'}</td>
          <td><span class="row-status status-nominal">${s.source_tag}</span></td>
          <td>${outcomeBadge}</td>
          <td style="color:#dc2626;font-weight:600;">(${Number(s.true_x).toFixed(2)}, ${Number(s.true_y).toFixed(2)})</td>
          <td class="num">${Number(s.true_activity_kbq).toFixed(0)}</td>
          <td style="color:#b91c1c;">(${preGuessX}, ${preGuessY})</td>
          <td class="num" style="color:#b91c1c;">${preErr.toFixed(3)} m</td>
          <td style="color:#0284c7;font-weight:600;">(${Number(s.post_pred_x).toFixed(2)}, ${Number(s.post_pred_y).toFixed(2)})</td>
          <td class="num" style="font-weight:700;color:${isRewarded ? '#15803d' : '#dc2626'};">${postErr.toFixed(3)} m</td>
          <td class="num" style="font-weight:700;color:${shiftColor};">${shiftSign}${rewardDelta.toFixed(3)} m</td>
          <td style="font-size:11px;color:#334155;">[${countsStr}]</td>
        </tr>
      `;
    }
    tbody.innerHTML = html;
  };

  RadMon.updateRunLogTable = function () {
    const tbody = document.getElementById('runLogTableBody');
    if (!tbody) return;
    const sectorUpper = String(state.facilityKey).toUpperCase();
    setText('logCountLabel', `(${state.runLogs.length} RUNS LOGGED IN ${sectorUpper})`);

    if (state.runLogs.length === 0) {
      tbody.innerHTML = `
        <tr>
          <td colspan="12" style="text-align:center;color:#64748b;padding:24px;">
            0 RUNS LOGGED IN ${sectorUpper}. Click "STEP 1 MC FRAME", "SIMULATE NEW EVENT", or "START CONTINUOUS RUN" to record OpenMC runs.
          </td>
        </tr>
      `;
      return;
    }

    let html = '';
    const slice = state.runLogs.slice(0, 100);
    for (const row of slice) {
      const lowError = Number(row.ai_error_m) < 1.0;
      const aiStateBadge = row.ai_is_trained
        ? `<span class="row-status status-rewarded">TRAINED (${row.ai_epochs} EP)</span>`
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
          <td style="font-size:11px;color:#334155;">[${countsStr}]</td>
        </tr>
      `;
    }
    tbody.innerHTML = html;
  };

  RadMon.updateSnapshotArchive = function () {
    const listEl = document.getElementById('snapshotArchiveList');
    if (!listEl) return;

    const caps = Array.isArray(state.capturesHistory) ? state.capturesHistory : [];
    if (caps.length === 0) {
      listEl.innerHTML = `
        <div class="empty-archive-msg">
          No snapshots captured yet — click "SCREENSHOT MAP &amp; GRAPHS" to save a timestamped bundle into captures/.
        </div>
      `;
      return;
    }

    let html = '';
    for (const c of caps.slice(0, 12)) {
      const folderPath = String(c.folder_path || '');
      const files = Array.isArray(c.files) ? c.files : [];
      const linksHtml = files
        .map(
          /** @param {string} f */ (f) =>
            `<a class="snapshot-link" href="/${folderPath}/${f}" target="_blank" rel="noopener">${f}</a>`
        )
        .join('');

      html += `
        <div class="snapshot-card">
          <div class="snapshot-card-head">
            <span>${folderPath}/</span>
            <span style="color:#64748b;font-weight:500;">${c.captured_at_utc || ''}</span>
          </div>
          <div class="snapshot-links">${linksHtml}</div>
        </div>
      `;
    }
    listEl.innerHTML = html;
  };
})();
