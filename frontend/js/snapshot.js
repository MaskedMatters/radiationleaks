/**
 * frontend/js/snapshot.js
 * Captures high-resolution PNG screenshots of the 2D Facility Map, 4-Layer Neural Pathway
 * Graph, CDF Error Chart, Sensor Flux Chart, and a Composite Overview Board, then saves
 * them via `POST /api/capture-snapshot` into `captures/snapshot_<timestamp>/`.
 */

(function () {
  'use strict';

  /** @type {Record<string, any>} */
  const RadMon = /** @type {any} */ (window).RadMon;
  const { state, setText, getButton, apiPost } = RadMon;

  RadMon.captureSnapshotBundle = async function () {
    const mapCanvas = RadMon.facilityCanvas;
    const pathwayCanvas = RadMon.pathwayCanvas;
    const cdfCanvas = document.getElementById('cdfChart');
    const fluxCanvas = document.getElementById('fluxChart');
    const rewardCanvas = document.getElementById('rewardCurveChart');
    if (!mapCanvas) return;

    // Ensure all tab panes are momentarily measurable so Chart.js canvases in inactive tabs
    // render at full non-zero dimensions before calling .toDataURL()
    const hiddenPanes = /** @type {HTMLElement[]} */ ([]);
    document.querySelectorAll('.tab-pane').forEach((el) => {
      if (el instanceof HTMLElement && !el.classList.contains('active')) {
        hiddenPanes.push(el);
        el.style.display = 'flex';
        el.style.position = 'fixed';
        el.style.left = '-9999px';
        el.style.width = '1400px';
      }
    });

    if (typeof RadMon.renderFacilityCanvas === 'function') {
      RadMon.renderFacilityCanvas();
    }
    if (typeof RadMon.renderNeuralPathwayCanvas === 'function') {
      RadMon.renderNeuralPathwayCanvas();
    }
    if (RadMon.chartStore) {
      for (const k of Object.keys(RadMon.chartStore)) {
        const ch = RadMon.chartStore[k];
        if (ch && typeof ch.resize === 'function') {
          ch.resize();
          ch.update('none');
        }
      }
    }

    const mapDataUrl = mapCanvas.toDataURL('image/png');
    const cdfDataUrl =
      cdfCanvas instanceof HTMLCanvasElement ? cdfCanvas.toDataURL('image/png') : '';
    const fluxDataUrl =
      fluxCanvas instanceof HTMLCanvasElement ? fluxCanvas.toDataURL('image/png') : '';
    const pathwayDataUrl =
      pathwayCanvas instanceof HTMLCanvasElement
        ? pathwayCanvas.toDataURL('image/png')
        : '';

    // Build a high-res composite board (1360 x 1060)
    const compCanvas = document.createElement('canvas');
    compCanvas.width = 1360;
    compCanvas.height = 1060;
    const g = compCanvas.getContext('2d');
    let compositeDataUrl = '';

    if (g) {
      g.fillStyle = '#f8f9fa';
      g.fillRect(0, 0, compCanvas.width, compCanvas.height);

      // Header Banner
      g.fillStyle = '#ffffff';
      g.fillRect(16, 14, 1328, 56);
      g.strokeStyle = '#cbd5e1';
      g.strokeRect(16, 14, 1328, 56);

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

      // Left Box: 2D Facility Map (650x650)
      g.fillStyle = '#ffffff';
      g.fillRect(16, 84, 650, 650);
      g.strokeStyle = '#cbd5e1';
      g.strokeRect(16, 84, 650, 650);
      g.drawImage(mapCanvas, 20, 88, 642, 642);

      // Right Top Box: CDF Error Distribution Chart
      g.fillStyle = '#ffffff';
      g.fillRect(680, 84, 664, 318);
      g.strokeStyle = '#cbd5e1';
      g.strokeRect(680, 84, 664, 318);
      g.fillStyle = '#334155';
      g.font = '700 11px "JetBrains Mono", monospace';
      g.fillText('CDF ERROR DISTRIBUTION (AI NEURAL NETWORK)', 694, 104);
      if (cdfCanvas instanceof HTMLCanvasElement && cdfCanvas.width > 0) {
        g.drawImage(cdfCanvas, 690, 114, 644, 278);
      }

      // Right Bottom Box: Sensor Flux Profile Chart
      g.fillStyle = '#ffffff';
      g.fillRect(680, 416, 664, 318);
      g.strokeStyle = '#cbd5e1';
      g.strokeRect(680, 416, 664, 318);
      g.fillStyle = '#334155';
      g.font = '700 11px "JetBrains Mono", monospace';
      g.fillText('SENSOR FLUX PROFILE (16-CH RECORDED vs. AI PREDICTION)', 694, 436);
      if (fluxCanvas instanceof HTMLCanvasElement && fluxCanvas.width > 0) {
        g.drawImage(fluxCanvas, 690, 446, 644, 278);
      }

      // Bottom Full-Width Box: Neural Pathway & Reward Convergence
      g.fillStyle = '#ffffff';
      g.fillRect(16, 748, 1328, 296);
      g.strokeStyle = '#cbd5e1';
      g.strokeRect(16, 748, 1328, 296);
      g.fillStyle = '#334155';
      g.font = '700 11px "JetBrains Mono", monospace';
      g.fillText(
        'AI NEURAL PATHWAY REINFORCEMENT (REWARDED +Δ|W| vs. PENALIZED -Δ|W|) & EPOCH CONVERGENCE',
        30,
        768
      );
      if (pathwayCanvas instanceof HTMLCanvasElement && pathwayCanvas.width > 0) {
        g.drawImage(pathwayCanvas, 26, 776, 760, 258);
      }
      if (rewardCanvas instanceof HTMLCanvasElement && rewardCanvas.width > 0) {
        g.drawImage(rewardCanvas, 800, 776, 530, 258);
      }

      compositeDataUrl = compCanvas.toDataURL('image/png');
    }

    // Restore hidden tab panes
    for (const el of hiddenPanes) {
      el.style.display = '';
      el.style.position = '';
      el.style.left = '';
      el.style.width = '';
    }

    const btn1 = getButton('btnCaptureSnapshot');
    const btn2 = getButton('btnCaptureHeader');
    if (btn1) btn1.textContent = 'SAVING SNAPSHOT...';
    if (btn2) btn2.textContent = 'SAVING SNAPSHOT...';

    const res = await apiPost('/api/capture-snapshot', {
      map_png: mapDataUrl,
      cdf_png: cdfDataUrl,
      flux_png: fluxDataUrl,
      pathway_png: pathwayDataUrl,
      composite_png: compositeDataUrl
    });

    if (res && res.ok && res.capture) {
      const folderPath = res.capture.folder_path;
      setText('lastSnapshotBanner', `SAVED: ${folderPath}/`);
      if (btn1) btn1.textContent = `SAVED TO ${folderPath}`;
      if (btn2) btn2.textContent = 'SNAPSHOT SAVED!';
      window.setTimeout(() => {
        if (btn1) btn1.textContent = 'SCREENSHOT MAP & GRAPHS';
        if (btn2) btn2.textContent = 'CAPTURE NEW SNAPSHOT BUNDLE';
      }, 2500);
    } else {
      if (btn1) btn1.textContent = 'SCREENSHOT MAP & GRAPHS';
      if (btn2) btn2.textContent = 'CAPTURE NEW SNAPSHOT BUNDLE';
    }
  };
})();

