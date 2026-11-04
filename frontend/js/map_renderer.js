/**
 * frontend/js/map_renderer.js
 * 2D Facility Floorplan Canvas renderer (720x720) with interactive source dragging,
 * 22 cm concrete shielding wall barriers, 16 NaI(Tl) sensor count-rate heat bubbles,
 * AI localization crosshair, and optional spatial training trajectory vectors.
 */

(function () {
  'use strict';

  /** @type {Record<string, any>} */
  const RadMon = /** @type {any} */ (window).RadMon;
  const { state, setText, apiPost, DOMAIN_SIZE, CANVAS_SIZE, PAD } = RadMon;

  const canvasEl = document.getElementById('facilityCanvas');
  const canvas = canvasEl instanceof HTMLCanvasElement ? canvasEl : null;
  const ctx = canvas ? canvas.getContext('2d') : null;
  RadMon.facilityCanvas = canvas;

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

  /**
   * Draws a directed arrow line from (x1, y1) to (x2, y2).
   * @param {CanvasRenderingContext2D} g
   * @param {number} x1
   * @param {number} y1
   * @param {number} x2
   * @param {number} y2
   * @param {number} headLen
   */
  function drawArrowLine(g, x1, y1, x2, y2, headLen) {
    const dx = x2 - x1;
    const dy = y2 - y1;
    const len = Math.hypot(dx, dy);
    if (len < 2) return;
    const angle = Math.atan2(dy, dx);
    g.beginPath();
    g.moveTo(x1, y1);
    g.lineTo(x2, y2);
    g.stroke();

    g.beginPath();
    g.moveTo(x2, y2);
    g.lineTo(
      x2 - headLen * Math.cos(angle - Math.PI / 6),
      y2 - headLen * Math.sin(angle - Math.PI / 6)
    );
    g.lineTo(
      x2 - headLen * Math.cos(angle + Math.PI / 6),
      y2 - headLen * Math.sin(angle + Math.PI / 6)
    );
    g.closePath();
    g.fill();
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
    context.font = '600 10.5px "JetBrains Mono", monospace';
    const padX = 6;
    const boxH = 18;
    const textW = context.measureText(text).width;
    const boxW = textW + padX * 2;

    const clampedX = Math.max(PAD + 4, Math.min(CANVAS_SIZE - PAD - boxW - 4, x));
    const clampedY = Math.max(PAD + 4, Math.min(CANVAS_SIZE - PAD - boxH - 4, y));

    context.fillStyle = 'rgba(255, 255, 255, 0.95)';
    context.strokeStyle = borderColor;
    context.lineWidth = 1.2;
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

  RadMon.renderFacilityCanvas = function () {
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

    // Architectural Grid
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

    // Zone Labels
    ctx.font = '600 10.5px "JetBrains Mono", monospace';
    ctx.fillStyle = '#94a3b8';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    for (const z of facility.zones) {
      const tl = worldToCanvas(Number(z.x), Number(z.y), W, H);
      ctx.fillText(String(z.label), tl.cx + 8, tl.cy + 8);
    }

    // 22 cm Concrete Shielding Walls (drawn as physical structural barriers)
    for (const w of facility.walls) {
      const p1 = worldToCanvas(Number(w.x1), Number(w.y1), W, H);
      const p2 = worldToCanvas(Number(w.x2), Number(w.y2), W, H);
      // Outer concrete casing
      ctx.strokeStyle = '#475569';
      ctx.lineWidth = 7.0;
      ctx.lineCap = 'butt';
      ctx.beginPath();
      ctx.moveTo(p1.cx, p1.cy);
      ctx.lineTo(p2.cx, p2.cy);
      ctx.stroke();

      // Inner concrete core highlight
      ctx.strokeStyle = '#94a3b8';
      ctx.lineWidth = 3.0;
      ctx.beginPath();
      ctx.moveTo(p1.cx, p1.cy);
      ctx.lineTo(p2.cx, p2.cy);
      ctx.stroke();
    }

    // Perimeter Frame & Axis Ticks
    ctx.strokeStyle = '#334155';
    ctx.lineWidth = 1.5;
    ctx.strokeRect(PAD, PAD, plotW, plotH);

    ctx.fillStyle = '#64748b';
    ctx.font = '500 10.5px "JetBrains Mono", monospace';
    for (let m = 0; m <= DOMAIN_SIZE; m += 2) {
      const pos = PAD + m * pxPerMeter;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'bottom';
      ctx.fillText(`${m}m`, pos, PAD - 7);
      ctx.textBaseline = 'top';
      ctx.fillText(`${m}m`, pos, PAD + plotH + 7);
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      ctx.fillText(`${m}m`, PAD - 7, pos);
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

    // --- Optional Spatial Training Trajectories Overlay (Strictly for Active Sector) ---
    if (state.showTrainingTrajectories) {
      const pt = state.pathwayTelemetry || {};
      const trajectories =
        Array.isArray(pt.spatial_trajectories) && pt.spatial_trajectories.length > 0
          ? pt.spatial_trajectories
          : state.trainingSamples.filter(
              /** @param {Record<string, any>} s */ (s) => s.facility === state.facilityKey
            );

      for (const t of trajectories.slice(0, 36)) {
        if (typeof t.pre_pred_x !== 'number' || typeof t.post_pred_x !== 'number') {
          continue;
        }
        const preP = worldToCanvas(Number(t.pre_pred_x), Number(t.pre_pred_y), W, H);
        const postP = worldToCanvas(Number(t.post_pred_x), Number(t.post_pred_y), W, H);
        const targetP = worldToCanvas(Number(t.true_x), Number(t.true_y), W, H);
        const isRewarded = t.outcome
          ? t.outcome === 'REWARDED'
          : Number(t.post_err_m) <= Number(t.pre_err_m);

        // 1. Red 'x' at Penalized Pre-Train Guess
        ctx.strokeStyle = 'rgba(220, 38, 38, 0.55)';
        ctx.lineWidth = 1.3;
        ctx.beginPath();
        ctx.moveTo(preP.cx - 3.5, preP.cy - 3.5);
        ctx.lineTo(preP.cx + 3.5, preP.cy + 3.5);
        ctx.moveTo(preP.cx + 3.5, preP.cy - 3.5);
        ctx.lineTo(preP.cx - 3.5, preP.cy + 3.5);
        ctx.stroke();

        // 2. Arrow from Pre-Train Guess -> Post-Train Prediction
        ctx.save();
        if (isRewarded) {
          ctx.strokeStyle = 'rgba(21, 128, 61, 0.45)';
          ctx.fillStyle = 'rgba(21, 128, 61, 0.60)';
          ctx.lineWidth = 1.4;
        } else {
          ctx.setLineDash([3, 2]);
          ctx.strokeStyle = 'rgba(220, 38, 38, 0.48)';
          ctx.fillStyle = 'rgba(220, 38, 38, 0.60)';
          ctx.lineWidth = 1.4;
        }
        drawArrowLine(ctx, preP.cx, preP.cy, postP.cx, postP.cy, 5.5);
        ctx.restore();

        // 3. Residual line to True Target
        ctx.save();
        ctx.setLineDash([2, 2]);
        ctx.strokeStyle = 'rgba(2, 132, 199, 0.30)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(postP.cx, postP.cy);
        ctx.lineTo(targetP.cx, targetP.cy);
        ctx.stroke();
        ctx.restore();

        // 4. Dot at Post-Train Prediction
        ctx.beginPath();
        ctx.arc(postP.cx, postP.cy, 2.8, 0, Math.PI * 2);
        ctx.fillStyle = isRewarded
          ? 'rgba(21, 128, 61, 0.80)'
          : 'rgba(220, 38, 38, 0.80)';
        ctx.fill();
      }
    }

    // --- 16 Fixed NaI(Tl) Sensor Probes & Wall Shielding Indicators ---
    const rawValues = state.sensorReadings.map(
      /** @param {Record<string, any>} s */ (s) => Number(s.raw)
    );
    const maxRaw = Math.max(...rawValues, state.ambientBackgroundCps + 10);

    for (const s of state.sensorReadings) {
      const pt = worldToCanvas(Number(s.x), Number(s.y), W, H);
      const rawVal = Number(s.raw);
      const nWalls = Number(s.wallCrossings || 0);
      const wallTrans =
        typeof s.wallTransmission === 'number' ? Number(s.wallTransmission) : 1.0;

      // If this sensor is hovered in the table, draw its ray from True Source showing wall attenuation
      if (state.hoveredSensorId === s.id) {
        ctx.save();
        ctx.beginPath();
        if (nWalls > 0) {
          ctx.setLineDash([5, 4]);
          ctx.strokeStyle = 'rgba(180, 83, 9, 0.75)';
        } else {
          ctx.strokeStyle = 'rgba(21, 128, 61, 0.70)';
        }
        ctx.lineWidth = 1.8;
        ctx.moveTo(truePt.cx, truePt.cy);
        ctx.lineTo(pt.cx, pt.cy);
        ctx.stroke();
        ctx.restore();
      }

      const netRatio =
        rawVal > 0
          ? Math.max(
              0,
              (rawVal - state.ambientBackgroundCps) /
                Math.max(1, maxRaw - state.ambientBackgroundCps)
            )
          : 0;
      const bubbleRadius = rawVal > 0 ? 11 + Math.sqrt(netRatio) * 35 : 9;
      const isAlarm = s.status === 'ALARM';
      const isElevated = s.status === 'ELEVATED';

      ctx.beginPath();
      ctx.arc(pt.cx, pt.cy, bubbleRadius, 0, Math.PI * 2);
      if (isAlarm) {
        ctx.fillStyle = 'rgba(220, 38, 38, 0.12)';
        ctx.strokeStyle = 'rgba(220, 38, 38, 0.42)';
      } else if (isElevated) {
        ctx.fillStyle = 'rgba(217, 119, 6, 0.12)';
        ctx.strokeStyle = 'rgba(217, 119, 6, 0.42)';
      } else {
        ctx.fillStyle = 'rgba(100, 116, 139, 0.05)';
        ctx.strokeStyle = 'rgba(100, 116, 139, 0.22)';
      }
      ctx.lineWidth = 1;
      ctx.fill();
      ctx.stroke();

      if (state.hoveredSensorId === s.id) {
        ctx.beginPath();
        ctx.arc(pt.cx, pt.cy, bubbleRadius + 4, 0, Math.PI * 2);
        ctx.strokeStyle = '#0f172a';
        ctx.lineWidth = 1.6;
        ctx.stroke();
      }

      ctx.beginPath();
      ctx.arc(pt.cx, pt.cy, 5.5, 0, Math.PI * 2);
      ctx.fillStyle = '#ffffff';
      ctx.fill();
      ctx.lineWidth = 1.8;
      ctx.strokeStyle = isAlarm ? '#dc2626' : isElevated ? '#d97706' : '#334155';
      ctx.stroke();

      ctx.beginPath();
      ctx.arc(pt.cx, pt.cy, 2.0, 0, Math.PI * 2);
      ctx.fillStyle = isAlarm ? '#dc2626' : isElevated ? '#d97706' : '#334155';
      ctx.fill();

      ctx.font = '700 10.5px "JetBrains Mono", monospace';
      ctx.fillStyle = '#0f172a';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'bottom';
      const shieldTag = nWalls > 0 ? `${s.id} [${nWalls}W]` : String(s.id);
      ctx.fillText(shieldTag, pt.cx, pt.cy - 8);

      ctx.font = '500 10px "JetBrains Mono", monospace';
      ctx.fillStyle = isAlarm ? '#b91c1c' : nWalls > 0 ? '#64748b' : '#334155';
      ctx.textBaseline = 'top';
      const transPct = Math.round(wallTrans * 100);
      ctx.fillText(
        nWalls > 0 ? `${rawVal} cps (${transPct}%)` : `${rawVal} cps`,
        pt.cx,
        pt.cy + 8
      );
    }

    if (state.hasRun) {
      const aiPt = worldToCanvas(Number(aiSol.x), Number(aiSol.y), W, H);

      // Distance Delta Vector from True Source to AI Estimate
      ctx.beginPath();
      ctx.setLineDash([4, 3]);
      ctx.moveTo(truePt.cx, truePt.cy);
      ctx.lineTo(aiPt.cx, aiPt.cy);
      ctx.strokeStyle = '#dc2626';
      ctx.lineWidth = 1.6;
      ctx.stroke();
      ctx.setLineDash([]);

      // AI Crosshair
      const chSize = 12;
      ctx.strokeStyle = '#0284c7';
      ctx.lineWidth = 2.0;
      ctx.beginPath();
      ctx.moveTo(aiPt.cx - chSize, aiPt.cy);
      ctx.lineTo(aiPt.cx + chSize, aiPt.cy);
      ctx.moveTo(aiPt.cx, aiPt.cy - chSize);
      ctx.lineTo(aiPt.cx, aiPt.cy + chSize);
      ctx.stroke();

      ctx.beginPath();
      ctx.arc(aiPt.cx, aiPt.cy, 5.0, 0, Math.PI * 2);
      ctx.strokeStyle = '#0284c7';
      ctx.lineWidth = 1.6;
      ctx.stroke();

      const sectorUpper = String(state.facilityKey).toUpperCase();
      const tag = aiSol.is_trained
        ? `AI-${sectorUpper}`
        : `AI-UNTRAINED (${sectorUpper})`;
      drawCanvasCallout(
        ctx,
        aiPt.cx + 12,
        aiPt.cy + 8,
        `${tag} (${Number(aiSol.x).toFixed(2)}, ${Number(aiSol.y).toFixed(2)}) | Δ=${Number(aiSol.error_meters).toFixed(2)}m`,
        '#0284c7'
      );
    }

    // True Leak Source Diamond Marker
    ctx.save();
    ctx.translate(truePt.cx, truePt.cy);
    ctx.rotate(Math.PI / 4);
    ctx.fillStyle = '#dc2626';
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    ctx.rect(-6.0, -6.0, 12, 12);
    ctx.fill();
    ctx.stroke();
    ctx.restore();

    drawCanvasCallout(
      ctx,
      truePt.cx + 12,
      truePt.cy - 20,
      `TRUE (${Number(state.trueSource.x).toFixed(2)}, ${Number(state.trueSource.y).toFixed(2)})`,
      '#dc2626'
    );

    ctx.restore();
    ctx.restore();
  };

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
    RadMon.renderFacilityCanvas();

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

  RadMon.initMapInteractions = function () {
    if (!canvas) return;

    canvas.addEventListener('mousedown', (evt) => {
      const { cx, cy } = getCanvasRelativeCoords(evt);
      const { wx, wy } = canvasToWorld(cx, cy, CANVAS_SIZE, CANVAS_SIZE);
      state.draggingSource = true;
      dispatchSourcePositionToBackend(wx, wy, 'CANVAS-RELOC');
    });

    window.addEventListener('mousemove', (evt) => {
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
  };
})();
