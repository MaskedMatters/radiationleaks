/**
 * frontend/js/pathway_renderer.js
 * Renders the spacious 4-layer AI Neural Pathway & Synapse Reinforcement Canvas (1120x400)
 * and the 16-Channel Sensor Attribution Grid in Tab 02.
 */

(function () {
  'use strict';

  /** @type {Record<string, any>} */
  const RadMon = /** @type {any} */ (window).RadMon;
  const { state, setText } = RadMon;

  const pathwayCanvasEl = document.getElementById('neuralPathwayCanvas');
  const pathwayCanvas =
    pathwayCanvasEl instanceof HTMLCanvasElement ? pathwayCanvasEl : null;
  const pathwayCtx = pathwayCanvas ? pathwayCanvas.getContext('2d') : null;
  RadMon.pathwayCanvas = pathwayCanvas;

  RadMon.renderNeuralPathwayCanvas = function () {
    if (!pathwayCanvas || !pathwayCtx) return;

    const pt = state.pathwayTelemetry || {};
    const isTrained = Boolean(pt.is_trained);

    const rewSyn = Number(pt.rewarded_synapses || 0);
    const penSyn = Number(pt.penalized_synapses || 0);
    const totSyn = Math.max(1, rewSyn + penSyn);
    const rewPct = ((rewSyn / totSyn) * 100).toFixed(1);
    const spatTraj = Array.isArray(pt.spatial_trajectories)
      ? pt.spatial_trajectories
      : [];
    const rewPaths = spatTraj.filter(
      /** @param {Record<string, any>} s */ (s) =>
        s.outcome ? s.outcome === 'REWARDED' : Number(s.post_err_m) <= Number(s.pre_err_m)
    ).length;

    if (isTrained) {
      setText(
        'pathwaySummaryLabel',
        `REWARDED SYNAPSES: ${rewSyn} (${rewPct}%) | PENALIZED: ${penSyn} | SPATIAL PATHS REWARDED: ${rewPaths}/${spatTraj.length}`
      );
    } else {
      setText(
        'pathwaySummaryLabel',
        'UNTRAINED (0 KNOWLEDGE) — TRAIN AI TO VISUALIZE REWARDED vs. PENALIZED SYNAPSE PATHWAYS'
      );
    }

    const dpr = window.devicePixelRatio || 1;
    const W = 1120;
    const H = 400;
    if (pathwayCanvas.width !== W * dpr || pathwayCanvas.height !== H * dpr) {
      pathwayCanvas.width = W * dpr;
      pathwayCanvas.height = H * dpr;
    }

    const g = pathwayCtx;
    g.save();
    g.scale(dpr, dpr);

    g.fillStyle = '#ffffff';
    g.fillRect(0, 0, W, H);

    // Column Header Banner
    g.fillStyle = '#f8fafc';
    g.fillRect(0, 0, W, 32);
    g.strokeStyle = '#e2e8f0';
    g.beginPath();
    g.moveTo(0, 32);
    g.lineTo(W, 32);
    g.stroke();

    const colX = [145, 445, 745, 995];
    const colTitles = [
      'LAYER 0: 16 NaI(Tl) SENSORS (S-01..S-16)',
      'LAYER 1: TOP 8 HIDDEN NEURONS (128 RELU)',
      'LAYER 2: TOP 6 HIDDEN NEURONS (64 RELU)',
      'LAYER 3: LOCALIZATION OUTPUTS (3)'
    ];

    g.font = '700 11px "JetBrains Mono", monospace';
    g.fillStyle = '#334155';
    g.textAlign = 'center';
    g.textBaseline = 'middle';
    for (let c = 0; c < 4; c++) {
      g.fillText(colTitles[c], colX[c], 16);
    }

    const topY = 48;
    const botY = H - 20;
    const availH = botY - topY;

    // 1. Input Sensor Nodes (16)
    /** @type {{x: number, y: number, label: string, strength: number, delta: number, rewarded: boolean}[]} */
    const inNodes = [];
    const sensorAttr = Array.isArray(pt.sensor_attribution)
      ? pt.sensor_attribution
      : [];
    let maxStrength = 1.0;
    for (const sa of sensorAttr) {
      if (Number(sa.synapse_strength) > maxStrength) {
        maxStrength = Number(sa.synapse_strength);
      }
    }

    for (let i = 0; i < 16; i++) {
      const ny = topY + (i / 15) * availH;
      const sa = sensorAttr[i] || {};
      inNodes.push({
        x: colX[0],
        y: ny,
        label: `S-${String(i + 1).padStart(2, '0')}`,
        strength: Number(sa.synapse_strength || 0.5),
        delta: Number(sa.delta_strength || 0.0),
        rewarded: sa.status !== 'PENALIZED'
      });
    }

    // 2. Hidden Layer 1 Nodes (Top 8)
    const rawH1 = Array.isArray(pt.h1_nodes) ? pt.h1_nodes : [];
    /** @type {{x: number, y: number, label: string, act: number}[]} */
    const h1Nodes = [];
    for (let i = 0; i < 8; i++) {
      const ny = topY + 18 + (i / 7) * (availH - 36);
      const item = rawH1[i] || {};
      h1Nodes.push({
        x: colX[1],
        y: ny,
        label: String(item.neuron_id || `H1-${String(i).padStart(3, '0')}`),
        act: Number(item.activation || 0.0)
      });
    }

    // 3. Hidden Layer 2 Nodes (Top 6)
    const rawH2 = Array.isArray(pt.h2_nodes) ? pt.h2_nodes : [];
    /** @type {{x: number, y: number, label: string, act: number}[]} */
    const h2Nodes = [];
    for (let i = 0; i < 6; i++) {
      const ny = topY + 34 + (i / 5) * (availH - 68);
      const item = rawH2[i] || {};
      h2Nodes.push({
        x: colX[2],
        y: ny,
        label: String(item.neuron_id || `H2-${String(i).padStart(2, '0')}`),
        act: Number(item.activation || 0.0)
      });
    }

    // 4. Output Head Nodes (3)
    const rawOut = Array.isArray(pt.out_nodes) ? pt.out_nodes : [];
    const defaultOutLabels = ['EST X [m]', 'EST Y [m]', 'ACT [kBq]'];
    /** @type {{x: number, y: number, label: string, sig: number}[]} */
    const outNodes = [];
    for (let i = 0; i < 3; i++) {
      const ny = topY + 68 + (i / 2) * (availH - 136);
      const item = rawOut[i] || {};
      outNodes.push({
        x: colX[3],
        y: ny,
        label: String(item.label || defaultOutLabels[i]),
        sig: Number(item.sig || 0.5)
      });
    }

    /**
     * Draws a synapse connection line.
     * Solid Green = Rewarded (+Δ|W|), Dashed Red = Penalized (-Δ|W|)
     * @param {number} x1
     * @param {number} y1
     * @param {number} x2
     * @param {number} y2
     * @param {Record<string, any>} edge
     */
    function drawSynapseEdge(x1, y1, x2, y2, edge) {
      g.save();
      if (!isTrained) {
        g.setLineDash([2, 4]);
        g.strokeStyle = 'rgba(148, 163, 184, 0.25)';
        g.lineWidth = 0.9;
      } else {
        const deltaMag = Number(edge.delta || 0);
        const absDelta = Math.min(1.0, Math.abs(deltaMag) * 3.5);
        const width = 0.95 + absDelta * 2.8;
        const alpha = 0.22 + absDelta * 0.68;
        if (edge.rewarded) {
          g.strokeStyle = `rgba(21, 128, 61, ${alpha.toFixed(2)})`;
          g.lineWidth = width;
        } else {
          g.setLineDash([5, 3]);
          g.strokeStyle = `rgba(220, 38, 38, ${alpha.toFixed(2)})`;
          g.lineWidth = width;
        }
      }
      g.beginPath();
      g.moveTo(x1, y1);
      g.lineTo(x2, y2);
      g.stroke();
      g.restore();
    }

    // Draw Edges Layer 1: Input -> H1
    const edgesInH1 = Array.isArray(pt.edges_in_h1) ? pt.edges_in_h1 : [];
    for (const e of edgesInH1) {
      const src = inNodes[Number(e.from)];
      const dst = h1Nodes[Number(e.to)];
      if (src && dst) {
        drawSynapseEdge(src.x + 8, src.y, dst.x - 8, dst.y, e);
      }
    }

    // Draw Edges Layer 2: H1 -> H2
    const edgesH1H2 = Array.isArray(pt.edges_h1_h2) ? pt.edges_h1_h2 : [];
    for (const e of edgesH1H2) {
      const src = h1Nodes[Number(e.from)];
      const dst = h2Nodes[Number(e.to)];
      if (src && dst) {
        drawSynapseEdge(src.x + 8, src.y, dst.x - 8, dst.y, e);
      }
    }

    // Draw Edges Layer 3: H2 -> Output
    const edgesH2Out = Array.isArray(pt.edges_h2_out) ? pt.edges_h2_out : [];
    for (const e of edgesH2Out) {
      const src = h2Nodes[Number(e.from)];
      const dst = outNodes[Number(e.to)];
      if (src && dst) {
        drawSynapseEdge(src.x + 8, src.y, dst.x - 8, dst.y, e);
      }
    }

    // Draw Input Sensor Nodes + Strength Bars
    for (let i = 0; i < inNodes.length; i++) {
      const n = inNodes[i];
      const barMaxW = 64;
      const barW = Math.min(barMaxW, Math.max(4, (n.strength / maxStrength) * barMaxW));

      g.fillStyle = '#f1f5f9';
      g.fillRect(16, n.y - 4.5, barMaxW, 9);
      g.fillStyle = !isTrained ? '#94a3b8' : n.rewarded ? '#15803d' : '#dc2626';
      g.fillRect(16 + (barMaxW - barW), n.y - 4.5, barW, 9);

      g.font = '700 10.5px "JetBrains Mono", monospace';
      g.fillStyle = '#0f172a';
      g.textAlign = 'right';
      g.textBaseline = 'middle';
      g.fillText(n.label, n.x - 10, n.y);

      g.beginPath();
      g.arc(n.x, n.y, 5.0, 0, Math.PI * 2);
      g.fillStyle = '#ffffff';
      g.fill();
      g.lineWidth = 1.8;
      g.strokeStyle = !isTrained ? '#64748b' : n.rewarded ? '#15803d' : '#dc2626';
      g.stroke();
    }

    // Draw H1 Nodes
    for (const n of h1Nodes) {
      g.beginPath();
      g.arc(n.x, n.y, 7.5, 0, Math.PI * 2);
      g.fillStyle = '#ffffff';
      g.fill();
      g.lineWidth = 2.0;
      g.strokeStyle = isTrained ? '#15803d' : '#64748b';
      g.stroke();

      g.font = '700 10px "JetBrains Mono", monospace';
      g.fillStyle = '#0f172a';
      g.textAlign = 'center';
      g.textBaseline = 'bottom';
      g.fillText(`${n.label} (a=${n.act.toFixed(2)})`, n.x, n.y - 10);
    }

    // Draw H2 Nodes
    for (const n of h2Nodes) {
      g.beginPath();
      g.arc(n.x, n.y, 7.5, 0, Math.PI * 2);
      g.fillStyle = '#ffffff';
      g.fill();
      g.lineWidth = 2.0;
      g.strokeStyle = isTrained ? '#0284c7' : '#64748b';
      g.stroke();

      g.font = '700 10px "JetBrains Mono", monospace';
      g.fillStyle = '#0f172a';
      g.textAlign = 'center';
      g.textBaseline = 'bottom';
      g.fillText(`${n.label} (a=${n.act.toFixed(2)})`, n.x, n.y - 10);
    }

    // Draw Output Nodes
    for (const n of outNodes) {
      g.beginPath();
      g.arc(n.x, n.y, 8.5, 0, Math.PI * 2);
      g.fillStyle = isTrained ? '#f0f9ff' : '#f8f9fa';
      g.fill();
      g.lineWidth = 2.2;
      g.strokeStyle = isTrained ? '#0284c7' : '#64748b';
      g.stroke();

      g.font = '700 11px "JetBrains Mono", monospace';
      g.fillStyle = '#0f172a';
      g.textAlign = 'left';
      g.textBaseline = 'middle';
      g.fillText(n.label, n.x + 14, n.y - 6);

      g.font = '500 10px "JetBrains Mono", monospace';
      g.fillStyle = '#475569';
      g.fillText(`σ=${n.sig.toFixed(3)}`, n.x + 14, n.y + 8);
    }

    g.restore();

    // Update 16-Sensor Attribution Grid below the canvas
    renderSensorAttributionGrid(sensorAttr, isTrained);
  };

  /**
   * @param {Record<string, any>[]} sensorAttr
   * @param {boolean} isTrained
   */
  function renderSensorAttributionGrid(sensorAttr, isTrained) {
    const gridEl = document.getElementById('sensorAttributionGrid');
    if (!gridEl) return;

    if (!Array.isArray(sensorAttr) || sensorAttr.length === 0) {
      gridEl.innerHTML = '';
      return;
    }

    let html = '';
    for (const sa of sensorAttr) {
      const deltaVal = Number(sa.delta_strength || 0);
      const isRew = sa.status === 'REWARDED';
      const cardClass = !isTrained ? '' : isRew ? 'rewarded' : 'penalized';
      const badgeClass = !isTrained
        ? 'status-nominal'
        : isRew
        ? 'status-rewarded'
        : 'status-alarm';
      const badgeText = !isTrained ? 'RAW' : isRew ? 'REWARDED' : 'PENALIZED';
      const sign = deltaVal >= 0 ? '+' : '';

      html += `
        <div class="sensor-attr-card ${cardClass}">
          <div class="sensor-attr-top">
            <span>${sa.id}</span>
            <span class="row-status ${badgeClass}">${badgeText}</span>
          </div>
          <div class="sensor-attr-val">||W||: ${Number(sa.synapse_strength).toFixed(2)} (${sign}${deltaVal.toFixed(2)})</div>
        </div>
      `;
    }
    gridEl.innerHTML = html;
  }
})();

