/**
 * frontend/js/state.js
 * Central state store, constants, DOM utilities, and REST API helper for RAD-MON-04.
 */

(function () {
  'use strict';

  /** @type {Record<string, any>} */
  const RadMon =
    /** @type {any} */ (window).RadMon ||
    (/** @type {any} */ (window).RadMon = {});

  RadMon.DOMAIN_SIZE = 20.0;
  RadMon.CANVAS_SIZE = 720;
  RadMon.PAD = 42;

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

  RadMon.DEFAULT_SOLUTION = DEFAULT_SOLUTION;

  /** @type {Record<string, any>} */
  RadMon.state = {
    activeTab: 'tab-map',
    connected: false,
    hasRun: false,
    running: false,
    batchRunning: false,
    trainingRunning: false,
    showTrainingTrajectories: true,
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
      model_version: 'RADNET-v3.2-PATHWAY-MLP',
      is_trained: false,
      knowledge_state: 'UNTRAINED (0 KNOWLEDGE)',
      trained_epochs: 0,
      samples_seen: 0,
      total_samples: 0,
      last_loss: 0.0,
      val_mae_meters: 0.0,
      weight_norms: { W1_l2: 0.0, W2_l2: 0.0, W3_l2: 0.0 }
    },
    pathwayTelemetry: {
      facility_key: 'sector-a',
      is_trained: false,
      trained_epochs: 0,
      samples_seen: 0,
      total_synapses: 12416,
      rewarded_synapses: 0,
      penalized_synapses: 0,
      h1_nodes: [],
      h2_nodes: [],
      out_nodes: [],
      edges_in_h1: [],
      edges_h1_h2: [],
      edges_h2_out: [],
      sensor_attribution: [],
      epoch_curve: [],
      spatial_trajectories: []
    },
    validationErrorsAi: [],
    runLogs: [],
    trainingSamples: [],
    capturesHistory: [],
    hoveredSensorId: '',
    draggingSource: false
  };

  /**
   * @param {string} id
   * @param {string} text
   */
  RadMon.setText = function (id, text) {
    const el = document.getElementById(id);
    if (el) {
      el.textContent = text;
    }
  };

  /**
   * @param {string} id
   * @returns {HTMLInputElement | HTMLSelectElement | null}
   */
  RadMon.getFormControl = function (id) {
    const el = document.getElementById(id);
    if (el instanceof HTMLInputElement || el instanceof HTMLSelectElement) {
      return el;
    }
    return null;
  };

  /**
   * @param {string} id
   * @returns {HTMLButtonElement | null}
   */
  RadMon.getButton = function (id) {
    const el = document.getElementById(id);
    return el instanceof HTMLButtonElement ? el : null;
  };

  /**
   * @param {boolean} ok
   * @param {string} customText
   */
  RadMon.setConnectionStatus = function (ok, customText) {
    RadMon.state.connected = ok;
    const pill = document.getElementById('backendConnPill');
    if (!pill) return;
    if (ok) {
      pill.className = 'meta-pill conn-ok';
      pill.textContent = customText || 'BACKEND: OPENMC STREAM ONLINE';
    } else {
      pill.className = 'meta-pill conn-err';
      pill.textContent = customText || 'BACKEND: OFFLINE (RUN python3 server.py)';
    }
  };

  /**
   * @param {string} path
   * @param {Record<string, any>} payload
   */
  RadMon.apiPost = async function (path, payload) {
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
      RadMon.setConnectionStatus(false, '');
      return null;
    }
  };
})();

