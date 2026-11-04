/**
 * frontend/js/tabs.js
 * Tab navigation controller and badge updater for the 4 workspace views.
 */

(function () {
  'use strict';

  /** @type {Record<string, any>} */
  const RadMon = /** @type {any} */ (window).RadMon;
  const { state, setText } = RadMon;

  /**
   * Switches the active workspace tab and refreshes canvases/charts inside the revealed tab.
   * @param {string} tabId
   */
  RadMon.switchTab = function (tabId) {
    state.activeTab = tabId;

    document.querySelectorAll('.tab-btn').forEach((btn) => {
      const target = btn.getAttribute('data-tab');
      if (target === tabId) {
        btn.classList.add('active');
      } else {
        btn.classList.remove('active');
      }
    });

    document.querySelectorAll('.tab-pane').forEach((pane) => {
      if (pane.id === tabId) {
        pane.classList.add('active');
      } else {
        pane.classList.remove('active');
      }
    });

    // Redraw canvases and resize Chart.js instances so hidden-tab canvases render crisply
    window.requestAnimationFrame(() => {
      if (typeof RadMon.renderFacilityCanvas === 'function') {
        RadMon.renderFacilityCanvas();
      }
      if (typeof RadMon.renderNeuralPathwayCanvas === 'function') {
        RadMon.renderNeuralPathwayCanvas();
      }
      if (RadMon.chartStore) {
        for (const key of Object.keys(RadMon.chartStore)) {
          const chart = RadMon.chartStore[key];
          if (chart && typeof chart.resize === 'function') {
            chart.resize();
            chart.update('none');
          }
        }
      }
    });
  };

  /**
   * Updates the live counters displayed in the 4 tab navigation buttons.
   */
  RadMon.updateTabBadges = function () {
    const aiStat = state.aiModelStatus || {};
    const nEpochs = Number(aiStat.trained_epochs || 0);
    const nSamples = Array.isArray(state.trainingSamples)
      ? state.trainingSamples.length
      : Number(aiStat.samples_seen || 0);
    const nRuns = Array.isArray(state.runLogs) ? state.runLogs.length : 0;

    setText(
      'tabBadgeMap',
      state.running ? 'STREAMING' : state.hasRun ? `RUN #${state.frameCount}` : '16 CH'
    );
    setText('tabBadgePathways', `${nEpochs} EP`);
    setText('tabBadgeTraining', `${nSamples} SAMPLES`);
    setText('tabBadgeLogs', `${nRuns} RUNS`);
  };

  RadMon.initTabs = function () {
    document.querySelectorAll('.tab-btn').forEach((btn) => {
      btn.addEventListener('click', () => {
        const target = btn.getAttribute('data-tab');
        if (target) {
          RadMon.switchTab(target);
        }
      });
    });
  };
})();

