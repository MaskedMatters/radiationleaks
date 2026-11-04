/**
 * frontend/js/charts.js
 * Initializes and updates the 3 Chart.js telemetry charts:
 * 1. Sensor Flux Profile Bar Chart (`#fluxChart`) in Tab 01
 * 2. Epoch Reward Score (%) & Error MAE (m) Convergence Chart (`#rewardCurveChart`) in Tab 02
 * 3. 100-Run Validation CDF Error Distribution Chart (`#cdfChart`) in Tab 02
 */

(function () {
  'use strict';

  /** @type {Record<string, any>} */
  const RadMon = /** @type {any} */ (window).RadMon;
  const { state, setText } = RadMon;

  /** @type {Record<string, any>} */
  const chartStore = {
    cdf: null,
    flux: null,
    rewardCurve: null
  };
  RadMon.chartStore = chartStore;

  RadMon.initCharts = function () {
    const ChartLib = /** @type {any} */ (window)['Chart'];
    if (!ChartLib) return;

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
    ChartLib.defaults.font.size = 11;
    ChartLib.defaults.color = '#475569';

    // 1. CDF Error Distribution Chart
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
              backgroundColor: 'rgba(2, 132, 199, 0.12)',
              borderWidth: 2.2,
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
              labels: { boxWidth: 12, boxHeight: 12, padding: 12, font: { size: 11 } }
            }
          },
          scales: {
            x: {
              title: {
                display: true,
                text: 'AI Localization Error Δ (meters)',
                font: { size: 11, weight: '600' },
                color: '#475569'
              },
              grid: { color: '#f1f5f9' },
              ticks: { maxTicksLimit: 9 }
            },
            y: {
              min: 0,
              max: 100,
              title: {
                display: true,
                text: 'Cumulative %',
                font: { size: 11, weight: '600' },
                color: '#475569'
              },
              grid: { color: '#f1f5f9' },
              ticks: { stepSize: 25 }
            }
          }
        }
      });
    }

    // 2. Sensor Flux Profile Bar Chart
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
              borderRadius: 2
            },
            {
              label: 'AI Predicted Flux E[λ] (cps)',
              data: [],
              backgroundColor: 'rgba(2, 132, 199, 0.30)',
              borderColor: '#0284c7',
              borderWidth: 1,
              borderRadius: 2
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
              labels: { boxWidth: 12, boxHeight: 12, padding: 12, font: { size: 11 } }
            }
          },
          scales: {
            x: { grid: { display: false } },
            y: {
              beginAtZero: true,
              title: {
                display: true,
                text: 'Count Rate (cps)',
                font: { size: 11, weight: '600' },
                color: '#475569'
              },
              grid: { color: '#f1f5f9' }
            }
          }
        }
      });
    }

    // 3. Epoch Reward & Error Convergence Chart
    const rewardCanvas = document.getElementById('rewardCurveChart');
    if (rewardCanvas instanceof HTMLCanvasElement) {
      const rewardCtx = rewardCanvas.getContext('2d');
      chartStore.rewardCurve = new ChartLib(rewardCtx, {
        type: 'line',
        plugins: [whiteBgPlugin],
        data: {
          labels: [],
          datasets: [
            {
              label: 'Reward Score (%)',
              data: [],
              borderColor: '#15803d',
              backgroundColor: 'rgba(21, 128, 61, 0.10)',
              borderWidth: 2.2,
              pointRadius: 0,
              fill: true,
              tension: 0.15,
              yAxisID: 'yReward'
            },
            {
              label: 'Validation Error MAE (m)',
              data: [],
              borderColor: '#dc2626',
              backgroundColor: 'transparent',
              borderWidth: 2.0,
              borderDash: [4, 3],
              pointRadius: 0,
              fill: false,
              tension: 0.15,
              yAxisID: 'yMae'
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
              labels: { boxWidth: 12, boxHeight: 12, padding: 10, font: { size: 11 } }
            }
          },
          scales: {
            x: {
              title: {
                display: true,
                text: 'Training Epoch',
                font: { size: 11, weight: '600' },
                color: '#475569'
              },
              grid: { color: '#f1f5f9' },
              ticks: { maxTicksLimit: 9 }
            },
            yReward: {
              type: 'linear',
              position: 'left',
              min: 0,
              max: 100,
              title: {
                display: true,
                text: 'Reward Score (%)',
                font: { size: 11, weight: '600' },
                color: '#15803d'
              },
              grid: { color: '#f1f5f9' }
            },
            yMae: {
              type: 'linear',
              position: 'right',
              beginAtZero: true,
              title: {
                display: true,
                text: 'Error MAE (m)',
                font: { size: 11, weight: '600' },
                color: '#dc2626'
              },
              grid: { drawOnChartArea: false }
            }
          }
        }
      });
    }
  };

  RadMon.updateRewardCurveChart = function () {
    const pt = state.pathwayTelemetry || {};
    const curve = Array.isArray(pt.epoch_curve) ? pt.epoch_curve : [];

    if (curve.length === 0) {
      setText('epochCurveSummary', '0 EPOCHS RECORDED');
      if (chartStore.rewardCurve) {
        chartStore.rewardCurve.data.labels = [];
        chartStore.rewardCurve.data.datasets[0].data = [];
        chartStore.rewardCurve.data.datasets[1].data = [];
        chartStore.rewardCurve.update();
      }
      return;
    }

    const firstPt = curve[0];
    const lastPt = curve[curve.length - 1];
    setText(
      'epochCurveSummary',
      `EP ${lastPt.epoch} | REWARD: ${Number(lastPt.reward_pct).toFixed(1)}% | MAE: ${Number(firstPt.mae_m).toFixed(2)}m → ${Number(lastPt.mae_m).toFixed(2)}m`
    );

    if (chartStore.rewardCurve) {
      chartStore.rewardCurve.data.labels = curve.map(
        /** @param {Record<string, any>} c */ (c) => `EP ${c.epoch}`
      );
      chartStore.rewardCurve.data.datasets[0].data = curve.map(
        /** @param {Record<string, any>} c */ (c) => Number(c.reward_pct)
      );
      chartStore.rewardCurve.data.datasets[1].data = curve.map(
        /** @param {Record<string, any>} c */ (c) => Number(c.mae_m)
      );
      chartStore.rewardCurve.update();
    }
  };

  RadMon.updateCdfChart = function () {
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
  };

  RadMon.updateFluxChart = function () {
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
  };
})();

