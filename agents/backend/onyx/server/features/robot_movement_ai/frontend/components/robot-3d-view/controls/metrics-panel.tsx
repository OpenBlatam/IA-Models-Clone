/**
 * Metrics Panel Component
 * @module robot-3d-view/controls/metrics-panel
 */

'use client';

import { memo, useState, useEffect } from 'react';
import { metricsManager, type Metric } from '../utils/metrics';
import { formatNumber, formatDuration } from '../utils/ui-enhancements';

/**
 * Metrics Panel Component
 * 
 * Displays advanced metrics and statistics.
 * 
 * @returns Metrics panel component
 */
export const MetricsPanel = memo(() => {
  const [isOpen, setIsOpen] = useState(false);
  const [metrics, setMetrics] = useState<Metric[]>([]);

  useEffect(() => {
    if (!isOpen) return;

    const updateMetrics = () => {
      setMetrics(metricsManager.getAllMetrics());
    };

    updateMetrics();
    const interval = setInterval(updateMetrics, 1000);
    return () => clearInterval(interval);
  }, [isOpen]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'm') {
        e.preventDefault();
        setIsOpen(!isOpen);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen]);

  if (!isOpen) {
    return (
      <button
        onClick={() => setIsOpen(true)}
        className="absolute bottom-4 right-32 z-50 px-3 py-2 bg-gray-800/95 backdrop-blur-md hover:bg-gray-700/95 border border-gray-700/50 rounded-lg text-white text-xs font-medium transition-all shadow-lg"
        title="Open metrics panel (Ctrl+M)"
        aria-label="Open metrics panel"
      >
        📊 Metrics
      </button>
    );
  }

  return (
    <div
      className="absolute inset-0 z-[100] flex items-center justify-center bg-black/50 backdrop-blur-sm"
      onClick={() => setIsOpen(false)}
      role="dialog"
      aria-modal="true"
      aria-label="Metrics panel"
    >
      <div
        className="bg-gray-800/95 backdrop-blur-md border border-gray-700/50 rounded-lg p-6 max-w-4xl w-full mx-4 max-h-[80vh] overflow-y-auto shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-bold text-white">Metrics & Statistics</h2>
          <button
            onClick={() => setIsOpen(false)}
            className="text-gray-400 hover:text-white transition-colors"
            aria-label="Close metrics panel"
          >
            ✕
          </button>
        </div>

        <div className="space-y-4">
          {metrics.length === 0 ? (
            <div className="text-gray-400 text-center py-8">No metrics available</div>
          ) : (
            metrics.map((metric) => {
              const lastValue = metricsManager.getLastValue(metric.name);
              const average = metricsManager.getAverage(metric.name, 60000); // Last minute
              const min = metricsManager.getMin(metric.name, 60000);
              const max = metricsManager.getMax(metric.name, 60000);
              const p95 = metricsManager.getPercentile(metric.name, 95, 60000);

              return (
                <div
                  key={metric.name}
                  className="p-4 bg-gray-700/50 rounded border border-gray-600"
                >
                  <div className="flex items-center justify-between mb-2">
                    <h3 className="font-semibold text-white">{metric.name}</h3>
                    <span className="text-xs text-gray-400">{metric.type}</span>
                  </div>
                  {metric.description && (
                    <p className="text-sm text-gray-400 mb-3">{metric.description}</p>
                  )}

                  <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                    <div>
                      <div className="text-xs text-gray-400">Current</div>
                      <div className="text-lg font-bold text-white">
                        {lastValue !== undefined
                          ? formatNumber(lastValue, 2)
                          : 'N/A'}
                        {metric.unit && <span className="text-sm text-gray-400 ml-1">{metric.unit}</span>}
                      </div>
                    </div>
                    {average !== undefined && (
                      <div>
                        <div className="text-xs text-gray-400">Average</div>
                        <div className="text-lg font-bold text-white">
                          {formatNumber(average, 2)}
                          {metric.unit && <span className="text-sm text-gray-400 ml-1">{metric.unit}</span>}
                        </div>
                      </div>
                    )}
                    {min !== undefined && max !== undefined && (
                      <>
                        <div>
                          <div className="text-xs text-gray-400">Min</div>
                          <div className="text-lg font-bold text-white">
                            {formatNumber(min, 2)}
                            {metric.unit && <span className="text-sm text-gray-400 ml-1">{metric.unit}</span>}
                          </div>
                        </div>
                        <div>
                          <div className="text-xs text-gray-400">Max</div>
                          <div className="text-lg font-bold text-white">
                            {formatNumber(max, 2)}
                            {metric.unit && <span className="text-sm text-gray-400 ml-1">{metric.unit}</span>}
                          </div>
                        </div>
                      </>
                    )}
                    {p95 !== undefined && (
                      <div>
                        <div className="text-xs text-gray-400">P95</div>
                        <div className="text-lg font-bold text-white">
                          {formatNumber(p95, 2)}
                          {metric.unit && <span className="text-sm text-gray-400 ml-1">{metric.unit}</span>}
                        </div>
                      </div>
                    )}
                  </div>

                  <div className="mt-2 text-xs text-gray-500">
                    {metric.values.length} values recorded
                  </div>
                </div>
              );
            })
          )}
        </div>

        <div className="mt-4 flex gap-2">
          <button
            onClick={() => {
              metricsManager.clear();
              setMetrics([]);
            }}
            className="px-4 py-2 bg-red-600 hover:bg-red-700 rounded text-white text-sm transition-colors"
          >
            Clear All Metrics
          </button>
          <button
            onClick={() => {
              const exported = metricsManager.export();
              const blob = new Blob([exported], { type: 'application/json' });
              const url = URL.createObjectURL(blob);
              const link = document.createElement('a');
              link.href = url;
              link.download = `metrics-${Date.now()}.json`;
              document.body.appendChild(link);
              link.click();
              document.body.removeChild(link);
              URL.revokeObjectURL(url);
            }}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded text-white text-sm transition-colors"
          >
            Export Metrics
          </button>
        </div>

        <div className="mt-2 text-xs text-gray-400">
          Press <kbd className="px-1 py-0.5 bg-gray-700 rounded">Ctrl+M</kbd> to toggle
        </div>
      </div>
    </div>
  );
});

MetricsPanel.displayName = 'MetricsPanel';



