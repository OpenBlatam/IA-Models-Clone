'use client';

import { useState, useEffect } from 'react';
import { Monitor, Activity, TrendingUp, AlertCircle } from 'lucide-react';

interface Metric {
  name: string;
  value: number;
  unit: string;
  trend: 'up' | 'down' | 'stable';
  threshold?: { warning: number; critical: number };
}

export default function MonitoringDashboard() {
  const [metrics, setMetrics] = useState<Metric[]>([
    {
      name: 'CPU Usage',
      value: 45,
      unit: '%',
      trend: 'up',
      threshold: { warning: 70, critical: 90 },
    },
    {
      name: 'Memory Usage',
      value: 62,
      unit: '%',
      trend: 'down',
      threshold: { warning: 80, critical: 95 },
    },
    {
      name: 'Disk I/O',
      value: 38,
      unit: 'MB/s',
      trend: 'stable',
    },
    {
      name: 'Network Traffic',
      value: 125,
      unit: 'Mbps',
      trend: 'up',
    },
  ]);

  useEffect(() => {
    const interval = setInterval(() => {
      setMetrics((prev) =>
        prev.map((m) => ({
          ...m,
          value: Math.max(0, Math.min(100, m.value + (Math.random() - 0.5) * 5)),
          trend: Math.random() > 0.5 ? 'up' : 'down',
        }))
      );
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  const getStatusColor = (metric: Metric) => {
    if (metric.threshold) {
      if (metric.value >= metric.threshold.critical) return 'text-red-400';
      if (metric.value >= metric.threshold.warning) return 'text-yellow-400';
    }
    return 'text-green-400';
  };

  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case 'up':
        return <TrendingUp className="w-4 h-4 text-green-400" />;
      case 'down':
        return <TrendingUp className="w-4 h-4 text-red-400 rotate-180" />;
      default:
        return <Activity className="w-4 h-4 text-gray-400" />;
    }
  };

  return (
    <div className="space-y-6">
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
        <div className="flex items-center gap-2 mb-6">
          <Monitor className="w-5 h-5 text-primary-400" />
          <h3 className="text-lg font-semibold text-white">Dashboard de Monitoreo</h3>
        </div>

        {/* Metrics Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {metrics.map((metric, index) => (
            <div
              key={index}
              className="p-4 bg-gray-700/50 rounded-lg border border-gray-600"
            >
              <div className="flex items-center justify-between mb-2">
                <h4 className="text-sm text-gray-300">{metric.name}</h4>
                {getTrendIcon(metric.trend)}
              </div>
              <div className="flex items-baseline gap-2">
                <p className={`text-2xl font-bold ${getStatusColor(metric)}`}>
                  {metric.value.toFixed(metric.unit === '%' ? 0 : 1)}
                </p>
                <span className="text-sm text-gray-400">{metric.unit}</span>
              </div>
              {metric.threshold && (
                <div className="mt-2 text-xs text-gray-400">
                  <span>Warning: {metric.threshold.warning}{metric.unit} </span>
                  <span>Critical: {metric.threshold.critical}{metric.unit}</span>
                </div>
              )}
            </div>
          ))}
        </div>

        {/* Alerts */}
        <div className="mt-6 p-4 bg-yellow-500/10 border border-yellow-500/50 rounded-lg">
          <div className="flex items-center gap-2 mb-2">
            <AlertCircle className="w-4 h-4 text-yellow-400" />
            <h4 className="text-sm font-semibold text-yellow-400">Alertas Activas</h4>
          </div>
          <p className="text-sm text-gray-300">
            {metrics.filter((m) => m.threshold && m.value >= m.threshold.warning).length} métricas
            requieren atención
          </p>
        </div>
      </div>
    </div>
  );
}


