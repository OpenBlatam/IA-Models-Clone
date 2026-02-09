'use client';

import { useState, useEffect } from 'react';
import { Activity, TrendingUp, Zap, Cpu } from 'lucide-react';

interface Metric {
  name: string;
  value: number;
  unit: string;
  trend: 'up' | 'down' | 'stable';
  icon: React.ReactNode;
}

export default function RealTimeDashboard() {
  const [metrics, setMetrics] = useState<Metric[]>([
    {
      name: 'Velocidad',
      value: 0.65,
      unit: 'm/s',
      trend: 'up',
      icon: <Activity className="w-5 h-5" />,
    },
    {
      name: 'Energía',
      value: 85,
      unit: '%',
      trend: 'stable',
      icon: <Zap className="w-5 h-5" />,
    },
    {
      name: 'CPU',
      value: 42,
      unit: '%',
      trend: 'down',
      icon: <Cpu className="w-5 h-5" />,
    },
    {
      name: 'Rendimiento',
      value: 92,
      unit: '%',
      trend: 'up',
      icon: <TrendingUp className="w-5 h-5" />,
    },
  ]);

  useEffect(() => {
    const interval = setInterval(() => {
      setMetrics((prev) =>
        prev.map((m) => ({
          ...m,
          value: m.value + (Math.random() - 0.5) * 0.1,
          trend: Math.random() > 0.5 ? 'up' : 'down',
        }))
      );
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  const getTrendColor = (trend: string) => {
    switch (trend) {
      case 'up':
        return 'text-green-400';
      case 'down':
        return 'text-red-400';
      default:
        return 'text-gray-400';
    }
  };

  return (
    <div className="space-y-6">
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
        <div className="flex items-center gap-2 mb-6">
          <Activity className="w-5 h-5 text-primary-400" />
          <h3 className="text-lg font-semibold text-white">Dashboard en Tiempo Real</h3>
        </div>

        {/* Metrics Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {metrics.map((metric, index) => (
            <div
              key={index}
              className="p-4 bg-gray-700/50 rounded-lg border border-gray-600 hover:border-primary-500/50 transition-colors"
            >
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2 text-primary-400">
                  {metric.icon}
                </div>
                <span className={`text-sm ${getTrendColor(metric.trend)}`}>
                  {metric.trend === 'up' ? '↑' : metric.trend === 'down' ? '↓' : '→'}
                </span>
              </div>
              <h4 className="text-sm text-gray-400 mb-1">{metric.name}</h4>
              <div className="flex items-baseline gap-2">
                <p className="text-2xl font-bold text-white">
                  {metric.value.toFixed(metric.unit === '%' ? 0 : 2)}
                </p>
                <span className="text-sm text-gray-400">{metric.unit}</span>
              </div>
            </div>
          ))}
        </div>

        {/* Real-time Chart Placeholder */}
        <div className="mt-6 p-4 bg-gray-700/50 rounded-lg border border-gray-600">
          <h4 className="text-sm font-medium text-white mb-3">Gráfico en Tiempo Real</h4>
          <div className="h-64 bg-gray-800 rounded-lg flex items-center justify-center">
            <p className="text-gray-400">Gráfico de métricas en tiempo real</p>
          </div>
        </div>
      </div>
    </div>
  );
}


