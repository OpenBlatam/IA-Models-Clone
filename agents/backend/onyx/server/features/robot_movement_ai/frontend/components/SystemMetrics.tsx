'use client';

import { useState, useEffect } from 'react';
import { BarChart3, Cpu, HardDrive, MemoryStick, Wifi } from 'lucide-react';

interface Metric {
  name: string;
  icon: React.ReactNode;
  value: number;
  max: number;
  unit: string;
  color: string;
}

export default function SystemMetrics() {
  const [metrics, setMetrics] = useState<Metric[]>([
    {
      name: 'CPU',
      icon: <Cpu className="w-5 h-5" />,
      value: 45,
      max: 100,
      unit: '%',
      color: 'bg-blue-500',
    },
    {
      name: 'Memoria',
      icon: <MemoryStick className="w-5 h-5" />,
      value: 62,
      max: 100,
      unit: '%',
      color: 'bg-green-500',
    },
    {
      name: 'Disco',
      icon: <HardDrive className="w-5 h-5" />,
      value: 38,
      max: 100,
      unit: '%',
      color: 'bg-yellow-500',
    },
    {
      name: 'Red',
      icon: <Wifi className="w-5 h-5" />,
      value: 78,
      max: 100,
      unit: '%',
      color: 'bg-purple-500',
    },
  ]);

  useEffect(() => {
    const interval = setInterval(() => {
      setMetrics((prev) =>
        prev.map((m) => ({
          ...m,
          value: Math.max(0, Math.min(100, m.value + (Math.random() - 0.5) * 5)),
        }))
      );
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  const getBarColor = (value: number) => {
    if (value >= 80) return 'bg-red-500';
    if (value >= 60) return 'bg-yellow-500';
    return 'bg-green-500';
  };

  return (
    <div className="space-y-6">
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
        <div className="flex items-center gap-2 mb-6">
          <BarChart3 className="w-5 h-5 text-primary-400" />
          <h3 className="text-lg font-semibold text-white">Métricas del Sistema</h3>
        </div>

        {/* Metrics Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {metrics.map((metric, index) => {
            const percentage = (metric.value / metric.max) * 100;
            return (
              <div
                key={index}
                className="p-4 bg-gray-700/50 rounded-lg border border-gray-600"
              >
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <div className="text-primary-400">{metric.icon}</div>
                    <h4 className="font-semibold text-white">{metric.name}</h4>
                  </div>
                  <span className="text-lg font-bold text-white">
                    {metric.value.toFixed(0)}{metric.unit}
                  </span>
                </div>
                <div className="w-full bg-gray-700 rounded-full h-2 mb-2">
                  <div
                    className={`h-2 rounded-full transition-all ${getBarColor(percentage)}`}
                    style={{ width: `${percentage}%` }}
                  />
                </div>
                <div className="flex items-center justify-between text-xs text-gray-400">
                  <span>0{metric.unit}</span>
                  <span>{metric.max}{metric.unit}</span>
                </div>
              </div>
            );
          })}
        </div>

        {/* Summary */}
        <div className="mt-6 p-4 bg-gray-700/50 rounded-lg border border-gray-600">
          <h4 className="text-sm font-medium text-white mb-3">Resumen del Sistema</h4>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div>
              <p className="text-gray-400">Uso Promedio</p>
              <p className="text-white font-semibold">
                {(metrics.reduce((acc, m) => acc + m.value, 0) / metrics.length).toFixed(1)}%
              </p>
            </div>
            <div>
              <p className="text-gray-400">Recursos Críticos</p>
              <p className="text-white font-semibold">
                {metrics.filter((m) => m.value >= 80).length}
              </p>
            </div>
            <div>
              <p className="text-gray-400">Recursos Normales</p>
              <p className="text-white font-semibold">
                {metrics.filter((m) => m.value < 80 && m.value >= 60).length}
              </p>
            </div>
            <div>
              <p className="text-gray-400">Recursos Óptimos</p>
              <p className="text-white font-semibold">
                {metrics.filter((m) => m.value < 60).length}
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}


