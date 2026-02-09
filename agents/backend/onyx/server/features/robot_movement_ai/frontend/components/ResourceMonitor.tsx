'use client';

import { useState, useEffect } from 'react';
import { Monitor, Cpu, HardDrive, Wifi } from 'lucide-react';

interface Resource {
  name: string;
  icon: React.ReactNode;
  current: number;
  max: number;
  unit: string;
  color: string;
}

export default function ResourceMonitor() {
  const [resources, setResources] = useState<Resource[]>([
    {
      name: 'CPU',
      icon: <Cpu className="w-5 h-5" />,
      current: 45,
      max: 100,
      unit: '%',
      color: 'bg-blue-500',
    },
    {
      name: 'Memoria',
      icon: <Monitor className="w-5 h-5" />,
      current: 62,
      max: 100,
      unit: '%',
      color: 'bg-green-500',
    },
    {
      name: 'Disco',
      icon: <HardDrive className="w-5 h-5" />,
      current: 38,
      max: 100,
      unit: '%',
      color: 'bg-yellow-500',
    },
    {
      name: 'Red',
      icon: <Wifi className="w-5 h-5" />,
      current: 78,
      max: 100,
      unit: '%',
      color: 'bg-purple-500',
    },
  ]);

  useEffect(() => {
    const interval = setInterval(() => {
      setResources((prev) =>
        prev.map((r) => ({
          ...r,
          current: Math.max(0, Math.min(100, r.current + (Math.random() - 0.5) * 5)),
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
          <Monitor className="w-5 h-5 text-primary-400" />
          <h3 className="text-lg font-semibold text-white">Monitor de Recursos</h3>
        </div>

        {/* Resources Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {resources.map((resource, index) => {
            const percentage = (resource.current / resource.max) * 100;
            return (
              <div
                key={index}
                className="p-4 bg-gray-700/50 rounded-lg border border-gray-600"
              >
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <div className="text-primary-400">{resource.icon}</div>
                    <h4 className="font-semibold text-white">{resource.name}</h4>
                  </div>
                  <span className="text-lg font-bold text-white">
                    {resource.current.toFixed(0)}{resource.unit}
                  </span>
                </div>
                <div className="w-full bg-gray-700 rounded-full h-2 mb-2">
                  <div
                    className={`h-2 rounded-full transition-all ${getBarColor(percentage)}`}
                    style={{ width: `${percentage}%` }}
                  />
                </div>
                <div className="flex items-center justify-between text-xs text-gray-400">
                  <span>0{resource.unit}</span>
                  <span>{resource.max}{resource.unit}</span>
                </div>
              </div>
            );
          })}
        </div>

        {/* Summary */}
        <div className="mt-6 p-4 bg-gray-700/50 rounded-lg border border-gray-600">
          <h4 className="text-sm font-medium text-white mb-3">Resumen</h4>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div>
              <p className="text-gray-400">Uso Promedio</p>
              <p className="text-white font-semibold">
                {(resources.reduce((acc, r) => acc + r.current, 0) / resources.length).toFixed(1)}%
              </p>
            </div>
            <div>
              <p className="text-gray-400">Recursos Críticos</p>
              <p className="text-white font-semibold">
                {resources.filter((r) => r.current >= 80).length}
              </p>
            </div>
            <div>
              <p className="text-gray-400">Recursos Normales</p>
              <p className="text-white font-semibold">
                {resources.filter((r) => r.current < 80 && r.current >= 60).length}
              </p>
            </div>
            <div>
              <p className="text-gray-400">Recursos Óptimos</p>
              <p className="text-white font-semibold">
                {resources.filter((r) => r.current < 60).length}
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}


