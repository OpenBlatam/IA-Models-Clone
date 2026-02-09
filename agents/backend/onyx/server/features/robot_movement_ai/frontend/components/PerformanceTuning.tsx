'use client';

import { useState } from 'react';
import { Gauge, Zap, TrendingUp, Settings } from 'lucide-react';
import { toast } from '@/lib/utils/toast';

interface TuningOption {
  id: string;
  name: string;
  description: string;
  current: number;
  recommended: number;
  unit: string;
}

export default function PerformanceTuning() {
  const [options, setOptions] = useState<TuningOption[]>([
    {
      id: '1',
      name: 'Intervalo de Polling',
      description: 'Frecuencia de actualización de datos',
      current: 1000,
      recommended: 500,
      unit: 'ms',
    },
    {
      id: '2',
      name: 'Tamaño de Cache',
      description: 'Cantidad de datos en caché',
      current: 50,
      recommended: 100,
      unit: 'MB',
    },
    {
      id: '3',
      name: 'Timeout de Conexión',
      description: 'Tiempo máximo de espera para conexiones',
      current: 5000,
      recommended: 3000,
      unit: 'ms',
    },
  ]);

  const handleApply = (id: string) => {
    setOptions((prev) =>
      prev.map((o) =>
        o.id === id ? { ...o, current: o.recommended } : o
      )
    );
    toast.success('Optimización aplicada');
  };

  const handleApplyAll = () => {
    setOptions((prev) =>
      prev.map((o) => ({ ...o, current: o.recommended }))
    );
    toast.success('Todas las optimizaciones aplicadas');
  };

  return (
    <div className="space-y-6">
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-2">
            <Gauge className="w-5 h-5 text-primary-400" />
            <h3 className="text-lg font-semibold text-white">Ajuste de Rendimiento</h3>
          </div>
          <button
            onClick={handleApplyAll}
            className="px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg transition-colors flex items-center gap-2"
          >
            <Zap className="w-4 h-4" />
            Aplicar Todo
          </button>
        </div>

        {/* Tuning Options */}
        <div className="space-y-4">
          {options.map((option) => {
            const needsOptimization = option.current !== option.recommended;
            return (
              <div
                key={option.id}
                className={`p-4 rounded-lg border ${
                  needsOptimization
                    ? 'bg-yellow-500/10 border-yellow-500/50'
                    : 'bg-gray-700/50 border-gray-600'
                }`}
              >
                <div className="flex items-start justify-between mb-3">
                  <div className="flex-1">
                    <h4 className="font-semibold text-white mb-1">{option.name}</h4>
                    <p className="text-sm text-gray-300 mb-2">{option.description}</p>
                    <div className="flex items-center gap-4 text-sm">
                      <div>
                        <span className="text-gray-400">Actual: </span>
                        <span className="text-white font-semibold">
                          {option.current} {option.unit}
                        </span>
                      </div>
                      <div>
                        <span className="text-gray-400">Recomendado: </span>
                        <span className="text-green-400 font-semibold">
                          {option.recommended} {option.unit}
                        </span>
                      </div>
                    </div>
                  </div>
                  {needsOptimization && (
                    <button
                      onClick={() => handleApply(option.id)}
                      className="px-3 py-1 bg-green-600 hover:bg-green-700 text-white text-sm rounded-lg transition-colors flex items-center gap-2"
                    >
                      <TrendingUp className="w-3 h-3" />
                      Aplicar
                    </button>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}


