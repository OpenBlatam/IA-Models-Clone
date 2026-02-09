'use client';

import { useState } from 'react';
import { Gauge, Play, StopCircle, BarChart3 } from 'lucide-react';
import { toast } from '@/lib/utils/toast';

interface ProfileResult {
  function: string;
  calls: number;
  totalTime: number;
  averageTime: number;
  percentage: number;
}

export default function PerformanceProfiler() {
  const [isProfiling, setIsProfiling] = useState(false);
  const [results, setResults] = useState<ProfileResult[]>([
    {
      function: 'moveRobot',
      calls: 1250,
      totalTime: 12500,
      averageTime: 10,
      percentage: 45,
    },
    {
      function: 'updateStatus',
      calls: 5432,
      totalTime: 5432,
      averageTime: 1,
      percentage: 20,
    },
    {
      function: 'processData',
      calls: 890,
      totalTime: 8900,
      averageTime: 10,
      percentage: 32,
    },
  ]);

  const handleStart = () => {
    setIsProfiling(true);
    toast.info('Iniciando perfilado...');
  };

  const handleStop = () => {
    setIsProfiling(false);
    toast.success('Perfilado detenido');
  };

  return (
    <div className="space-y-6">
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-2">
            <Gauge className="w-5 h-5 text-primary-400" />
            <h3 className="text-lg font-semibold text-white">Perfilador de Rendimiento</h3>
          </div>
          {!isProfiling ? (
            <button
              onClick={handleStart}
              className="px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg transition-colors flex items-center gap-2"
            >
              <Play className="w-4 h-4" />
              Iniciar
            </button>
          ) : (
            <button
              onClick={handleStop}
              className="px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg transition-colors flex items-center gap-2"
            >
              <StopCircle className="w-4 h-4" />
              Detener
            </button>
          )}
        </div>

        {/* Results */}
        <div className="space-y-3">
          {results.map((result, index) => (
            <div
              key={index}
              className="p-4 bg-gray-700/50 rounded-lg border border-gray-600"
            >
              <div className="flex items-center justify-between mb-2">
                <div className="flex-1">
                  <h4 className="font-semibold text-white mb-1">{result.function}</h4>
                  <div className="flex items-center gap-4 text-sm text-gray-300">
                    <span>Llamadas: {result.calls.toLocaleString()}</span>
                    <span>Tiempo Total: {result.totalTime}ms</span>
                    <span>Promedio: {result.averageTime}ms</span>
                  </div>
                </div>
                <div className="text-right">
                  <p className="text-lg font-bold text-primary-400">{result.percentage}%</p>
                </div>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-2 mt-2">
                <div
                  className="bg-primary-600 h-2 rounded-full transition-all"
                  style={{ width: `${result.percentage}%` }}
                />
              </div>
            </div>
          ))}
        </div>

        {/* Summary */}
        <div className="mt-6 p-4 bg-gray-700/50 rounded-lg border border-gray-600">
          <h4 className="text-sm font-medium text-white mb-3 flex items-center gap-2">
            <BarChart3 className="w-4 h-4" />
            Resumen
          </h4>
          <div className="grid grid-cols-3 gap-4 text-sm">
            <div>
              <p className="text-gray-400">Total Llamadas</p>
              <p className="text-white font-semibold">
                {results.reduce((acc, r) => acc + r.calls, 0).toLocaleString()}
              </p>
            </div>
            <div>
              <p className="text-gray-400">Tiempo Total</p>
              <p className="text-white font-semibold">
                {results.reduce((acc, r) => acc + r.totalTime, 0)}ms
              </p>
            </div>
            <div>
              <p className="text-gray-400">Funciones</p>
              <p className="text-white font-semibold">{results.length}</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}


