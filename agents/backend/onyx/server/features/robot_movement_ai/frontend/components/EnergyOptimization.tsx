'use client';

import { useState } from 'react';
import { apiClient } from '@/lib/api/client';
import { useRobotStore } from '@/lib/store/robotStore';
import { Battery, Zap, TrendingDown, Play } from 'lucide-react';
import { toast } from '@/lib/utils/toast';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';

export default function EnergyOptimization() {
  const { currentPosition } = useRobotStore();
  const [target, setTarget] = useState({ x: 0.5, y: 0.3, z: 0.2 });
  const [optimization, setOptimization] = useState<any>(null);
  const [isOptimizing, setIsOptimizing] = useState(false);

  const handleOptimize = async () => {
    if (!currentPosition) {
      toast.error('No hay posición actual disponible');
      return;
    }

    setIsOptimizing(true);
    try {
      // Simulate energy optimization
      await new Promise((resolve) => setTimeout(resolve, 2000));

      const distance = Math.sqrt(
        (target.x - currentPosition.x) ** 2 +
        (target.y - currentPosition.y) ** 2 +
        (target.z - currentPosition.z) ** 2
      );

      const optimized = {
        original: {
          energy: distance * 100,
          time: distance * 2,
          smoothness: 0.7,
        },
        optimized: {
          energy: distance * 75,
          time: distance * 2.2,
          smoothness: 0.9,
        },
        savings: {
          energy: ((distance * 100 - distance * 75) / (distance * 100)) * 100,
          time: ((distance * 2 - distance * 2.2) / (distance * 2)) * 100,
        },
      };

      setOptimization(optimized);
      toast.success('Optimización de energía completada');
    } catch (error: any) {
      toast.error(`Error: ${error.message || 'Failed to optimize'}`);
    } finally {
      setIsOptimizing(false);
    }
  };

  const chartData = optimization
    ? [
        {
          name: 'Energía',
          Original: optimization.original.energy,
          Optimizado: optimization.optimized.energy,
        },
        {
          name: 'Tiempo',
          Original: optimization.original.time,
          Optimizado: optimization.optimized.time,
        },
      ]
    : [];

  return (
    <div className="space-y-6">
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
        <div className="flex items-center gap-2 mb-6">
          <Battery className="w-5 h-5 text-primary-400" />
          <h3 className="text-lg font-semibold text-white">Optimización de Energía</h3>
        </div>

        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Posición Objetivo
            </label>
            <div className="grid grid-cols-3 gap-4">
              <input
                type="number"
                step="0.01"
                value={target.x}
                onChange={(e) => setTarget({ ...target, x: parseFloat(e.target.value) || 0 })}
                className="px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-primary-500"
                placeholder="X"
              />
              <input
                type="number"
                step="0.01"
                value={target.y}
                onChange={(e) => setTarget({ ...target, y: parseFloat(e.target.value) || 0 })}
                className="px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-primary-500"
                placeholder="Y"
              />
              <input
                type="number"
                step="0.01"
                value={target.z}
                onChange={(e) => setTarget({ ...target, z: parseFloat(e.target.value) || 0 })}
                className="px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-primary-500"
                placeholder="Z"
              />
            </div>
          </div>

          <button
            onClick={handleOptimize}
            disabled={isOptimizing || !currentPosition}
            className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white font-medium rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Zap className="w-4 h-4" />
            {isOptimizing ? 'Optimizando...' : 'Optimizar Energía'}
          </button>
        </div>
      </div>

      {optimization && (
        <>
          {/* Results */}
          <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
            <h4 className="text-lg font-semibold text-white mb-4">Resultados de Optimización</h4>
            <div className="grid grid-cols-2 gap-6">
              <div>
                <h5 className="text-sm font-medium text-gray-400 mb-3">Original</h5>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-gray-300">Energía:</span>
                    <span className="text-white font-mono">{optimization.original.energy.toFixed(2)} J</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-300">Tiempo:</span>
                    <span className="text-white font-mono">{optimization.original.time.toFixed(2)} s</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-300">Suavidad:</span>
                    <span className="text-white font-mono">{optimization.original.smoothness.toFixed(2)}</span>
                  </div>
                </div>
              </div>
              <div>
                <h5 className="text-sm font-medium text-gray-400 mb-3">Optimizado</h5>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-gray-300">Energía:</span>
                    <span className="text-green-400 font-mono">{optimization.optimized.energy.toFixed(2)} J</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-300">Tiempo:</span>
                    <span className="text-white font-mono">{optimization.optimized.time.toFixed(2)} s</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-300">Suavidad:</span>
                    <span className="text-green-400 font-mono">{optimization.optimized.smoothness.toFixed(2)}</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Savings */}
            <div className="mt-6 p-4 bg-green-500/10 border border-green-500/50 rounded-lg">
              <div className="flex items-center gap-2 mb-2">
                <TrendingDown className="w-5 h-5 text-green-400" />
                <h5 className="font-semibold text-green-400">Ahorro</h5>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <span className="text-gray-300">Energía:</span>
                  <span className="ml-2 text-green-400 font-bold">
                    {optimization.savings.energy.toFixed(1)}%
                  </span>
                </div>
                <div>
                  <span className="text-gray-300">Tiempo:</span>
                  <span className="ml-2 text-white font-bold">
                    {optimization.savings.time.toFixed(1)}%
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Chart */}
          <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
            <h4 className="text-lg font-semibold text-white mb-4">Comparación</h4>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="name" stroke="#9CA3AF" />
                <YAxis stroke="#9CA3AF" />
                <Tooltip
                  contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151' }}
                />
                <Legend />
                <Bar dataKey="Original" fill="#EF4444" />
                <Bar dataKey="Optimizado" fill="#10B981" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </>
      )}
    </div>
  );
}


