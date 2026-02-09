'use client';

import { useState } from 'react';
import { apiClient } from '@/lib/api/client';
import { useRobotStore } from '@/lib/store/robotStore';
import { GitCompare, Play, Download, Sparkles } from 'lucide-react';
import { toast } from '@/lib/utils/toast';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';

interface TrajectoryData {
  algorithm: string;
  points: number;
  distance: number;
  time: number;
  energy?: number;
  smoothness?: number;
}

export default function TrajectoryComparison() {
  const { currentPosition } = useRobotStore();
  const [target, setTarget] = useState({ x: 0.5, y: 0.3, z: 0.2 });
  const [trajectories, setTrajectories] = useState<TrajectoryData[]>([]);
  const [isComparing, setIsComparing] = useState(false);

  const handleCompare = async () => {
    if (!currentPosition) {
      toast.error('No hay posición actual disponible');
      return;
    }

    setIsComparing(true);
    try {
      const [astarResult, rrtResult] = await Promise.all([
        apiClient.optimizeAStar(target, 0.05),
        apiClient.optimizeRRT(target, 1000, 0.1),
      ]);

      const astarData: TrajectoryData = {
        algorithm: 'A*',
        points: astarResult.trajectory_points || 0,
        distance: astarResult.analysis?.total_distance || 0,
        time: astarResult.analysis?.estimated_time || 0,
        energy: astarResult.analysis?.energy_consumption,
        smoothness: astarResult.analysis?.smoothness,
      };

      const rrtData: TrajectoryData = {
        algorithm: 'RRT',
        points: rrtResult.trajectory_points || 0,
        distance: rrtResult.analysis?.total_distance || 0,
        time: rrtResult.analysis?.estimated_time || 0,
        energy: rrtResult.analysis?.energy_consumption,
        smoothness: rrtResult.analysis?.smoothness,
      };

      setTrajectories([astarData, rrtData]);
      toast.success('Comparación completada');
    } catch (error: any) {
      toast.error(`Error: ${error.message || 'Failed to compare trajectories'}`);
    } finally {
      setIsComparing(false);
    }
  };

  const chartData = trajectories.map((t) => ({
    algorithm: t.algorithm,
    distance: t.distance,
    time: t.time,
    points: t.points,
    energy: t.energy || 0,
    smoothness: t.smoothness || 0,
  }));

  return (
    <div className="space-y-6">
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
        <div className="flex items-center gap-2 mb-6">
          <GitCompare className="w-5 h-5 text-primary-400" />
          <h3 className="text-lg font-semibold text-white">Comparación de Trayectorias</h3>
        </div>

        {/* Target Position */}
        <div className="mb-6">
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
          onClick={handleCompare}
          disabled={isComparing || !currentPosition}
          className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white font-medium rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <Sparkles className="w-4 h-4" />
          {isComparing ? 'Comparando...' : 'Comparar Algoritmos'}
        </button>
      </div>

      {/* Comparison Results */}
      {trajectories.length > 0 && (
        <>
          {/* Comparison Table */}
          <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
            <h4 className="text-lg font-semibold text-white mb-4">Resultados de la Comparación</h4>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-gray-700">
                    <th className="text-left p-3 text-gray-300">Algoritmo</th>
                    <th className="text-left p-3 text-gray-300">Puntos</th>
                    <th className="text-left p-3 text-gray-300">Distancia (m)</th>
                    <th className="text-left p-3 text-gray-300">Tiempo (s)</th>
                    <th className="text-left p-3 text-gray-300">Energía</th>
                    <th className="text-left p-3 text-gray-300">Suavidad</th>
                  </tr>
                </thead>
                <tbody>
                  {trajectories.map((t, index) => (
                    <tr key={index} className="border-b border-gray-700/50">
                      <td className="p-3 text-white font-medium">{t.algorithm}</td>
                      <td className="p-3 text-gray-300">{t.points}</td>
                      <td className="p-3 text-gray-300">{t.distance.toFixed(3)}</td>
                      <td className="p-3 text-gray-300">{t.time.toFixed(2)}</td>
                      <td className="p-3 text-gray-300">{t.energy?.toFixed(2) || 'N/A'}</td>
                      <td className="p-3 text-gray-300">{t.smoothness?.toFixed(2) || 'N/A'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Charts */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
              <h4 className="text-lg font-semibold text-white mb-4">Distancia vs Tiempo</h4>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis dataKey="algorithm" stroke="#9CA3AF" />
                  <YAxis stroke="#9CA3AF" />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151' }}
                  />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="distance"
                    stroke="#0EA5E9"
                    strokeWidth={2}
                    name="Distancia (m)"
                  />
                  <Line
                    type="monotone"
                    dataKey="time"
                    stroke="#10B981"
                    strokeWidth={2}
                    name="Tiempo (s)"
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>

            <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
              <h4 className="text-lg font-semibold text-white mb-4">Puntos vs Energía</h4>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis dataKey="algorithm" stroke="#9CA3AF" />
                  <YAxis stroke="#9CA3AF" />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151' }}
                  />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="points"
                    stroke="#F59E0B"
                    strokeWidth={2}
                    name="Puntos"
                  />
                  <Line
                    type="monotone"
                    dataKey="energy"
                    stroke="#EF4444"
                    strokeWidth={2}
                    name="Energía"
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

