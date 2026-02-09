'use client';

import { useState } from 'react';
import { apiClient } from '@/lib/api/client';
import { useRobotStore } from '@/lib/store/robotStore';
import { Sparkles, Play, Download } from 'lucide-react';
import { toast } from '@/lib/utils/toast';

type Algorithm = 'astar' | 'rrt';

export default function TrajectoryOptimizer() {
  const { currentPosition } = useRobotStore();
  const [target, setTarget] = useState({ x: 0.5, y: 0.3, z: 0.2 });
  const [algorithm, setAlgorithm] = useState<Algorithm>('astar');
  const [isOptimizing, setIsOptimizing] = useState(false);
  const [result, setResult] = useState<any>(null);

  const handleOptimize = async () => {
    if (!currentPosition) {
      toast.error('No hay posición actual disponible');
      return;
    }

    setIsOptimizing(true);
    try {
      let data;
      if (algorithm === 'astar') {
        data = await apiClient.optimizeAStar(target, 0.05);
      } else {
        data = await apiClient.optimizeRRT(target, 1000, 0.1);
      }
      setResult(data);
      toast.success(`Trayectoria optimizada con ${algorithm.toUpperCase()}`);
    } catch (error: any) {
      toast.error(`Error: ${error.message || 'Failed to optimize trajectory'}`);
    } finally {
      setIsOptimizing(false);
    }
  };

  const handleExport = async () => {
    if (!result) return;

    try {
      const waypoints = [
        { x: currentPosition!.x, y: currentPosition!.y, z: currentPosition!.z },
        target,
      ];
      const data = await apiClient.exportTrajectory({ waypoints }, 'json');
      toast.success(`Trayectoria exportada: ${data.filepath}`);
    } catch (error: any) {
      toast.error(`Error: ${error.message || 'Failed to export trajectory'}`);
    }
  };

  return (
    <div className="bg-white rounded-lg p-tesla-lg border border-gray-200 shadow-sm">
      <div className="flex items-center gap-tesla-sm mb-tesla-lg">
        <Sparkles className="w-5 h-5 text-tesla-blue" />
        <h3 className="text-lg font-semibold text-tesla-black">Optimización de Trayectoria</h3>
      </div>

      <div className="space-y-tesla-lg">
        {/* Algorithm Selection */}
        <div>
          <label className="block text-sm font-medium text-tesla-black mb-tesla-sm">
            Algoritmo
          </label>
          <div className="flex gap-tesla-lg">
            <label className="flex items-center gap-tesla-sm cursor-pointer group">
              <input
                type="radio"
                value="astar"
                checked={algorithm === 'astar'}
                onChange={(e) => setAlgorithm(e.target.value as Algorithm)}
                className="w-4 h-4 text-tesla-blue focus:ring-2 focus:ring-tesla-blue focus:ring-offset-2"
              />
              <span className="text-tesla-black font-medium group-hover:text-tesla-blue transition-colors">A*</span>
            </label>
            <label className="flex items-center gap-tesla-sm cursor-pointer group">
              <input
                type="radio"
                value="rrt"
                checked={algorithm === 'rrt'}
                onChange={(e) => setAlgorithm(e.target.value as Algorithm)}
                className="w-4 h-4 text-tesla-blue focus:ring-2 focus:ring-tesla-blue focus:ring-offset-2"
              />
              <span className="text-tesla-black font-medium group-hover:text-tesla-blue transition-colors">RRT</span>
            </label>
          </div>
        </div>

        {/* Target Position */}
        <div>
          <label className="block text-sm font-medium text-tesla-black mb-tesla-sm">
            Posición Objetivo
          </label>
          <div className="grid grid-cols-3 gap-tesla-md">
            <div>
              <input
                type="number"
                step="0.01"
                value={target.x}
                onChange={(e) => setTarget({ ...target, x: parseFloat(e.target.value) || 0 })}
                className="w-full px-tesla-md py-tesla-sm bg-white border border-gray-300 rounded-md text-tesla-black focus:outline-none focus:ring-2 focus:ring-tesla-blue focus:border-transparent transition-all"
                placeholder="X"
                aria-label="Coordenada X"
              />
            </div>
            <div>
              <input
                type="number"
                step="0.01"
                value={target.y}
                onChange={(e) => setTarget({ ...target, y: parseFloat(e.target.value) || 0 })}
                className="w-full px-4 py-3 bg-white border border-gray-300 rounded-md text-tesla-black focus:outline-none focus:ring-2 focus:ring-tesla-blue focus:border-transparent transition-all"
                placeholder="Y"
                aria-label="Coordenada Y"
              />
            </div>
            <div>
              <input
                type="number"
                step="0.01"
                value={target.z}
                onChange={(e) => setTarget({ ...target, z: parseFloat(e.target.value) || 0 })}
                className="w-full px-4 py-3 bg-white border border-gray-300 rounded-md text-tesla-black focus:outline-none focus:ring-2 focus:ring-tesla-blue focus:border-transparent transition-all"
                placeholder="Z"
                aria-label="Coordenada Z"
              />
            </div>
          </div>
        </div>

        {/* Actions */}
        <div className="flex gap-tesla-sm">
          <button
            onClick={handleOptimize}
            disabled={isOptimizing || !currentPosition}
            className="flex-1 flex items-center justify-center gap-tesla-sm px-tesla-lg py-tesla-sm bg-tesla-blue hover:bg-opacity-90 text-white font-medium rounded-md transition-all disabled:opacity-50 disabled:cursor-not-allowed min-h-[44px]"
            aria-label={isOptimizing ? 'Optimizando trayectoria' : 'Optimizar trayectoria'}
          >
            <Play className="w-4 h-4" />
            {isOptimizing ? 'Optimizando...' : 'Optimizar'}
          </button>
          {result && (
            <button
              onClick={handleExport}
              className="flex items-center justify-center gap-tesla-sm px-tesla-lg py-tesla-sm bg-white border-2 border-gray-300 hover:border-gray-400 text-tesla-black font-medium rounded-md transition-all min-h-[44px]"
              aria-label="Exportar trayectoria"
            >
              <Download className="w-4 h-4" />
              Exportar
            </button>
          )}
        </div>

        {/* Results */}
        {result && (
          <div className="mt-tesla-md p-tesla-lg bg-gray-50 rounded-md border border-gray-200">
            <h4 className="text-sm font-semibold text-tesla-black mb-tesla-sm">Resultados</h4>
            <div className="space-y-tesla-sm text-sm">
              <div className="flex justify-between items-center">
                <span className="text-tesla-gray-dark">Puntos de trayectoria:</span>
                <span className="text-tesla-black font-semibold font-mono">{result.trajectory_points || 0}</span>
              </div>
              {result.analysis && (
                <>
                  <div className="flex justify-between items-center">
                    <span className="text-tesla-gray-dark">Distancia total:</span>
                    <span className="text-tesla-black font-semibold font-mono">
                      {result.analysis.total_distance?.toFixed(3) || 'N/A'} m
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-tesla-gray-dark">Tiempo estimado:</span>
                    <span className="text-tesla-black font-semibold font-mono">
                      {result.analysis.estimated_time?.toFixed(2) || 'N/A'} s
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-tesla-gray-dark">Sin colisiones:</span>
                    <span
                      className={`font-semibold font-mono ${
                        result.analysis.collision_free ? 'text-green-600' : 'text-red-600'
                      }`}
                    >
                      {result.analysis.collision_free ? 'Sí' : 'No'}
                    </span>
                  </div>
                </>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

