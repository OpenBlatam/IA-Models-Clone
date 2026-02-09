'use client';

import { useState } from 'react';
import { Play, Pause, Square, RotateCcw } from 'lucide-react';
import { toast } from '@/lib/utils/toast';

interface TrajectoryPoint {
  x: number;
  y: number;
  z: number;
  time: number;
}

export default function TrajectorySimulator() {
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentPoint, setCurrentPoint] = useState(0);
  const [speed, setSpeed] = useState(1);
  const [trajectory] = useState<TrajectoryPoint[]>([
    { x: 0, y: 0, z: 0, time: 0 },
    { x: 0.5, y: 0.3, z: 0.2, time: 1 },
    { x: 0.8, y: 0.5, z: 0.4, time: 2 },
    { x: 1.0, y: 0.7, z: 0.3, time: 3 },
    { x: 0.5, y: 0.5, z: 0.5, time: 4 },
  ]);

  const handlePlay = () => {
    setIsPlaying(true);
    toast.info('Simulación iniciada');
  };

  const handlePause = () => {
    setIsPlaying(false);
    toast.info('Simulación pausada');
  };

  const handleStop = () => {
    setIsPlaying(false);
    setCurrentPoint(0);
    toast.info('Simulación detenida');
  };

  const handleReset = () => {
    setCurrentPoint(0);
    setIsPlaying(false);
    toast.info('Simulación reiniciada');
  };

  const currentPosition = trajectory[currentPoint] || trajectory[0];

  return (
    <div className="space-y-6">
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
        <div className="flex items-center gap-2 mb-6">
          <Play className="w-5 h-5 text-primary-400" />
          <h3 className="text-lg font-semibold text-white">Simulador de Trayectorias</h3>
        </div>

        {/* Current Position */}
        <div className="mb-6 p-4 bg-gray-700/50 rounded-lg border border-gray-600">
          <h4 className="text-sm font-medium text-gray-300 mb-3">Posición Actual:</h4>
          <div className="grid grid-cols-3 gap-4">
            <div>
              <p className="text-xs text-gray-400 mb-1">X</p>
              <p className="text-lg font-semibold text-white">{currentPosition.x.toFixed(3)}</p>
            </div>
            <div>
              <p className="text-xs text-gray-400 mb-1">Y</p>
              <p className="text-lg font-semibold text-white">{currentPosition.y.toFixed(3)}</p>
            </div>
            <div>
              <p className="text-xs text-gray-400 mb-1">Z</p>
              <p className="text-lg font-semibold text-white">{currentPosition.z.toFixed(3)}</p>
            </div>
          </div>
        </div>

        {/* Controls */}
        <div className="mb-6 flex items-center gap-4">
          <div className="flex gap-2">
            {!isPlaying ? (
              <button
                onClick={handlePlay}
                className="px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg transition-colors flex items-center gap-2"
              >
                <Play className="w-4 h-4" />
                Reproducir
              </button>
            ) : (
              <button
                onClick={handlePause}
                className="px-4 py-2 bg-yellow-600 hover:bg-yellow-700 text-white rounded-lg transition-colors flex items-center gap-2"
              >
                <Pause className="w-4 h-4" />
                Pausar
              </button>
            )}
            <button
              onClick={handleStop}
              className="px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg transition-colors flex items-center gap-2"
            >
              <Square className="w-4 h-4" />
              Detener
            </button>
            <button
              onClick={handleReset}
              className="px-4 py-2 bg-gray-600 hover:bg-gray-700 text-white rounded-lg transition-colors flex items-center gap-2"
            >
              <RotateCcw className="w-4 h-4" />
              Reiniciar
            </button>
          </div>
          <div className="flex-1">
            <label className="block text-sm text-gray-300 mb-2">Velocidad: {speed}x</label>
            <input
              type="range"
              min="0.1"
              max="5"
              step="0.1"
              value={speed}
              onChange={(e) => setSpeed(parseFloat(e.target.value))}
              className="w-full"
            />
          </div>
        </div>

        {/* Progress */}
        <div className="mb-6">
          <div className="flex items-center justify-between text-sm text-gray-400 mb-2">
            <span>Progreso</span>
            <span>{currentPoint + 1} / {trajectory.length}</span>
          </div>
          <div className="w-full bg-gray-700 rounded-full h-2">
            <div
              className="bg-primary-600 h-2 rounded-full transition-all"
              style={{ width: `${((currentPoint + 1) / trajectory.length) * 100}%` }}
            />
          </div>
        </div>

        {/* Trajectory Points */}
        <div className="p-4 bg-gray-700/50 rounded-lg border border-gray-600">
          <h4 className="text-sm font-medium text-gray-300 mb-3">Puntos de Trayectoria:</h4>
          <div className="space-y-2 max-h-48 overflow-y-auto">
            {trajectory.map((point, index) => (
              <div
                key={index}
                className={`p-2 rounded ${
                  index === currentPoint
                    ? 'bg-primary-600/20 border border-primary-500'
                    : 'bg-gray-600/30'
                }`}
              >
                <div className="flex items-center justify-between text-sm">
                  <span className="text-white">Punto {index + 1}</span>
                  <span className="text-gray-400">
                    ({point.x.toFixed(2)}, {point.y.toFixed(2)}, {point.z.toFixed(2)})
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}


