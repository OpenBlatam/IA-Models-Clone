'use client';

import { useState, useEffect } from 'react';
import { useRobotStore } from '@/lib/store/robotStore';
import { Gamepad2, ArrowUp, ArrowDown, ArrowLeft, ArrowRight, RotateCcw } from 'lucide-react';
import { toast } from '@/lib/utils/toast';

export default function RemoteControl() {
  const { moveTo, currentPosition } = useRobotStore();
  const [isActive, setIsActive] = useState(false);
  const [step, setStep] = useState(0.1);

  const handleMove = (direction: 'forward' | 'backward' | 'left' | 'right' | 'up' | 'down') => {
    if (!currentPosition) {
      toast.error('No hay posición actual disponible');
      return;
    }

    const newPosition = { ...currentPosition };
    switch (direction) {
      case 'forward':
        newPosition.y += step;
        break;
      case 'backward':
        newPosition.y -= step;
        break;
      case 'left':
        newPosition.x -= step;
        break;
      case 'right':
        newPosition.x += step;
        break;
      case 'up':
        newPosition.z += step;
        break;
      case 'down':
        newPosition.z -= step;
        break;
    }
    moveTo(newPosition);
  };

  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      if (!isActive) return;

      switch (e.key) {
        case 'ArrowUp':
          e.preventDefault();
          handleMove('forward');
          break;
        case 'ArrowDown':
          e.preventDefault();
          handleMove('backward');
          break;
        case 'ArrowLeft':
          e.preventDefault();
          handleMove('left');
          break;
        case 'ArrowRight':
          e.preventDefault();
          handleMove('right');
          break;
        case 'w':
        case 'W':
          e.preventDefault();
          handleMove('up');
          break;
        case 's':
        case 'S':
          e.preventDefault();
          handleMove('down');
          break;
      }
    };

    if (isActive) {
      window.addEventListener('keydown', handleKeyPress);
    }

    return () => {
      window.removeEventListener('keydown', handleKeyPress);
    };
  }, [isActive, step, currentPosition]);

  return (
    <div className="space-y-6">
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-2">
            <Gamepad2 className="w-5 h-5 text-primary-400" />
            <h3 className="text-lg font-semibold text-white">Control Remoto</h3>
          </div>
          <div className="flex items-center gap-4">
            <label className="flex items-center gap-2">
              <span className="text-sm text-gray-300">Paso:</span>
              <input
                type="number"
                step="0.01"
                min="0.01"
                max="1"
                value={step}
                onChange={(e) => setStep(parseFloat(e.target.value) || 0.1)}
                className="w-20 px-2 py-1 bg-gray-700 border border-gray-600 rounded text-white text-sm"
              />
            </label>
            <button
              onClick={() => setIsActive(!isActive)}
              className={`px-4 py-2 rounded-lg transition-colors ${
                isActive
                  ? 'bg-red-600 hover:bg-red-700 text-white'
                  : 'bg-primary-600 hover:bg-primary-700 text-white'
              }`}
            >
              {isActive ? 'Desactivar' : 'Activar'}
            </button>
          </div>
        </div>

        {isActive && (
          <div className="mb-4 p-4 bg-yellow-500/10 border border-yellow-500/50 rounded-lg">
            <p className="text-sm text-yellow-400">
              Control activo. Usa las flechas del teclado o los botones para mover el robot.
            </p>
          </div>
        )}

        {/* Control Pad */}
        <div className="flex flex-col items-center gap-4">
          {/* Up */}
          <button
            onClick={() => handleMove('up')}
            disabled={!isActive}
            className="p-4 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <ArrowUp className="w-6 h-6 text-white" />
          </button>

          {/* Middle Row */}
          <div className="flex gap-4">
            <button
              onClick={() => handleMove('left')}
              disabled={!isActive}
              className="p-4 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <ArrowLeft className="w-6 h-6 text-white" />
            </button>
            <button
              onClick={() => handleMove('down')}
              disabled={!isActive}
              className="p-4 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <ArrowDown className="w-6 h-6 text-white" />
            </button>
            <button
              onClick={() => handleMove('right')}
              disabled={!isActive}
              className="p-4 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <ArrowRight className="w-6 h-6 text-white" />
            </button>
          </div>

          {/* Forward/Backward */}
          <div className="flex gap-4 mt-4">
            <button
              onClick={() => handleMove('forward')}
              disabled={!isActive}
              className="px-6 py-3 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed text-white font-medium"
            >
              Adelante
            </button>
            <button
              onClick={() => handleMove('backward')}
              disabled={!isActive}
              className="px-6 py-3 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed text-white font-medium"
            >
              Atrás
            </button>
          </div>

          {/* Instructions */}
          <div className="mt-6 p-4 bg-gray-700/50 rounded-lg">
            <h4 className="text-sm font-semibold text-white mb-2">Controles:</h4>
            <ul className="text-xs text-gray-300 space-y-1">
              <li>• Flechas: Mover en X/Y</li>
              <li>• W/S: Mover en Z (arriba/abajo)</li>
              <li>• Paso configurable</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}


