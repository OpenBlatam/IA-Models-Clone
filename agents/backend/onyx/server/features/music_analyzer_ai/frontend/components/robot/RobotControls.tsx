'use client';

import { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { robotApiService, type MoveToRequest } from '@/lib/api/robot-api';
import { Move, Square, AlertTriangle, Navigation } from 'lucide-react';
import toast from 'react-hot-toast';

export function RobotControls() {
  const [position, setPosition] = useState<MoveToRequest>({
    x: 0,
    y: 0,
    z: 0,
  });

  const moveMutation = useMutation({
    mutationFn: (pos: MoveToRequest) => robotApiService.moveTo(pos),
    onSuccess: () => {
      toast.success('Comando de movimiento enviado');
    },
    onError: (error: any) => {
      toast.error(error.response?.data?.detail || 'Error al mover el robot');
    },
  });

  const stopMutation = useMutation({
    mutationFn: () => robotApiService.stop(),
    onSuccess: () => {
      toast.success('Robot detenido');
    },
    onError: (error: any) => {
      toast.error(error.response?.data?.detail || 'Error al detener el robot');
    },
  });

  const emergencyStopMutation = useMutation({
    mutationFn: () => robotApiService.emergencyStop(),
    onSuccess: () => {
      toast.success('Parada de emergencia activada');
    },
    onError: (error: any) => {
      toast.error('Error en parada de emergencia');
    },
  });

  const handleMove = () => {
    moveMutation.mutate(position);
  };

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
      <h2 className="text-xl font-semibold text-white mb-4 flex items-center gap-2">
        <Navigation className="w-6 h-6" />
        Controles de Movimiento
      </h2>

      <div className="space-y-4">
        {/* Position Inputs */}
        <div className="grid grid-cols-3 gap-3">
          <div>
            <label className="block text-sm text-gray-300 mb-1">X</label>
            <input
              type="number"
              step="0.1"
              value={position.x}
              onChange={(e) =>
                setPosition({ ...position, x: parseFloat(e.target.value) || 0 })
              }
              className="w-full px-3 py-2 bg-white/20 border border-white/30 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-green-400"
            />
          </div>
          <div>
            <label className="block text-sm text-gray-300 mb-1">Y</label>
            <input
              type="number"
              step="0.1"
              value={position.y}
              onChange={(e) =>
                setPosition({ ...position, y: parseFloat(e.target.value) || 0 })
              }
              className="w-full px-3 py-2 bg-white/20 border border-white/30 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-green-400"
            />
          </div>
          <div>
            <label className="block text-sm text-gray-300 mb-1">Z</label>
            <input
              type="number"
              step="0.1"
              value={position.z}
              onChange={(e) =>
                setPosition({ ...position, z: parseFloat(e.target.value) || 0 })
              }
              className="w-full px-3 py-2 bg-white/20 border border-white/30 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-green-400"
            />
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex flex-col gap-2">
          <button
            onClick={handleMove}
            disabled={moveMutation.isPending}
            className="flex items-center justify-center gap-2 px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg transition-colors disabled:opacity-50"
          >
            <Move className="w-5 h-5" />
            Mover a Posición
          </button>

          <button
            onClick={() => stopMutation.mutate()}
            disabled={stopMutation.isPending}
            className="flex items-center justify-center gap-2 px-4 py-2 bg-yellow-600 hover:bg-yellow-700 text-white rounded-lg transition-colors disabled:opacity-50"
          >
            <Square className="w-5 h-5" />
            Detener
          </button>

          <button
            onClick={() => emergencyStopMutation.mutate()}
            disabled={emergencyStopMutation.isPending}
            className="flex items-center justify-center gap-2 px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg transition-colors disabled:opacity-50"
          >
            <AlertTriangle className="w-5 h-5" />
            Parada de Emergencia
          </button>
        </div>
      </div>
    </div>
  );
}

