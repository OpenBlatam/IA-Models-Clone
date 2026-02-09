'use client';

import { useRobotStore } from '@/lib/store/robotStore';
import { Zap, Home, Square, RotateCcw, Play, Pause } from 'lucide-react';
import { toast } from '@/lib/utils/toast';

export default function QuickActions() {
  const { moveTo, stop, currentPosition } = useRobotStore();

  const quickActions = [
    {
      name: 'Home',
      icon: <Home className="w-5 h-5" />,
      action: () => {
        moveTo({ x: 0, y: 0, z: 0 });
        toast.info('Moviendo a posición home');
      },
      color: 'bg-blue-600 hover:bg-blue-700',
    },
    {
      name: 'Stop',
      icon: <Square className="w-5 h-5" />,
      action: () => {
        stop();
        toast.warning('Robot detenido');
      },
      color: 'bg-red-600 hover:bg-red-700',
    },
    {
      name: 'Reset',
      icon: <RotateCcw className="w-5 h-5" />,
      action: () => {
        if (currentPosition) {
          moveTo(currentPosition);
          toast.info('Posición reiniciada');
        }
      },
      color: 'bg-yellow-600 hover:bg-yellow-700',
    },
  ];

  return (
    <div className="bg-white rounded-lg p-tesla-lg border border-gray-200 shadow-sm">
      <h3 className="text-sm font-semibold text-tesla-black mb-tesla-md flex items-center gap-tesla-sm uppercase tracking-wide">
        <Zap className="w-4 h-4 text-tesla-blue" />
        Acciones Rápidas
      </h3>
      <div className="grid grid-cols-3 gap-tesla-sm">
        {quickActions.map((action, index) => (
          <button
            key={index}
            onClick={action.action}
            className="bg-white border-2 border-gray-300 hover:border-tesla-blue text-tesla-black p-tesla-md rounded-md transition-all flex flex-col items-center gap-tesla-sm hover:shadow-md"
          >
            <div className="text-tesla-blue">{action.icon}</div>
            <span className="text-xs font-medium">{action.name}</span>
          </button>
        ))}
      </div>
    </div>
  );
}


