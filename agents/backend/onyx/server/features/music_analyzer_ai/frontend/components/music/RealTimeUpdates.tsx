'use client';

import { useEffect, useState } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { Radio } from 'lucide-react';

export function RealTimeUpdates() {
  const queryClient = useQueryClient();
  const [isEnabled, setIsEnabled] = useState(true);

  useEffect(() => {
    if (!isEnabled) return;

    const interval = setInterval(() => {
      // Invalidar queries para actualización en tiempo real
      queryClient.invalidateQueries({ queryKey: ['music-analytics'] });
      queryClient.invalidateQueries({ queryKey: ['favorites'] });
      queryClient.invalidateQueries({ queryKey: ['history'] });
    }, 30000); // Cada 30 segundos

    return () => clearInterval(interval);
  }, [isEnabled, queryClient]);

  return (
    <button
      onClick={() => setIsEnabled(!isEnabled)}
      className={`flex items-center gap-2 px-3 py-2 rounded-lg transition-colors ${
        isEnabled
          ? 'bg-green-600/20 text-green-300 border border-green-500/30'
          : 'bg-white/10 text-gray-300 hover:bg-white/20'
      }`}
      title={isEnabled ? 'Actualizaciones en tiempo real activas' : 'Activar actualizaciones en tiempo real'}
    >
      <Radio className={`w-4 h-4 ${isEnabled ? 'animate-pulse' : ''}`} />
      <span className="text-sm">Tiempo Real</span>
    </button>
  );
}


