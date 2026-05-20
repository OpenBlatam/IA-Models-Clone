'use client';

import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { musicApiService } from '@/lib/api/music-api';
import { History, Clock, Music } from 'lucide-react';
import { formatDuration } from '@/lib/utils';

export function HistoryView() {
  const [userId] = useState('user123');

  const { data: history, isLoading } = useQuery({
    queryKey: ['history', userId],
    queryFn: () => musicApiService.getHistory(userId, 50),
  });

  if (isLoading) {
    return (
      <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
        <p className="text-gray-300">Cargando historial...</p>
      </div>
    );
  }

  const historyList = history?.history || [];

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
      <div className="flex items-center gap-2 mb-4">
        <History className="w-6 h-6 text-purple-300" />
        <h2 className="text-2xl font-semibold text-white">Historial de Análisis</h2>
      </div>

      {historyList.length === 0 ? (
        <div className="text-center py-12">
          <Clock className="w-16 h-16 text-gray-500 mx-auto mb-4" />
          <p className="text-gray-400">No hay análisis en el historial</p>
        </div>
      ) : (
        <div className="space-y-2 max-h-96 overflow-y-auto">
          {historyList.map((item: any) => (
            <div
              key={item.analysis_id}
              className="flex items-center gap-3 p-3 bg-white/5 hover:bg-white/10 rounded-lg transition-colors"
            >
              <div className="w-12 h-12 rounded bg-purple-500 flex items-center justify-center flex-shrink-0">
                <Music className="w-6 h-6 text-white" />
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-white font-medium truncate">
                  {item.track_name || 'Canción desconocida'}
                </p>
                <p className="text-sm text-gray-300 truncate">
                  {item.artists ? (Array.isArray(item.artists) ? item.artists.join(', ') : item.artists) : 'Artista desconocido'}
                </p>
                <div className="flex items-center gap-4 mt-1">
                  {item.analyzed_at && (
                    <div className="flex items-center gap-1 text-xs text-gray-400">
                      <Clock className="w-3 h-3" />
                      {new Date(item.analyzed_at).toLocaleString()}
                    </div>
                  )}
                  {item.key_signature && (
                    <span className="text-xs text-purple-300">
                      {item.key_signature}
                    </span>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

