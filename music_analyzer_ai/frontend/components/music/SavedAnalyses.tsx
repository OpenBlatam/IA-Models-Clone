'use client';

import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { musicApiService } from '@/lib/api/music-api';
import { Save, Clock, Music, Trash2 } from 'lucide-react';
import toast from 'react-hot-toast';

export function SavedAnalyses() {
  const [userId] = useState('user123');

  const { data: history } = useQuery({
    queryKey: ['history', userId],
    queryFn: () => musicApiService.getHistory(userId, 20),
  });

  const savedAnalyses = history?.history || [];

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
      <div className="flex items-center gap-2 mb-4">
        <Save className="w-6 h-6 text-purple-300" />
        <h2 className="text-2xl font-semibold text-white">Análisis Guardados</h2>
      </div>

      {savedAnalyses.length === 0 ? (
        <div className="text-center py-12">
          <Save className="w-16 h-16 text-gray-500 mx-auto mb-4" />
          <p className="text-gray-400">No hay análisis guardados</p>
        </div>
      ) : (
        <div className="space-y-2 max-h-96 overflow-y-auto">
          {savedAnalyses.map((analysis: any) => (
            <div
              key={analysis.analysis_id}
              className="flex items-center gap-3 p-3 bg-white/5 hover:bg-white/10 rounded-lg transition-colors"
            >
              <div className="w-12 h-12 rounded bg-purple-500 flex items-center justify-center flex-shrink-0">
                <Music className="w-6 h-6 text-white" />
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-white font-medium truncate">
                  {analysis.track_name || 'Canción desconocida'}
                </p>
                <p className="text-sm text-gray-300 truncate">
                  {Array.isArray(analysis.artists) ? analysis.artists.join(', ') : analysis.artists || 'Artista desconocido'}
                </p>
                <div className="flex items-center gap-3 mt-1">
                  {analysis.analyzed_at && (
                    <div className="flex items-center gap-1 text-xs text-gray-400">
                      <Clock className="w-3 h-3" />
                      {new Date(analysis.analyzed_at).toLocaleDateString()}
                    </div>
                  )}
                  {analysis.key_signature && (
                    <span className="text-xs text-purple-300">
                      {analysis.key_signature}
                    </span>
                  )}
                </div>
              </div>
              <button
                className="p-2 text-red-400 hover:text-red-300 hover:bg-red-500/20 rounded-lg transition-colors"
                onClick={() => toast.info('Funcionalidad de eliminar próximamente')}
              >
                <Trash2 className="w-5 h-5" />
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}


