'use client';

import { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { musicApiService } from '@/lib/api/music-api';
import { Music, Loader2, Sparkles } from 'lucide-react';
import toast from 'react-hot-toast';
import { type Track } from '@/lib/api/music-api';

export function PlaylistGenerator() {
  const [criteria, setCriteria] = useState({
    mood: '',
    activity: '',
    genre: '',
    energy: '',
    tempo: '',
    duration: '30',
  });
  const [generatedPlaylist, setGeneratedPlaylist] = useState<Track[]>([]);

  const generateMutation = useMutation({
    mutationFn: async (params: any) => {
      // Simular generación de playlist basada en criterios
      const recommendations = await musicApiService.getRecommendationsByMood?.(params.mood) || 
                              await musicApiService.getRecommendationsByActivity?.(params.activity) ||
                              { recommendations: [] };
      return recommendations.recommendations || [];
    },
    onSuccess: (data) => {
      setGeneratedPlaylist(data);
      toast.success(`Playlist generada con ${data.length} canciones`);
    },
    onError: () => {
      toast.error('Error al generar playlist');
    },
  });

  const handleGenerate = () => {
    if (!criteria.mood && !criteria.activity && !criteria.genre) {
      toast.error('Selecciona al menos un criterio');
      return;
    }
    generateMutation.mutate(criteria);
  };

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
      <div className="flex items-center gap-2 mb-4">
        <Sparkles className="w-6 h-6 text-purple-300" />
        <h2 className="text-2xl font-semibold text-white">Generador de Playlist</h2>
      </div>

      <div className="space-y-4 mb-6">
        <div className="grid md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm text-gray-400 mb-2">Mood</label>
            <select
              value={criteria.mood}
              onChange={(e) => setCriteria({ ...criteria, mood: e.target.value })}
              className="w-full px-4 py-2 bg-white/20 border border-white/30 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-400"
            >
              <option value="">Seleccionar mood</option>
              <option value="happy">Feliz</option>
              <option value="sad">Triste</option>
              <option value="energetic">Energético</option>
              <option value="calm">Calmado</option>
              <option value="romantic">Romántico</option>
            </select>
          </div>

          <div>
            <label className="block text-sm text-gray-400 mb-2">Actividad</label>
            <select
              value={criteria.activity}
              onChange={(e) => setCriteria({ ...criteria, activity: e.target.value })}
              className="w-full px-4 py-2 bg-white/20 border border-white/30 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-400"
            >
              <option value="">Seleccionar actividad</option>
              <option value="workout">Ejercicio</option>
              <option value="study">Estudio</option>
              <option value="party">Fiesta</option>
              <option value="relax">Relajación</option>
              <option value="driving">Conducción</option>
            </select>
          </div>

          <div>
            <label className="block text-sm text-gray-400 mb-2">Género</label>
            <input
              type="text"
              value={criteria.genre}
              onChange={(e) => setCriteria({ ...criteria, genre: e.target.value })}
              placeholder="pop, rock, jazz..."
              className="w-full px-4 py-2 bg-white/20 border border-white/30 rounded-lg text-white placeholder-gray-300 focus:outline-none focus:ring-2 focus:ring-purple-400"
            />
          </div>

          <div>
            <label className="block text-sm text-gray-400 mb-2">Duración (minutos)</label>
            <input
              type="number"
              min="10"
              max="120"
              value={criteria.duration}
              onChange={(e) => setCriteria({ ...criteria, duration: e.target.value })}
              className="w-full px-4 py-2 bg-white/20 border border-white/30 rounded-lg text-white placeholder-gray-300 focus:outline-none focus:ring-2 focus:ring-purple-400"
            />
          </div>
        </div>

        <button
          onClick={handleGenerate}
          disabled={generateMutation.isPending}
          className="w-full px-6 py-3 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors disabled:opacity-50 flex items-center justify-center gap-2"
        >
          {generateMutation.isPending ? (
            <>
              <Loader2 className="w-5 h-5 animate-spin" />
              Generando...
            </>
          ) : (
            <>
              <Sparkles className="w-5 h-5" />
              Generar Playlist
            </>
          )}
        </button>
      </div>

      {generatedPlaylist.length > 0 && (
        <div className="space-y-2 max-h-96 overflow-y-auto">
          <h3 className="text-lg font-semibold text-white mb-3">
            Playlist Generada ({generatedPlaylist.length} canciones)
          </h3>
          {generatedPlaylist.map((track, idx) => (
            <div
              key={track.id || idx}
              className="flex items-center gap-3 p-3 bg-white/5 hover:bg-white/10 rounded-lg transition-colors"
            >
              <div className="w-12 h-12 rounded bg-purple-500 flex items-center justify-center flex-shrink-0">
                <Music className="w-6 h-6 text-white" />
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-white font-medium truncate">{track.name}</p>
                <p className="text-sm text-gray-300 truncate">
                  {Array.isArray(track.artists) ? track.artists.join(', ') : track.artists}
                </p>
              </div>
              <span className="text-sm text-gray-400">#{idx + 1}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}


