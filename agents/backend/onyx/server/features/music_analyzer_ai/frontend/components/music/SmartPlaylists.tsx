'use client';

import { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { musicApiService } from '@/lib/api/music-api';
import { Sparkles, Loader2, Music } from 'lucide-react';
import toast from 'react-hot-toast';
import { type Track } from '@/lib/api/music-api';

export function SmartPlaylists() {
  const [playlistType, setPlaylistType] = useState<'mood' | 'activity' | 'genre' | 'energy'>('mood');
  const [criteria, setCriteria] = useState('');
  const [generatedPlaylist, setGeneratedPlaylist] = useState<Track[]>([]);

  const generateMutation = useMutation({
    mutationFn: async (params: { type: string; criteria: string }) => {
      switch (params.type) {
        case 'mood':
          return await musicApiService.getRecommendationsByMood?.(params.criteria) || { recommendations: [] };
        case 'activity':
          return await musicApiService.getRecommendationsByActivity?.(params.criteria) || { recommendations: [] };
        default:
          return { recommendations: [] };
      }
    },
    onSuccess: (data) => {
      setGeneratedPlaylist(data.recommendations || []);
      toast.success(`Playlist generada con ${data.recommendations?.length || 0} canciones`);
    },
    onError: () => {
      toast.error('Error al generar playlist');
    },
  });

  const handleGenerate = () => {
    if (!criteria.trim()) {
      toast.error('Selecciona un criterio');
      return;
    }
    generateMutation.mutate({ type: playlistType, criteria });
  };

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
      <div className="flex items-center gap-2 mb-4">
        <Sparkles className="w-6 h-6 text-purple-300" />
        <h2 className="text-2xl font-semibold text-white">Playlists Inteligentes</h2>
      </div>

      <div className="space-y-4 mb-6">
        <div>
          <label className="block text-sm text-gray-400 mb-2">Tipo de Playlist</label>
          <div className="grid grid-cols-4 gap-2">
            {(['mood', 'activity', 'genre', 'energy'] as const).map((type) => (
              <button
                key={type}
                onClick={() => setPlaylistType(type)}
                className={`px-4 py-2 rounded-lg transition-colors ${
                  playlistType === type
                    ? 'bg-purple-600 text-white'
                    : 'bg-white/10 text-gray-300 hover:bg-white/20'
                }`}
              >
                {type === 'mood' && 'Mood'}
                {type === 'activity' && 'Actividad'}
                {type === 'genre' && 'Género'}
                {type === 'energy' && 'Energía'}
              </button>
            ))}
          </div>
        </div>

        <div>
          <label className="block text-sm text-gray-400 mb-2">
            {playlistType === 'mood' && 'Selecciona un mood'}
            {playlistType === 'activity' && 'Selecciona una actividad'}
            {playlistType === 'genre' && 'Ingresa un género'}
            {playlistType === 'energy' && 'Nivel de energía (0-1)'}
          </label>
          {playlistType === 'mood' ? (
            <select
              value={criteria}
              onChange={(e) => setCriteria(e.target.value)}
              className="w-full px-4 py-2 bg-white/20 border border-white/30 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-400"
            >
              <option value="">Seleccionar mood</option>
              <option value="happy">Feliz</option>
              <option value="sad">Triste</option>
              <option value="energetic">Energético</option>
              <option value="calm">Calmado</option>
              <option value="romantic">Romántico</option>
            </select>
          ) : playlistType === 'activity' ? (
            <select
              value={criteria}
              onChange={(e) => setCriteria(e.target.value)}
              className="w-full px-4 py-2 bg-white/20 border border-white/30 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-400"
            >
              <option value="">Seleccionar actividad</option>
              <option value="workout">Ejercicio</option>
              <option value="study">Estudio</option>
              <option value="party">Fiesta</option>
              <option value="relax">Relajación</option>
              <option value="driving">Conducción</option>
            </select>
          ) : (
            <input
              type={playlistType === 'energy' ? 'number' : 'text'}
              min={playlistType === 'energy' ? '0' : undefined}
              max={playlistType === 'energy' ? '1' : undefined}
              step={playlistType === 'energy' ? '0.1' : undefined}
              value={criteria}
              onChange={(e) => setCriteria(e.target.value)}
              placeholder={playlistType === 'genre' ? 'pop, rock, jazz...' : '0.0 - 1.0'}
              className="w-full px-4 py-2 bg-white/20 border border-white/30 rounded-lg text-white placeholder-gray-300 focus:outline-none focus:ring-2 focus:ring-purple-400"
            />
          )}
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
              Generar Playlist Inteligente
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


