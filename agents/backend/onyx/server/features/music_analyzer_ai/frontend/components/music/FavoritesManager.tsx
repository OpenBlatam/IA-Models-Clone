'use client';

import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { musicApiService, type Track } from '@/lib/api/music-api';
import { Heart, Trash2, Music } from 'lucide-react';
import toast from 'react-hot-toast';
import { formatDuration } from '@/lib/utils';

export function FavoritesManager() {
  const [userId] = useState('user123'); // En producción, obtener del contexto de auth
  const queryClient = useQueryClient();

  const { data: favorites, isLoading } = useQuery({
    queryKey: ['favorites', userId],
    queryFn: () => musicApiService.getFavorites(userId),
  });

  const removeFavoriteMutation = useMutation({
    mutationFn: (trackId: string) =>
      musicApiService.removeFromFavorites?.(userId, trackId) || Promise.resolve(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['favorites', userId] });
      toast.success('Eliminado de favoritos');
    },
    onError: () => {
      toast.error('Error al eliminar de favoritos');
    },
  });

  const addFavoriteMutation = useMutation({
    mutationFn: ({ trackId, trackName, artists }: { trackId: string; trackName: string; artists: string[] }) =>
      musicApiService.addToFavorites(userId, trackId, trackName, artists),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['favorites', userId] });
      toast.success('Agregado a favoritos');
    },
    onError: () => {
      toast.error('Error al agregar a favoritos');
    },
  });

  if (isLoading) {
    return (
      <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
        <p className="text-gray-300">Cargando favoritos...</p>
      </div>
    );
  }

  const favoritesList = favorites?.favorites || [];

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-2xl font-semibold text-white flex items-center gap-2">
          <Heart className="w-6 h-6 text-red-400" />
          Mis Favoritos
        </h2>
        <span className="text-sm text-gray-400">
          {favoritesList.length} canciones
        </span>
      </div>

      {favoritesList.length === 0 ? (
        <div className="text-center py-12">
          <Heart className="w-16 h-16 text-gray-500 mx-auto mb-4" />
          <p className="text-gray-400">No tienes canciones favoritas aún</p>
        </div>
      ) : (
        <div className="space-y-2 max-h-96 overflow-y-auto">
          {favoritesList.map((fav: any) => (
            <div
              key={fav.track_id}
              className="flex items-center gap-3 p-3 bg-white/5 hover:bg-white/10 rounded-lg transition-colors"
            >
              <div className="w-12 h-12 rounded bg-purple-500 flex items-center justify-center flex-shrink-0">
                <Music className="w-6 h-6 text-white" />
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-white font-medium truncate">{fav.track_name}</p>
                <p className="text-sm text-gray-300 truncate">
                  {Array.isArray(fav.artists) ? fav.artists.join(', ') : fav.artists}
                </p>
                {fav.added_at && (
                  <p className="text-xs text-gray-400">
                    Agregado: {new Date(fav.added_at).toLocaleDateString()}
                  </p>
                )}
              </div>
              <button
                onClick={() => removeFavoriteMutation.mutate(fav.track_id)}
                className="p-2 text-red-400 hover:text-red-300 hover:bg-red-500/20 rounded-lg transition-colors"
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

