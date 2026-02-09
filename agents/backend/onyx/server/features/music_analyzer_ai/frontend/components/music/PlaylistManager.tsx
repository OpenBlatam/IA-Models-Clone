'use client';

import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { musicApiService, type Track } from '@/lib/api/music-api';
import { Plus, Music, Trash2, ListMusic } from 'lucide-react';
import toast from 'react-hot-toast';

export function PlaylistManager() {
  const [userId] = useState('user123');
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [playlistName, setPlaylistName] = useState('');
  const [isPublic, setIsPublic] = useState(false);
  const queryClient = useQueryClient();

  const { data: playlists, isLoading } = useQuery({
    queryKey: ['playlists', userId],
    queryFn: () => musicApiService.getPlaylists(userId),
  });

  const createPlaylistMutation = useMutation({
    mutationFn: (name: string) => musicApiService.createPlaylist(userId, name, isPublic),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['playlists', userId] });
      toast.success('Playlist creada');
      setShowCreateForm(false);
      setPlaylistName('');
    },
    onError: () => {
      toast.error('Error al crear playlist');
    },
  });

  const handleCreatePlaylist = () => {
    if (!playlistName.trim()) {
      toast.error('Ingresa un nombre para la playlist');
      return;
    }
    createPlaylistMutation.mutate(playlistName);
  };

  const playlistsList = playlists?.playlists || [];

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-2xl font-semibold text-white flex items-center gap-2">
          <ListMusic className="w-6 h-6" />
          Mis Playlists
        </h2>
        <button
          onClick={() => setShowCreateForm(!showCreateForm)}
          className="flex items-center gap-2 px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors"
        >
          <Plus className="w-5 h-5" />
          Nueva Playlist
        </button>
      </div>

      {showCreateForm && (
        <div className="mb-4 p-4 bg-white/5 rounded-lg border border-white/10">
          <input
            type="text"
            value={playlistName}
            onChange={(e) => setPlaylistName(e.target.value)}
            placeholder="Nombre de la playlist"
            className="w-full px-4 py-2 mb-3 bg-white/20 border border-white/30 rounded-lg text-white placeholder-gray-300 focus:outline-none focus:ring-2 focus:ring-purple-400"
          />
          <div className="flex items-center gap-4 mb-3">
            <label className="flex items-center gap-2 text-gray-300">
              <input
                type="checkbox"
                checked={isPublic}
                onChange={(e) => setIsPublic(e.target.checked)}
                className="w-4 h-4"
              />
              <span>Pública</span>
            </label>
          </div>
          <div className="flex gap-2">
            <button
              onClick={handleCreatePlaylist}
              disabled={createPlaylistMutation.isPending}
              className="px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors disabled:opacity-50"
            >
              Crear
            </button>
            <button
              onClick={() => {
                setShowCreateForm(false);
                setPlaylistName('');
              }}
              className="px-4 py-2 bg-white/10 hover:bg-white/20 text-white rounded-lg transition-colors"
            >
              Cancelar
            </button>
          </div>
        </div>
      )}

      {isLoading ? (
        <p className="text-gray-300">Cargando playlists...</p>
      ) : playlistsList.length === 0 ? (
        <div className="text-center py-12">
          <ListMusic className="w-16 h-16 text-gray-500 mx-auto mb-4" />
          <p className="text-gray-400">No tienes playlists aún</p>
        </div>
      ) : (
        <div className="space-y-2 max-h-96 overflow-y-auto">
          {playlistsList.map((playlist: any) => (
            <div
              key={playlist.id}
              className="flex items-center gap-3 p-3 bg-white/5 hover:bg-white/10 rounded-lg transition-colors"
            >
              <div className="w-12 h-12 rounded bg-purple-500 flex items-center justify-center flex-shrink-0">
                <Music className="w-6 h-6 text-white" />
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-white font-medium truncate">{playlist.name}</p>
                <p className="text-sm text-gray-300">
                  {playlist.track_count || 0} canciones
                </p>
                <p className="text-xs text-gray-400">
                  {playlist.is_public ? 'Pública' : 'Privada'}
                </p>
              </div>
              <button
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

