import { useState, useCallback, useMemo } from 'react';
import type { Track } from '../../types/api';

export interface Playlist {
  id: string;
  name: string;
  tracks: Track[];
  createdAt: Date;
}

export function usePlaylist() {
  const [playlists, setPlaylists] = useState<Playlist[]>([]);
  const [currentPlaylistId, setCurrentPlaylistId] = useState<string | null>(
    null
  );

  const currentPlaylist = useMemo(
    () => playlists.find((p) => p.id === currentPlaylistId) || null,
    [playlists, currentPlaylistId]
  );

  const createPlaylist = useCallback((name: string) => {
    const newPlaylist: Playlist = {
      id: `playlist-${Date.now()}`,
      name,
      tracks: [],
      createdAt: new Date(),
    };
    setPlaylists((prev) => [...prev, newPlaylist]);
    return newPlaylist.id;
  }, []);

  const deletePlaylist = useCallback((playlistId: string) => {
    setPlaylists((prev) => prev.filter((p) => p.id !== playlistId));
    if (currentPlaylistId === playlistId) {
      setCurrentPlaylistId(null);
    }
  }, [currentPlaylistId]);

  const addTrackToPlaylist = useCallback(
    (playlistId: string, track: Track) => {
      setPlaylists((prev) =>
        prev.map((playlist) =>
          playlist.id === playlistId
            ? {
                ...playlist,
                tracks: playlist.tracks.some((t) => t.id === track.id)
                  ? playlist.tracks
                  : [...playlist.tracks, track],
              }
            : playlist
        )
      );
    },
    []
  );

  const removeTrackFromPlaylist = useCallback(
    (playlistId: string, trackId: string) => {
      setPlaylists((prev) =>
        prev.map((playlist) =>
          playlist.id === playlistId
            ? {
                ...playlist,
                tracks: playlist.tracks.filter((t) => t.id !== trackId),
              }
            : playlist
        )
      );
    },
    []
  );

  const reorderPlaylist = useCallback(
    (playlistId: string, fromIndex: number, toIndex: number) => {
      setPlaylists((prev) =>
        prev.map((playlist) => {
          if (playlist.id !== playlistId) return playlist;

          const newTracks = [...playlist.tracks];
          const [removed] = newTracks.splice(fromIndex, 1);
          newTracks.splice(toIndex, 0, removed);

          return { ...playlist, tracks: newTracks };
        })
      );
    },
    []
  );

  return {
    playlists,
    currentPlaylist,
    currentPlaylistId,
    setCurrentPlaylistId,
    createPlaylist,
    deletePlaylist,
    addTrackToPlaylist,
    removeTrackFromPlaylist,
    reorderPlaylist,
  };
}


