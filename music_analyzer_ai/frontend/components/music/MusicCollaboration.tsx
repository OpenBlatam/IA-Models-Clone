'use client';

import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { musicApiService } from '@/lib/api/music-api';
import { Network, Users, Music } from 'lucide-react';
import { type Track } from '@/lib/api/music-api';

interface MusicCollaborationProps {
  artistName?: string;
  onTrackSelect?: (track: Track) => void;
}

export function MusicCollaboration({ artistName, onTrackSelect }: MusicCollaborationProps) {
  const [selectedArtist, setSelectedArtist] = useState(artistName || '');

  const { data: network } = useQuery({
    queryKey: ['collaboration-network', selectedArtist],
    queryFn: () => musicApiService.getCollaborationNetwork?.(selectedArtist) || Promise.resolve({ network: [] }),
    enabled: !!selectedArtist,
  });

  const { data: collaborations } = useQuery({
    queryKey: ['collaborations', selectedArtist],
    queryFn: () => musicApiService.analyzeCollaborations?.(selectedArtist) || Promise.resolve({ collaborations: [] }),
    enabled: !!selectedArtist,
  });

  const networkData = network?.network || [];
  const collaborationData = collaborations?.collaborations || [];

  return (
    <div className="space-y-6">
      <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
        <div className="flex items-center gap-2 mb-4">
          <Network className="w-5 h-5 text-purple-300" />
          <h3 className="text-lg font-semibold text-white">Red de Colaboraciones</h3>
        </div>

        <div className="mb-4">
          <input
            type="text"
            value={selectedArtist}
            onChange={(e) => setSelectedArtist(e.target.value)}
            placeholder="Buscar artista..."
            className="w-full px-4 py-2 bg-white/20 border border-white/30 rounded-lg text-white placeholder-gray-300 focus:outline-none focus:ring-2 focus:ring-purple-400"
          />
        </div>

        {networkData.length > 0 ? (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {networkData.slice(0, 8).map((node: any, idx: number) => (
              <div
                key={idx}
                className="p-4 bg-white/5 rounded-lg border border-white/10 text-center"
              >
                <div className="w-12 h-12 rounded-full bg-purple-500 flex items-center justify-center mx-auto mb-2">
                  <Users className="w-6 h-6 text-white" />
                </div>
                <p className="text-white text-sm font-medium truncate">{node.name || 'Artista'}</p>
                <p className="text-xs text-gray-400 mt-1">{node.collaborations || 0} colaboraciones</p>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center py-8">
            <Network className="w-12 h-12 text-gray-500 mx-auto mb-2" />
            <p className="text-gray-400">Ingresa un artista para ver su red de colaboraciones</p>
          </div>
        )}
      </div>

      {collaborationData.length > 0 && (
        <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
          <div className="flex items-center gap-2 mb-4">
            <Music className="w-5 h-5 text-purple-300" />
            <h3 className="text-lg font-semibold text-white">Colaboraciones Recientes</h3>
          </div>

          <div className="space-y-2">
            {collaborationData.slice(0, 10).map((collab: any, idx: number) => (
              <div
                key={idx}
                className="flex items-center gap-3 p-3 bg-white/5 hover:bg-white/10 rounded-lg transition-colors"
              >
                <div className="w-10 h-10 rounded bg-purple-500 flex items-center justify-center flex-shrink-0">
                  <Music className="w-5 h-5 text-white" />
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-white font-medium truncate">
                    {collab.track_name || 'Canción desconocida'}
                  </p>
                  <p className="text-sm text-gray-300 truncate">
                    {collab.artists?.join(', ') || 'Artistas'}
                  </p>
                </div>
                <span className="text-xs text-gray-400">
                  {collab.year || 'Año desconocido'}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}


