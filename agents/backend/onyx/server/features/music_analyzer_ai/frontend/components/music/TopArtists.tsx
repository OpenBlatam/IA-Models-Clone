'use client';

import { useQuery } from '@tanstack/react-query';
import { musicApiService } from '@/lib/api/music-api';
import { User, Crown, Music } from 'lucide-react';

export function TopArtists() {
  const { data: analytics } = useQuery({
    queryKey: ['music-analytics'],
    queryFn: () => musicApiService.getAnalytics(),
    refetchInterval: 30000,
  });

  // Simular datos de artistas top (en producción vendría del backend)
  const topArtists = [
    { name: 'Artista 1', tracks: 45, popularity: 95 },
    { name: 'Artista 2', tracks: 38, popularity: 88 },
    { name: 'Artista 3', tracks: 32, popularity: 82 },
    { name: 'Artista 4', tracks: 28, popularity: 75 },
    { name: 'Artista 5', tracks: 25, popularity: 70 },
  ];

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
      <div className="flex items-center gap-2 mb-4">
        <Crown className="w-5 h-5 text-yellow-400" />
        <h3 className="text-lg font-semibold text-white">Top Artistas</h3>
      </div>

      <div className="space-y-3">
        {topArtists.map((artist, idx) => (
          <div
            key={idx}
            className="flex items-center gap-3 p-3 bg-white/5 hover:bg-white/10 rounded-lg transition-colors"
          >
            <div className="relative">
              <div className="w-12 h-12 rounded-full bg-purple-500 flex items-center justify-center flex-shrink-0">
                {idx === 0 ? (
                  <Crown className="w-6 h-6 text-white" />
                ) : (
                  <User className="w-6 h-6 text-white" />
                )}
              </div>
              <div className="absolute -bottom-1 -right-1 w-6 h-6 bg-purple-600 rounded-full flex items-center justify-center border-2 border-gray-900">
                <span className="text-xs font-bold text-white">{idx + 1}</span>
              </div>
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-white font-medium truncate">{artist.name}</p>
              <div className="flex items-center gap-2 mt-1">
                <Music className="w-3 h-3 text-gray-400" />
                <span className="text-xs text-gray-400">{artist.tracks} canciones</span>
              </div>
            </div>
            <div className="text-right">
              <div className="flex items-center gap-1">
                <div className="w-16 bg-gray-700 rounded-full h-2">
                  <div
                    className="bg-purple-500 h-2 rounded-full"
                    style={{ width: `${artist.popularity}%` }}
                  />
                </div>
                <span className="text-xs text-gray-400 w-8">{artist.popularity}%</span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}


