'use client';

import { useQuery } from '@tanstack/react-query';
import { musicApiService } from '@/lib/api/music-api';
import { TrendingUp, Music, Heart, Clock } from 'lucide-react';

export function MusicQuickStats() {
  const { data: stats } = useQuery({
    queryKey: ['music-stats', 'user123'],
    queryFn: async () => {
      // Simular estadísticas
      return {
        totalTracks: 150,
        totalFavorites: 45,
        totalPlaylists: 12,
        totalAnalysis: 89,
        avgEnergy: 0.72,
        avgDanceability: 0.68,
        topGenre: 'Pop',
        listeningTime: '24h 30m',
      };
    },
  });

  if (!stats) {
    return null;
  }

  const statCards = [
    {
      label: 'Canciones',
      value: stats.totalTracks,
      icon: Music,
      color: 'text-blue-400',
      bgColor: 'bg-blue-500/20',
    },
    {
      label: 'Favoritos',
      value: stats.totalFavorites,
      icon: Heart,
      color: 'text-red-400',
      bgColor: 'bg-red-500/20',
    },
    {
      label: 'Playlists',
      value: stats.totalPlaylists,
      icon: Music,
      color: 'text-purple-400',
      bgColor: 'bg-purple-500/20',
    },
    {
      label: 'Análisis',
      value: stats.totalAnalysis,
      icon: TrendingUp,
      color: 'text-green-400',
      bgColor: 'bg-green-500/20',
    },
  ];

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
      {statCards.map((stat, idx) => {
        const Icon = stat.icon;
        return (
          <div
            key={idx}
            className={`${stat.bgColor} backdrop-blur-lg rounded-xl p-4 border border-white/20`}
          >
            <div className="flex items-center justify-between mb-2">
              <Icon className={`w-5 h-5 ${stat.color}`} />
            </div>
            <p className="text-2xl font-bold text-white mb-1">{stat.value}</p>
            <p className="text-xs text-gray-400">{stat.label}</p>
          </div>
        );
      })}
    </div>
  );
}


