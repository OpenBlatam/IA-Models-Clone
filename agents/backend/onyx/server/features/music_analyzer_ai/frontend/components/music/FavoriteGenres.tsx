'use client';

import { useQuery } from '@tanstack/react-query';
import { musicApiService } from '@/lib/api/music-api';
import { Music, TrendingUp } from 'lucide-react';

export function FavoriteGenres() {
  const { data: analytics } = useQuery({
    queryKey: ['music-analytics'],
    queryFn: () => musicApiService.getAnalytics(),
    refetchInterval: 30000,
  });

  const genres = analytics?.stats?.genre_distribution || [];

  if (genres.length === 0) {
    return (
      <div className="bg-white/10 backdrop-blur-lg rounded-xl p-4 border border-white/20 text-center">
        <p className="text-gray-400 text-sm">No hay datos de géneros</p>
      </div>
    );
  }

  const topGenres = genres.slice(0, 5);

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
      <div className="flex items-center gap-2 mb-4">
        <TrendingUp className="w-5 h-5 text-purple-300" />
        <h3 className="text-lg font-semibold text-white">Géneros Favoritos</h3>
      </div>

      <div className="space-y-3">
        {topGenres.map((genre: any, idx: number) => {
          const maxCount = topGenres[0]?.count || 1;
          const percentage = (genre.count / maxCount) * 100;

          return (
            <div key={idx} className="space-y-1">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Music className="w-4 h-4 text-purple-300" />
                  <span className="text-white font-medium">{genre.genre || 'Desconocido'}</span>
                </div>
                <span className="text-sm text-gray-400">{genre.count} canciones</span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-2">
                <div
                  className="bg-purple-500 h-2 rounded-full transition-all"
                  style={{ width: `${percentage}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}


