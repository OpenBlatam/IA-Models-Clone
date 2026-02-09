'use client';

import { useQuery } from '@tanstack/react-query';
import { musicApiService } from '@/lib/api/music-api';
import { BarChart3, TrendingUp, Clock, Music } from 'lucide-react';

export function MusicStats() {
  const { data: analytics } = useQuery({
    queryKey: ['music-analytics'],
    queryFn: () => musicApiService.getAnalytics(),
    refetchInterval: 30000,
  });

  const stats = analytics?.stats || {};

  const totalTracks = stats.total_tracks || 0;
  const totalAnalyses = stats.total_analyses || 0;
  const avgResponseTime = stats.avg_response_time || 0;
  const successRate = stats.success_rate || 0;

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
      <div className="bg-white/10 backdrop-blur-lg rounded-xl p-4 border border-white/20">
        <div className="flex items-center gap-2 mb-2">
          <Music className="w-5 h-5 text-purple-300" />
          <span className="text-sm text-gray-400">Total Tracks</span>
        </div>
        <p className="text-2xl font-bold text-white">{totalTracks}</p>
        <p className="text-xs text-gray-400 mt-1">Canciones analizadas</p>
      </div>

      <div className="bg-white/10 backdrop-blur-lg rounded-xl p-4 border border-white/20">
        <div className="flex items-center gap-2 mb-2">
          <BarChart3 className="w-5 h-5 text-blue-300" />
          <span className="text-sm text-gray-400">Análisis</span>
        </div>
        <p className="text-2xl font-bold text-white">{totalAnalyses}</p>
        <p className="text-xs text-gray-400 mt-1">Total realizados</p>
      </div>

      <div className="bg-white/10 backdrop-blur-lg rounded-xl p-4 border border-white/20">
        <div className="flex items-center gap-2 mb-2">
          <Clock className="w-5 h-5 text-green-300" />
          <span className="text-sm text-gray-400">Tiempo Promedio</span>
        </div>
        <p className="text-2xl font-bold text-white">
          {avgResponseTime ? `${(avgResponseTime * 1000).toFixed(0)}ms` : 'N/A'}
        </p>
        <p className="text-xs text-gray-400 mt-1">Respuesta rápida</p>
      </div>

      <div className="bg-white/10 backdrop-blur-lg rounded-xl p-4 border border-white/20">
        <div className="flex items-center gap-2 mb-2">
          <TrendingUp className="w-5 h-5 text-yellow-300" />
          <span className="text-sm text-gray-400">Tasa Éxito</span>
        </div>
        <p className="text-2xl font-bold text-white">
          {successRate ? `${(successRate * 100).toFixed(1)}%` : 'N/A'}
        </p>
        <p className="text-xs text-gray-400 mt-1">Operaciones exitosas</p>
      </div>
    </div>
  );
}


