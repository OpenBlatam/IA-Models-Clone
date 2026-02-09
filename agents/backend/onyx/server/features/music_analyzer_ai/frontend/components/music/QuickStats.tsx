'use client';

import { useQuery } from '@tanstack/react-query';
import { musicApiService } from '@/lib/api/music-api';
import { Zap, TrendingUp, Activity, Clock } from 'lucide-react';

export function QuickStats() {
  const { data: analytics } = useQuery({
    queryKey: ['music-analytics'],
    queryFn: () => musicApiService.getAnalytics(),
    refetchInterval: 30000,
  });

  if (!analytics?.stats) return null;

  const stats = analytics.stats;

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
      <div className="bg-white/10 backdrop-blur-lg rounded-lg p-3 border border-white/20">
        <div className="flex items-center gap-2 mb-1">
          <Zap className="w-4 h-4 text-yellow-400" />
          <span className="text-xs text-gray-400">Rendimiento</span>
        </div>
        <p className="text-lg font-bold text-white">
          {stats.cache_hit_rate
            ? `${(stats.cache_hit_rate * 100).toFixed(0)}%`
            : 'N/A'}
        </p>
      </div>

      <div className="bg-white/10 backdrop-blur-lg rounded-lg p-3 border border-white/20">
        <div className="flex items-center gap-2 mb-1">
          <TrendingUp className="w-4 h-4 text-green-400" />
          <span className="text-xs text-gray-400">Éxito</span>
        </div>
        <p className="text-lg font-bold text-white">
          {stats.success_rate
            ? `${(stats.success_rate * 100).toFixed(0)}%`
            : 'N/A'}
        </p>
      </div>

      <div className="bg-white/10 backdrop-blur-lg rounded-lg p-3 border border-white/20">
        <div className="flex items-center gap-2 mb-1">
          <Activity className="w-4 h-4 text-blue-400" />
          <span className="text-xs text-gray-400">Requests</span>
        </div>
        <p className="text-lg font-bold text-white">{stats.total_requests || 0}</p>
      </div>

      <div className="bg-white/10 backdrop-blur-lg rounded-lg p-3 border border-white/20">
        <div className="flex items-center gap-2 mb-1">
          <Clock className="w-4 h-4 text-purple-400" />
          <span className="text-xs text-gray-400">Tiempo</span>
        </div>
        <p className="text-lg font-bold text-white">
          {stats.avg_response_time
            ? `${(stats.avg_response_time * 1000).toFixed(0)}ms`
            : 'N/A'}
        </p>
      </div>
    </div>
  );
}


