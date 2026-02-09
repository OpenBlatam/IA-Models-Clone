'use client';

import { useState, useEffect } from 'react';
import { Zap, TrendingUp, Activity } from 'lucide-react';
import { useQuery } from '@tanstack/react-query';
import { musicApiService } from '@/lib/api/music-api';

export function PerformanceOptimizer() {
  const [performance, setPerformance] = useState<any>(null);

  const { data: analytics } = useQuery({
    queryKey: ['music-analytics'],
    queryFn: () => musicApiService.getAnalytics(),
    refetchInterval: 10000,
  });

  useEffect(() => {
    if (analytics?.stats) {
      setPerformance({
        avgResponseTime: analytics.stats.avg_response_time,
        successRate: analytics.stats.success_rate,
        totalRequests: analytics.stats.total_requests,
        cacheHitRate: analytics.stats.cache_hit_rate || 0,
      });
    }
  }, [analytics]);

  if (!performance) return null;

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-4 border border-white/20">
      <div className="flex items-center gap-2 mb-3">
        <Zap className="w-5 h-5 text-yellow-400" />
        <h3 className="text-sm font-semibold text-white">Rendimiento del Sistema</h3>
      </div>
      <div className="grid grid-cols-2 gap-3 text-xs">
        <div>
          <p className="text-gray-400">Tiempo Respuesta</p>
          <p className="text-white font-medium">
            {(performance.avgResponseTime * 1000).toFixed(0)}ms
          </p>
        </div>
        <div>
          <p className="text-gray-400">Tasa Éxito</p>
          <p className="text-white font-medium">
            {(performance.successRate * 100).toFixed(1)}%
          </p>
        </div>
        <div>
          <p className="text-gray-400">Cache Hit</p>
          <p className="text-white font-medium">
            {(performance.cacheHitRate * 100).toFixed(1)}%
          </p>
        </div>
        <div>
          <p className="text-gray-400">Total Requests</p>
          <p className="text-white font-medium">{performance.totalRequests}</p>
        </div>
      </div>
    </div>
  );
}


