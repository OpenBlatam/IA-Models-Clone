'use client';

import { useQuery } from '@tanstack/react-query';
import { musicApiService } from '@/lib/api/music-api';
import { BarChart3, TrendingUp, Activity, Clock } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

export function MusicAnalytics() {
  const { data: analytics } = useQuery({
    queryKey: ['music-analytics'],
    queryFn: () => musicApiService.getAnalytics(),
    refetchInterval: 30000,
  });

  if (!analytics) {
    return (
      <div className="bg-white/10 backdrop-blur-lg rounded-xl p-4 border border-white/20 text-center">
        <p className="text-gray-400">Cargando analytics...</p>
      </div>
    );
  }

  // Simular datos de tendencia temporal
  const timeSeriesData = [
    { time: '00:00', requests: 10 },
    { time: '04:00', requests: 5 },
    { time: '08:00', requests: 25 },
    { time: '12:00', requests: 40 },
    { time: '16:00', requests: 35 },
    { time: '20:00', requests: 30 },
  ];

  return (
    <div className="space-y-6">
      <div className="grid md:grid-cols-4 gap-4">
        <div className="bg-white/10 backdrop-blur-lg rounded-xl p-4 border border-white/20">
          <div className="flex items-center gap-2 mb-2">
            <Activity className="w-5 h-5 text-purple-300" />
            <span className="text-sm text-gray-400">Total Requests</span>
          </div>
          <p className="text-2xl font-bold text-white">
            {analytics.stats?.total_requests || 0}
          </p>
        </div>

        <div className="bg-white/10 backdrop-blur-lg rounded-xl p-4 border border-white/20">
          <div className="flex items-center gap-2 mb-2">
            <TrendingUp className="w-5 h-5 text-green-300" />
            <span className="text-sm text-gray-400">Tasa Éxito</span>
          </div>
          <p className="text-2xl font-bold text-white">
            {analytics.stats?.success_rate
              ? `${(analytics.stats.success_rate * 100).toFixed(1)}%`
              : 'N/A'}
          </p>
        </div>

        <div className="bg-white/10 backdrop-blur-lg rounded-xl p-4 border border-white/20">
          <div className="flex items-center gap-2 mb-2">
            <Clock className="w-5 h-5 text-blue-300" />
            <span className="text-sm text-gray-400">Tiempo Promedio</span>
          </div>
          <p className="text-2xl font-bold text-white">
            {analytics.stats?.avg_response_time
              ? `${(analytics.stats.avg_response_time * 1000).toFixed(0)}ms`
              : 'N/A'}
          </p>
        </div>

        <div className="bg-white/10 backdrop-blur-lg rounded-xl p-4 border border-white/20">
          <div className="flex items-center gap-2 mb-2">
            <BarChart3 className="w-5 h-5 text-yellow-300" />
            <span className="text-sm text-gray-400">Cache Hit</span>
          </div>
          <p className="text-2xl font-bold text-white">
            {analytics.stats?.cache_hit_rate
              ? `${(analytics.stats.cache_hit_rate * 100).toFixed(1)}%`
              : 'N/A'}
          </p>
        </div>
      </div>

      <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
        <h3 className="text-lg font-semibold text-white mb-4">Tendencia de Requests</h3>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={timeSeriesData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#ffffff20" />
            <XAxis dataKey="time" stroke="#ffffff80" />
            <YAxis stroke="#ffffff80" />
            <Tooltip
              contentStyle={{
                backgroundColor: 'rgba(0, 0, 0, 0.8)',
                border: '1px solid rgba(255, 255, 255, 0.2)',
                borderRadius: '8px',
              }}
            />
            <Line
              type="monotone"
              dataKey="requests"
              stroke="#a855f7"
              strokeWidth={2}
              dot={{ fill: '#a855f7' }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}


