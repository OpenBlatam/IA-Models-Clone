'use client';

import { useQuery } from '@tanstack/react-query';
import { musicApiService } from '@/lib/api/music-api';
import { BarChart3, TrendingUp, Music, Clock } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';

export function StatisticsView() {
  const { data: analytics } = useQuery({
    queryKey: ['music-analytics'],
    queryFn: () => musicApiService.getAnalytics(),
    refetchInterval: 30000,
  });

  if (!analytics) {
    return (
      <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20 text-center">
        <p className="text-gray-400">Cargando estadísticas...</p>
      </div>
    );
  }

  const genreData = analytics.stats?.genre_distribution || [];
  const timeData = analytics.stats?.time_distribution || [];

  const COLORS = ['#a855f7', '#ec4899', '#8b5cf6', '#f472b6', '#c084fc'];

  return (
    <div className="space-y-6">
      <div className="grid md:grid-cols-4 gap-4">
        <div className="bg-white/10 backdrop-blur-lg rounded-xl p-4 border border-white/20">
          <div className="flex items-center gap-2 mb-2">
            <Music className="w-5 h-5 text-purple-300" />
            <span className="text-sm text-gray-400">Total Tracks</span>
          </div>
          <p className="text-2xl font-bold text-white">
            {analytics.stats?.total_tracks || 0}
          </p>
        </div>

        <div className="bg-white/10 backdrop-blur-lg rounded-xl p-4 border border-white/20">
          <div className="flex items-center gap-2 mb-2">
            <TrendingUp className="w-5 h-5 text-green-300" />
            <span className="text-sm text-gray-400">Análisis</span>
          </div>
          <p className="text-2xl font-bold text-white">
            {analytics.stats?.total_analyses || 0}
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
            <span className="text-sm text-gray-400">Tasa Éxito</span>
          </div>
          <p className="text-2xl font-bold text-white">
            {analytics.stats?.success_rate
              ? `${(analytics.stats.success_rate * 100).toFixed(1)}%`
              : 'N/A'}
          </p>
        </div>
      </div>

      {genreData.length > 0 && (
        <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
          <h3 className="text-lg font-semibold text-white mb-4">Distribución por Género</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={genreData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#ffffff20" />
              <XAxis dataKey="genre" stroke="#ffffff80" />
              <YAxis stroke="#ffffff80" />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'rgba(0, 0, 0, 0.8)',
                  border: '1px solid rgba(255, 255, 255, 0.2)',
                  borderRadius: '8px',
                }}
              />
              <Bar dataKey="count" fill="#a855f7" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {timeData.length > 0 && (
        <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
          <h3 className="text-lg font-semibold text-white mb-4">Distribución Temporal</h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={timeData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                outerRadius={80}
                fill="#8884d8"
                dataKey="count"
              >
                {timeData.map((entry: any, index: number) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip
                contentStyle={{
                  backgroundColor: 'rgba(0, 0, 0, 0.8)',
                  border: '1px solid rgba(255, 255, 255, 0.2)',
                  borderRadius: '8px',
                }}
              />
            </PieChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}


