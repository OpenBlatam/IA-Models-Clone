'use client';

import { TrendingUp, Music, Users, Activity } from 'lucide-react';

interface MusicDashboardProps {
  analytics: any;
}

export function MusicDashboard({ analytics }: MusicDashboardProps) {
  const stats = analytics?.stats || {};
  const requests = analytics?.requests || {};

  return (
    <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
      <div className="bg-white/10 backdrop-blur-lg rounded-xl p-4 border border-white/20">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm text-gray-400 mb-1">Total Requests</p>
            <p className="text-2xl font-bold text-white">
              {stats.total_requests || 0}
            </p>
          </div>
          <Activity className="w-8 h-8 text-purple-300" />
        </div>
      </div>

      <div className="bg-white/10 backdrop-blur-lg rounded-xl p-4 border border-white/20">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm text-gray-400 mb-1">Tiempo Promedio</p>
            <p className="text-2xl font-bold text-white">
              {stats.avg_response_time
                ? `${(stats.avg_response_time * 1000).toFixed(0)}ms`
                : '0ms'}
            </p>
          </div>
          <TrendingUp className="w-8 h-8 text-green-300" />
        </div>
      </div>

      <div className="bg-white/10 backdrop-blur-lg rounded-xl p-4 border border-white/20">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm text-gray-400 mb-1">Endpoints Únicos</p>
            <p className="text-2xl font-bold text-white">
              {stats.unique_endpoints || 0}
            </p>
          </div>
          <Music className="w-8 h-8 text-pink-300" />
        </div>
      </div>

      <div className="bg-white/10 backdrop-blur-lg rounded-xl p-4 border border-white/20">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm text-gray-400 mb-1">Tasa de Éxito</p>
            <p className="text-2xl font-bold text-white">
              {stats.success_rate
                ? `${(stats.success_rate * 100).toFixed(1)}%`
                : '100%'}
            </p>
          </div>
          <Users className="w-8 h-8 text-blue-300" />
        </div>
      </div>
    </div>
  );
}

