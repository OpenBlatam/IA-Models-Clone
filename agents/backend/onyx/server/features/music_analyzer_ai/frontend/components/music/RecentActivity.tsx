'use client';

import { useQuery } from '@tanstack/react-query';
import { musicApiService } from '@/lib/api/music-api';
import { Clock, Music, Heart, ListMusic, BarChart3 } from 'lucide-react';

export function RecentActivity() {
  const { data: history } = useQuery({
    queryKey: ['history', 'user123'],
    queryFn: () => musicApiService.getHistory('user123', 10),
  });

  const activities = history?.history || [];

  const getActivityIcon = (type: string) => {
    switch (type) {
      case 'analysis':
        return BarChart3;
      case 'favorite':
        return Heart;
      case 'playlist':
        return ListMusic;
      default:
        return Music;
    }
  };

  const getActivityColor = (type: string) => {
    switch (type) {
      case 'analysis':
        return 'text-blue-400';
      case 'favorite':
        return 'text-red-400';
      case 'playlist':
        return 'text-green-400';
      default:
        return 'text-purple-400';
    }
  };

  if (activities.length === 0) {
    return (
      <div className="bg-white/10 backdrop-blur-lg rounded-xl p-4 border border-white/20 text-center">
        <Clock className="w-12 h-12 text-gray-500 mx-auto mb-2" />
        <p className="text-gray-400">No hay actividad reciente</p>
      </div>
    );
  }

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
      <div className="flex items-center gap-2 mb-4">
        <Clock className="w-5 h-5 text-purple-300" />
        <h3 className="text-lg font-semibold text-white">Actividad Reciente</h3>
      </div>

      <div className="space-y-3 max-h-96 overflow-y-auto">
        {activities.map((activity: any, idx: number) => {
          const Icon = getActivityIcon(activity.type || 'track');
          const colorClass = getActivityColor(activity.type || 'track');

          return (
            <div
              key={idx}
              className="flex items-center gap-3 p-3 bg-white/5 hover:bg-white/10 rounded-lg transition-colors"
            >
              <div className={`w-10 h-10 rounded-full bg-white/10 flex items-center justify-center ${colorClass}`}>
                <Icon className="w-5 h-5" />
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-white font-medium truncate">
                  {activity.track_name || 'Canción desconocida'}
                </p>
                <p className="text-sm text-gray-400">
                  {activity.type === 'analysis' && 'Análisis realizado'}
                  {activity.type === 'favorite' && 'Agregado a favoritos'}
                  {activity.type === 'playlist' && 'Agregado a playlist'}
                  {!activity.type && 'Track visto'}
                  {activity.analyzed_at && ` • ${new Date(activity.analyzed_at).toLocaleDateString()}`}
                </p>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}


