'use client';

import { useState } from 'react';
import { Activity, Music, Heart, GitCompare, TrendingUp } from 'lucide-react';

interface ActivityItem {
  id: string;
  type: 'analysis' | 'favorite' | 'compare' | 'discovery';
  user: string;
  action: string;
  trackName?: string;
  timestamp: string;
  icon: any;
}

export function MusicActivity() {
  const [activities] = useState<ActivityItem[]>([
    {
      id: '1',
      type: 'analysis',
      user: 'MusicLover23',
      action: 'analizó',
      trackName: 'Bohemian Rhapsody',
      timestamp: 'Hace 5 minutos',
      icon: Music,
    },
    {
      id: '2',
      type: 'favorite',
      user: 'BeatHunter',
      action: 'agregó a favoritos',
      trackName: 'Stairway to Heaven',
      timestamp: 'Hace 15 minutos',
      icon: Heart,
    },
    {
      id: '3',
      type: 'compare',
      user: 'SoundExplorer',
      action: 'comparó',
      trackName: '3 canciones',
      timestamp: 'Hace 1 hora',
      icon: GitCompare,
    },
    {
      id: '4',
      type: 'discovery',
      user: 'MelodySeeker',
      action: 'descubrió',
      trackName: 'Nueva canción',
      timestamp: 'Hace 2 horas',
      icon: TrendingUp,
    },
  ]);

  const getActivityColor = (type: string) => {
    switch (type) {
      case 'analysis':
        return 'text-blue-400 bg-blue-500/20';
      case 'favorite':
        return 'text-red-400 bg-red-500/20';
      case 'compare':
        return 'text-purple-400 bg-purple-500/20';
      case 'discovery':
        return 'text-green-400 bg-green-500/20';
      default:
        return 'text-gray-400 bg-gray-500/20';
    }
  };

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
      <div className="flex items-center gap-2 mb-4">
        <Activity className="w-5 h-5 text-purple-300" />
        <h3 className="text-lg font-semibold text-white">Actividad Reciente</h3>
      </div>

      <div className="space-y-3">
        {activities.map((activity) => {
          const Icon = activity.icon;
          return (
            <div
              key={activity.id}
              className="flex items-center gap-3 p-3 bg-white/5 rounded-lg border border-white/10 hover:bg-white/10 transition-colors"
            >
              <div className={`p-2 rounded-lg ${getActivityColor(activity.type)}`}>
                <Icon className="w-4 h-4" />
              </div>
              <div className="flex-1">
                <p className="text-sm text-white">
                  <span className="font-semibold">{activity.user}</span>{' '}
                  <span className="text-gray-400">{activity.action}</span>{' '}
                  {activity.trackName && (
                    <span className="text-purple-300">{activity.trackName}</span>
                  )}
                </p>
                <p className="text-xs text-gray-400 mt-1">{activity.timestamp}</p>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}


