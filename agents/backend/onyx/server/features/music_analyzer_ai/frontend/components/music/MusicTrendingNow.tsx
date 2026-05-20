'use client';

import { useState } from 'react';
import { TrendingUp, Music, Play } from 'lucide-react';
import { type Track } from '@/lib/api/music-api';

interface MusicTrendingNowProps {
  onTrackSelect?: (track: Track) => void;
}

export function MusicTrendingNow({ onTrackSelect }: MusicTrendingNowProps) {
  const [trending] = useState<Array<Partial<Track> & { trend: number }>>([
    {
      id: '1',
      name: 'Bohemian Rhapsody',
      artists: ['Queen'],
      trend: 12.5,
    },
    {
      id: '2',
      name: 'Stairway to Heaven',
      artists: ['Led Zeppelin'],
      trend: 8.3,
    },
    {
      id: '3',
      name: 'Hotel California',
      artists: ['Eagles'],
      trend: 6.7,
    },
    {
      id: '4',
      name: 'Sweet Child O Mine',
      artists: ["Guns N' Roses"],
      trend: 5.2,
    },
  ]);

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
      <div className="flex items-center gap-2 mb-4">
        <TrendingUp className="w-5 h-5 text-green-400" />
        <h3 className="text-lg font-semibold text-white">Tendencias Ahora</h3>
      </div>

      <div className="space-y-3">
        {trending.map((track, idx) => (
          <div
            key={track.id || idx}
            className="flex items-center gap-3 p-3 bg-white/5 rounded-lg border border-white/10 hover:bg-white/10 transition-colors group"
          >
            <div className="flex items-center justify-center w-10 h-10 bg-purple-500/20 rounded-lg text-purple-300 font-bold">
              {idx + 1}
            </div>
            <div className="flex-1">
              <p className="text-white font-medium text-sm">{track.name}</p>
              <p className="text-gray-400 text-xs">
                {Array.isArray(track.artists) ? track.artists.join(', ') : track.artists}
              </p>
            </div>
            <div className="flex items-center gap-2">
              <div className="text-right">
                <div className="flex items-center gap-1 text-green-400 text-xs">
                  <TrendingUp className="w-3 h-3" />
                  <span>+{track.trend}%</span>
                </div>
              </div>
              <button
                onClick={() => onTrackSelect?.(track as Track)}
                className="p-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors opacity-0 group-hover:opacity-100"
              >
                <Play className="w-4 h-4" />
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}


