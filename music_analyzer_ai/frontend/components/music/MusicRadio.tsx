'use client';

import { useState } from 'react';
import { Radio, Play, Pause, SkipForward } from 'lucide-react';
import toast from 'react-hot-toast';

interface RadioStation {
  id: string;
  name: string;
  genre: string;
  listeners: number;
  isPlaying: boolean;
}

export function MusicRadio() {
  const [stations] = useState<RadioStation[]>([
    {
      id: '1',
      name: 'Pop Hits Radio',
      genre: 'Pop',
      listeners: 1250,
      isPlaying: false,
    },
    {
      id: '2',
      name: 'Rock Classics',
      genre: 'Rock',
      listeners: 890,
      isPlaying: false,
    },
    {
      id: '3',
      name: 'Jazz Lounge',
      genre: 'Jazz',
      listeners: 450,
      isPlaying: false,
    },
    {
      id: '4',
      name: 'Electronic Vibes',
      genre: 'Electronic',
      listeners: 1200,
      isPlaying: false,
    },
  ]);

  const [currentStation, setCurrentStation] = useState<string | null>(null);

  const handlePlay = (stationId: string) => {
    if (currentStation === stationId) {
      setCurrentStation(null);
      toast.info('Radio pausada');
    } else {
      setCurrentStation(stationId);
      toast.success('Reproduciendo radio');
    }
  };

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
      <div className="flex items-center gap-2 mb-4">
        <Radio className="w-5 h-5 text-purple-300" />
        <h3 className="text-lg font-semibold text-white">Radio en Vivo</h3>
      </div>

      <div className="space-y-3">
        {stations.map((station) => (
          <div
            key={station.id}
            className={`p-4 rounded-lg border transition-all ${
              currentStation === station.id
                ? 'bg-purple-500/20 border-purple-500/50'
                : 'bg-white/5 border-white/10 hover:bg-white/10'
            }`}
          >
            <div className="flex items-center justify-between mb-2">
              <div className="flex-1">
                <h4 className="text-white font-semibold">{station.name}</h4>
                <div className="flex items-center gap-3 mt-1">
                  <span className="text-xs text-gray-400">{station.genre}</span>
                  <span className="text-xs text-gray-500">•</span>
                  <span className="text-xs text-gray-400">{station.listeners} oyentes</span>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => handlePlay(station.id)}
                  className={`p-2 rounded-lg transition-colors ${
                    currentStation === station.id
                      ? 'bg-purple-600 hover:bg-purple-700'
                      : 'bg-white/10 hover:bg-white/20'
                  } text-white`}
                >
                  {currentStation === station.id ? (
                    <Pause className="w-4 h-4" />
                  ) : (
                    <Play className="w-4 h-4" />
                  )}
                </button>
                <button className="p-2 bg-white/10 hover:bg-white/20 text-white rounded-lg transition-colors">
                  <SkipForward className="w-4 h-4" />
                </button>
              </div>
            </div>
            {currentStation === station.id && (
              <div className="mt-2 pt-2 border-t border-white/10">
                <div className="flex items-center gap-2 text-xs text-gray-400">
                  <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
                  <span>En vivo</span>
                </div>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}


