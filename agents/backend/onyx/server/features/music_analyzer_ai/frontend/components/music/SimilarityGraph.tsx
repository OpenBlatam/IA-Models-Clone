'use client';

import { type Track } from '@/lib/api/music-api';
import { Network } from 'lucide-react';

interface SimilarityGraphProps {
  tracks: Track[];
  similarities?: Array<{
    track1: string;
    track2: string;
    score: number;
  }>;
}

export function SimilarityGraph({ tracks, similarities }: SimilarityGraphProps) {
  if (!similarities || similarities.length === 0) {
    return (
      <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20 text-center">
        <Network className="w-16 h-16 text-gray-500 mx-auto mb-4" />
        <p className="text-gray-400">No hay datos de similitud disponibles</p>
      </div>
    );
  }

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
      <div className="flex items-center gap-2 mb-4">
        <Network className="w-5 h-5 text-purple-300" />
        <h3 className="text-lg font-semibold text-white">Gráfico de Similitud</h3>
      </div>

      <div className="space-y-3">
        {similarities.slice(0, 10).map((sim, idx) => {
          const track1 = tracks.find((t) => t.id === sim.track1);
          const track2 = tracks.find((t) => t.id === sim.track2);

          return (
            <div key={idx} className="flex items-center gap-3 p-3 bg-white/5 rounded-lg">
              <div className="flex-1">
                <p className="text-sm text-white font-medium truncate">
                  {track1?.name || sim.track1}
                </p>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-24 bg-gray-700 rounded-full h-2">
                  <div
                    className="bg-purple-400 h-2 rounded-full"
                    style={{ width: `${sim.score * 100}%` }}
                  />
                </div>
                <span className="text-sm text-white font-medium w-12 text-right">
                  {Math.round(sim.score * 100)}%
                </span>
              </div>
              <div className="flex-1 text-right">
                <p className="text-sm text-white font-medium truncate">
                  {track2?.name || sim.track2}
                </p>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}


