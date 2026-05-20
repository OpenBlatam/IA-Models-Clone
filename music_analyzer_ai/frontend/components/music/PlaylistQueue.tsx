'use client';

import { useState } from 'react';
import { ListMusic, Play, Trash2, ArrowUp, ArrowDown } from 'lucide-react';
import { type Track } from '@/lib/api/music-api';

interface PlaylistQueueProps {
  tracks: Track[];
  currentTrackId?: string;
  onTrackSelect?: (track: Track) => void;
  onRemove?: (trackId: string) => void;
  onReorder?: (fromIndex: number, toIndex: number) => void;
}

export function PlaylistQueue({
  tracks,
  currentTrackId,
  onTrackSelect,
  onRemove,
  onReorder,
}: PlaylistQueueProps) {
  const [draggedIndex, setDraggedIndex] = useState<number | null>(null);

  const handleDragStart = (index: number) => {
    setDraggedIndex(index);
  };

  const handleDragOver = (e: React.DragEvent, index: number) => {
    e.preventDefault();
    if (draggedIndex === null) return;

    if (draggedIndex !== index) {
      onReorder?.(draggedIndex, index);
      setDraggedIndex(index);
    }
  };

  const handleDragEnd = () => {
    setDraggedIndex(null);
  };

  if (tracks.length === 0) {
    return (
      <div className="bg-white/10 backdrop-blur-lg rounded-xl p-4 border border-white/20 text-center">
        <ListMusic className="w-12 h-12 text-gray-500 mx-auto mb-2" />
        <p className="text-gray-400">La cola está vacía</p>
      </div>
    );
  }

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
      <div className="flex items-center gap-2 mb-4">
        <ListMusic className="w-5 h-5 text-purple-300" />
        <h3 className="text-lg font-semibold text-white">Cola de Reproducción</h3>
        <span className="text-sm text-gray-400">({tracks.length})</span>
      </div>

      <div className="space-y-2 max-h-96 overflow-y-auto">
        {tracks.map((track, idx) => {
          const isCurrent = track.id === currentTrackId;

          return (
            <div
              key={track.id || idx}
              draggable
              onDragStart={() => handleDragStart(idx)}
              onDragOver={(e) => handleDragOver(e, idx)}
              onDragEnd={handleDragEnd}
              className={`flex items-center gap-3 p-3 rounded-lg transition-colors ${
                isCurrent
                  ? 'bg-purple-600/30 border border-purple-500'
                  : 'bg-white/5 hover:bg-white/10'
              } cursor-move`}
            >
              <div className="flex items-center gap-2 flex-shrink-0">
                <span className="text-sm text-gray-400 w-6">{idx + 1}</span>
                {isCurrent && <Play className="w-4 h-4 text-purple-300" />}
              </div>

              {track.images && track.images[0] && (
                <img
                  src={track.images[0].url}
                  alt={track.name}
                  className="w-12 h-12 rounded flex-shrink-0"
                />
              )}

              <div
                className="flex-1 min-w-0 cursor-pointer"
                onClick={() => onTrackSelect?.(track)}
              >
                <p className={`font-medium truncate ${isCurrent ? 'text-white' : 'text-gray-200'}`}>
                  {track.name}
                </p>
                <p className="text-sm text-gray-400 truncate">
                  {Array.isArray(track.artists) ? track.artists.join(', ') : track.artists}
                </p>
              </div>

              <div className="flex items-center gap-1">
                {onReorder && (
                  <>
                    <button
                      onClick={() => onReorder(idx, Math.max(0, idx - 1))}
                      disabled={idx === 0}
                      className="p-1 text-gray-400 hover:text-white disabled:opacity-30"
                    >
                      <ArrowUp className="w-4 h-4" />
                    </button>
                    <button
                      onClick={() => onReorder(idx, Math.min(tracks.length - 1, idx + 1))}
                      disabled={idx === tracks.length - 1}
                      className="p-1 text-gray-400 hover:text-white disabled:opacity-30"
                    >
                      <ArrowDown className="w-4 h-4" />
                    </button>
                  </>
                )}
                {onRemove && (
                  <button
                    onClick={() => onRemove(track.id)}
                    className="p-1 text-gray-400 hover:text-red-300"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}


