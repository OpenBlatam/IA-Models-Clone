'use client';

import { useState } from 'react';
import { Play, Pause, Volume2, VolumeX } from 'lucide-react';
import { type Track } from '@/lib/api/music-api';
import { formatDuration } from '@/lib/utils';

interface TrackPreviewProps {
  track: Track;
}

export function TrackPreview({ track }: TrackPreviewProps) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [audio] = useState<HTMLAudioElement | null>(
    track.preview_url ? new Audio(track.preview_url) : null
  );

  const handlePlayPause = () => {
    if (!audio) return;

    if (isPlaying) {
      audio.pause();
    } else {
      audio.play();
    }
    setIsPlaying(!isPlaying);
  };

  const handleMute = () => {
    if (!audio) return;
    audio.muted = !isMuted;
    setIsMuted(!isMuted);
  };

  if (!track.preview_url) {
    return (
      <div className="bg-white/10 backdrop-blur-lg rounded-xl p-4 border border-white/20 text-center">
        <p className="text-gray-400 text-sm">Preview no disponible</p>
      </div>
    );
  }

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
      <div className="flex items-center gap-4 mb-4">
        {track.images && track.images[0] && (
          <img
            src={track.images[0].url}
            alt={track.name}
            className="w-20 h-20 rounded-lg"
          />
        )}
        <div className="flex-1">
          <h3 className="text-lg font-semibold text-white">{track.name}</h3>
          <p className="text-purple-200">{track.artists.join(', ')}</p>
          <p className="text-sm text-gray-400">{formatDuration(track.duration_ms)}</p>
        </div>
      </div>

      <div className="flex items-center gap-4">
        <button
          onClick={handlePlayPause}
          className="w-12 h-12 bg-purple-600 hover:bg-purple-700 text-white rounded-full flex items-center justify-center transition-colors"
        >
          {isPlaying ? <Pause className="w-6 h-6" /> : <Play className="w-6 h-6" />}
        </button>
        <button
          onClick={handleMute}
          className="w-10 h-10 bg-white/10 hover:bg-white/20 text-white rounded-full flex items-center justify-center transition-colors"
        >
          {isMuted ? <VolumeX className="w-5 h-5" /> : <Volume2 className="w-5 h-5" />}
        </button>
        <div className="flex-1 bg-gray-700 rounded-full h-2">
          <div className="bg-purple-400 h-2 rounded-full" style={{ width: '0%' }} />
        </div>
      </div>
    </div>
  );
}

