/**
 * Music Player Bar component.
 * Bottom player bar with playback controls.
 */

'use client';

import {
  Shuffle,
  SkipBack,
  Play,
  SkipForward,
  Repeat,
  Volume2,
} from 'lucide-react';
import { useState } from 'react';

interface MusicPlayerBarProps {
  selectedSong: string | null;
}

/**
 * Music Player Bar component.
 * Provides playback controls at the bottom of the screen.
 *
 * @param props - Component props
 * @returns Music Player Bar component
 */
export function MusicPlayerBar({ selectedSong }: MusicPlayerBarProps) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [volume, setVolume] = useState(50);

  return (
    <div className="h-16 bg-black border-t border-white/10 flex items-center justify-between px-6">
      {/* Left Section - Currently Playing (can be empty for now) */}
      <div className="flex-1 flex items-center gap-4 min-w-0">
        {selectedSong && (
          <>
            <div className="w-10 h-10 rounded bg-gradient-to-br from-orange-500/20 to-purple-500/20 flex-shrink-0" />
            <div className="min-w-0">
              <p className="text-sm font-medium text-white truncate">
                Song Title
              </p>
              <p className="text-xs text-gray-400 truncate">Artist Name</p>
            </div>
          </>
        )}
      </div>

      {/* Center Section - Playback Controls */}
      <div className="flex-1 flex items-center justify-center gap-4">
        <button className="p-2 hover:bg-white/10 rounded transition-colors">
          <Shuffle className="w-5 h-5 text-gray-400" />
        </button>
        <button className="p-2 hover:bg-white/10 rounded transition-colors">
          <SkipBack className="w-5 h-5 text-gray-400" />
        </button>
        <button
          onClick={() => setIsPlaying(!isPlaying)}
          className="w-12 h-12 bg-white/10 hover:bg-white/20 rounded-full flex items-center justify-center transition-colors"
        >
          {isPlaying ? (
            <div className="w-6 h-6 flex items-center justify-center gap-1">
              <div className="w-1 h-4 bg-white rounded animate-pulse" />
              <div className="w-1 h-4 bg-white rounded animate-pulse" style={{ animationDelay: '0.1s' }} />
              <div className="w-1 h-4 bg-white rounded animate-pulse" style={{ animationDelay: '0.2s' }} />
            </div>
          ) : (
            <Play className="w-6 h-6 text-white" />
          )}
        </button>
        <button className="p-2 hover:bg-white/10 rounded transition-colors">
          <SkipForward className="w-5 h-5 text-gray-400" />
        </button>
        <button className="p-2 hover:bg-white/10 rounded transition-colors">
          <Repeat className="w-5 h-5 text-gray-400" />
        </button>
      </div>

      {/* Right Section - Volume Control */}
      <div className="flex-1 flex items-center justify-end gap-3">
        <Volume2 className="w-5 h-5 text-gray-400" />
        <input
          type="range"
          min="0"
          max="100"
          value={volume}
          onChange={(e) => setVolume(Number(e.target.value))}
          className="w-24 h-1 bg-white/20 rounded-lg appearance-none cursor-pointer accent-orange-500"
        />
      </div>
    </div>
  );
}

