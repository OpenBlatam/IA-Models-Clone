/**
 * Song List Panel component.
 * Right panel displaying list of generated songs.
 */

'use client';

import { useState } from 'react';
import { Search, ThumbsUp, ThumbsDown, MoreVertical, Play } from 'lucide-react';
import { Button } from '@/components/ui';

interface Song {
  id: string;
  title: string;
  version: string;
  description: string;
  duration: string;
  thumbnail: string;
  isPreview?: boolean;
}

interface SongListPanelProps {
  selectedSong: string | null;
  onSongSelect: (songId: string) => void;
}

/**
 * Song List Panel component.
 * Displays list of generated songs with controls.
 *
 * @param props - Component props
 * @returns Song List Panel component
 */
export function SongListPanel({
  selectedSong,
  onSongSelect,
}: SongListPanelProps) {
  // Mock data - replace with actual API call
  const [songs] = useState<Song[]>([
    {
      id: '1',
      title: 'Chavo Rucos Punk',
      version: 'v4.5-all',
      description:
        'electrónica industrial, atmósfera caótica con sintetizadores distorsionados y percusión mecánica, influencia rusa, punk, punk agresivo, electronica',
      duration: '3:10',
      thumbnail: '/api/placeholder/100/100',
    },
    {
      id: '2',
      title: 'Chavo Rucos Punk',
      version: 'v4.5-all',
      description:
        'electrónica industrial, atmósfera caótica con sintetizadores distorsionados y percusión mecánica, influencia rusa, punk, punk agresivo, electronica',
      duration: '3:18',
      thumbnail: '/api/placeholder/100/100',
    },
    {
      id: '3',
      title: 'Chavo Rucos Punk',
      version: 'v5 Preview',
      description:
        'electrónica industrial, atmósfera caótica con sintetizadores distorsionados y percusión mecánica, influencia rusa, punk, punk agresivo, electronica',
      duration: '1:00',
      thumbnail: '/api/placeholder/100/100',
      isPreview: true,
    },
    {
      id: '4',
      title: 'Chavo Rucos Punk',
      version: 'v5 Preview',
      description:
        'electrónica industrial, atmósfera caótica con sintetizadores distorsionados y percusión mecánica, influencia rusa, punk, punk agresivo, electronica',
      duration: '1:00',
      thumbnail: '/api/placeholder/100/100',
      isPreview: true,
    },
    {
      id: '5',
      title: 'Chavo Rucos Punk',
      version: 'v4.5-all',
      description:
        'electrónica industrial, atmósfera caótica con sintetizadores distorsionados y percusión mecánica, influencia rusa, punk, punk agresivo, electronica',
      duration: '2:29',
      thumbnail: '/api/placeholder/100/100',
    },
    {
      id: '6',
      title: 'Chavo Rucos Punk',
      version: 'v4.5-all',
      description:
        'electrónica industrial, atmósfera caótica con sintetizadores distorsionados y percusión mecánica, influencia rusa, punk, punk agresivo, electronica',
      duration: '2:40',
      thumbnail: '/api/placeholder/100/100',
    },
    {
      id: '7',
      title: 'Verica y Rafa: Chavo Rucos Punk',
      version: 'v5 Preview',
      description:
        'electrónica industrial, atmósfera caótica con sintetizadores distorsionados y percusión mecánica, influencia rusa, punk, punk agresivo, electronica',
      duration: '1:00',
      thumbnail: '/api/placeholder/100/100',
      isPreview: true,
    },
  ]);

  return (
    <div className="h-full flex flex-col bg-black">
      {/* Search Bar */}
      <div className="p-4 border-b border-white/10">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
          <input
            type="text"
            placeholder="Q Search"
            className="w-full pl-10 pr-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-orange-500"
          />
        </div>
      </div>

      {/* Song List */}
      <div className="flex-1 overflow-y-auto p-4 space-y-1">
        {songs.map((song) => (
          <div
            key={song.id}
            onClick={() => onSongSelect(song.id)}
            className={`
              flex items-center gap-4 p-3 rounded cursor-pointer transition-colors
              ${
                selectedSong === song.id
                  ? 'bg-white/10'
                  : 'bg-transparent hover:bg-white/5'
              }
            `}
          >
            {/* Thumbnail */}
            <div className="w-20 h-20 rounded-lg bg-gradient-to-br from-orange-500/30 via-purple-500/20 to-pink-500/20 flex-shrink-0 flex items-center justify-center overflow-hidden relative group">
              <Play className="w-8 h-8 text-white/40 group-hover:text-white/70 transition-colors" />
            </div>

            {/* Song Info */}
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 mb-1">
                <h3 className="text-sm font-semibold text-white truncate">
                  {song.title}
                </h3>
                <span className="px-2 py-0.5 text-xs bg-white/10 text-gray-300 rounded flex-shrink-0">
                  {song.version}
                </span>
              </div>
              <p className="text-xs text-gray-400 line-clamp-2 mb-1.5 leading-relaxed">
                {song.description}
              </p>
              <div className="flex items-center gap-3">
                <span className="text-xs text-gray-400">{song.duration}</span>
                <div className="flex items-center gap-1">
                  <button 
                    onClick={(e) => {
                      e.stopPropagation();
                      // Handle like
                    }}
                    className="p-1 hover:bg-white/10 rounded transition-colors"
                  >
                    <ThumbsUp className="w-4 h-4 text-gray-400" />
                  </button>
                  <button 
                    onClick={(e) => {
                      e.stopPropagation();
                      // Handle dislike
                    }}
                    className="p-1 hover:bg-white/10 rounded transition-colors"
                  >
                    <ThumbsDown className="w-4 h-4 text-gray-400" />
                  </button>
                </div>
              </div>
            </div>

            {/* Actions */}
            <div className="flex items-center gap-2 flex-shrink-0">
              {song.isPreview && (
                <Button
                  variant="primary"
                  size="sm"
                  onClick={(e) => {
                    e.stopPropagation();
                    // Handle upgrade
                  }}
                  className="bg-gradient-to-r from-orange-500 to-orange-600 hover:from-orange-600 hover:to-orange-700 text-xs whitespace-nowrap"
                >
                  Upgrade for full song
                </Button>
              )}
              <button 
                onClick={(e) => {
                  e.stopPropagation();
                  // Handle menu
                }}
                className="p-2 hover:bg-white/10 rounded transition-colors"
              >
                <MoreVertical className="w-4 h-4 text-gray-400" />
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

