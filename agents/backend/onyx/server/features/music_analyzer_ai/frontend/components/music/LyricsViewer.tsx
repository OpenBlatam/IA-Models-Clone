'use client';

import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { musicApiService } from '@/lib/api/music-api';
import { FileText, Loader2 } from 'lucide-react';
import toast from 'react-hot-toast';

interface LyricsViewerProps {
  trackId: string;
  trackName: string;
  artists: string[];
}

export function LyricsViewer({ trackId, trackName, artists }: LyricsViewerProps) {
  const [showLyrics, setShowLyrics] = useState(false);

  const { data: lyrics, isLoading } = useQuery({
    queryKey: ['lyrics', trackId],
    queryFn: () => musicApiService.getLyrics?.(trackId) || Promise.resolve({ lyrics: null }),
    enabled: showLyrics,
  });

  if (!showLyrics) {
    return (
      <button
        onClick={() => setShowLyrics(true)}
        className="flex items-center gap-2 px-4 py-2 bg-white/10 hover:bg-white/20 text-white rounded-lg transition-colors"
      >
        <FileText className="w-5 h-5" />
        <span>Ver Letra</span>
      </button>
    );
  }

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <FileText className="w-5 h-5 text-purple-300" />
          <h3 className="text-lg font-semibold text-white">Letra de la Canción</h3>
        </div>
        <button
          onClick={() => setShowLyrics(false)}
          className="text-gray-400 hover:text-white"
        >
          ×
        </button>
      </div>

      {isLoading ? (
        <div className="flex items-center justify-center py-12">
          <Loader2 className="w-8 h-8 text-purple-300 animate-spin" />
        </div>
      ) : lyrics?.lyrics ? (
        <div className="prose prose-invert max-w-none">
          <pre className="text-white whitespace-pre-wrap font-sans text-sm leading-relaxed">
            {lyrics.lyrics}
          </pre>
        </div>
      ) : (
        <div className="text-center py-12">
          <p className="text-gray-400">Letra no disponible para esta canción</p>
        </div>
      )}
    </div>
  );
}


