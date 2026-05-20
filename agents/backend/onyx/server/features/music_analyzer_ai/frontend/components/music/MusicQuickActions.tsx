'use client';

import { Play, Heart, Share2, Download, MoreVertical } from 'lucide-react';
import { type Track } from '@/lib/api/music-api';

interface MusicQuickActionsProps {
  track: Track;
  onPlay?: () => void;
  onFavorite?: () => void;
  onShare?: () => void;
  onDownload?: () => void;
}

export function MusicQuickActions({
  track,
  onPlay,
  onFavorite,
  onShare,
  onDownload,
}: MusicQuickActionsProps) {
  return (
    <div className="flex items-center gap-2">
      <button
        onClick={onPlay}
        className="p-2 bg-purple-600 hover:bg-purple-700 text-white rounded-full transition-colors"
        title="Reproducir"
      >
        <Play className="w-4 h-4" />
      </button>
      <button
        onClick={onFavorite}
        className="p-2 bg-white/10 hover:bg-white/20 text-white rounded-full transition-colors"
        title="Agregar a favoritos"
      >
        <Heart className="w-4 h-4" />
      </button>
      <button
        onClick={onShare}
        className="p-2 bg-white/10 hover:bg-white/20 text-white rounded-full transition-colors"
        title="Compartir"
      >
        <Share2 className="w-4 h-4" />
      </button>
      <button
        onClick={onDownload}
        className="p-2 bg-white/10 hover:bg-white/20 text-white rounded-full transition-colors"
        title="Descargar"
      >
        <Download className="w-4 h-4" />
      </button>
      <button
        className="p-2 bg-white/10 hover:bg-white/20 text-white rounded-full transition-colors"
        title="Más opciones"
      >
        <MoreVertical className="w-4 h-4" />
      </button>
    </div>
  );
}


