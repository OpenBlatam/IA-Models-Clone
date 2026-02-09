'use client';

import { Heart, Download, Share2, BarChart3, GitCompare, Sparkles } from 'lucide-react';
import { type Track } from '@/lib/api/music-api';

interface QuickActionsProps {
  track: Track;
  onFavorite?: () => void;
  onAnalyze?: () => void;
  onCompare?: () => void;
  onRecommendations?: () => void;
  onExport?: () => void;
  onShare?: () => void;
}

export function QuickActions({
  track,
  onFavorite,
  onAnalyze,
  onCompare,
  onRecommendations,
  onExport,
  onShare,
}: QuickActionsProps) {
  const actions = [
    { icon: BarChart3, label: 'Analizar', onClick: onAnalyze, color: 'bg-purple-600' },
    { icon: Heart, label: 'Favorito', onClick: onFavorite, color: 'bg-red-600' },
    { icon: GitCompare, label: 'Comparar', onClick: onCompare, color: 'bg-blue-600' },
    { icon: Sparkles, label: 'Recomendaciones', onClick: onRecommendations, color: 'bg-pink-600' },
    { icon: Download, label: 'Exportar', onClick: onExport, color: 'bg-green-600' },
    { icon: Share2, label: 'Compartir', onClick: onShare, color: 'bg-yellow-600' },
  ].filter((action) => action.onClick);

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-4 border border-white/20">
      <h3 className="text-sm font-semibold text-white mb-3">Acciones Rápidas</h3>
      <div className="grid grid-cols-3 gap-2">
        {actions.map((action, idx) => {
          const Icon = action.icon;
          return (
            <button
              key={idx}
              onClick={action.onClick}
              className={`${action.color} hover:opacity-80 text-white p-3 rounded-lg transition-all transform hover:scale-105 flex flex-col items-center gap-1`}
            >
              <Icon className="w-5 h-5" />
              <span className="text-xs">{action.label}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}

