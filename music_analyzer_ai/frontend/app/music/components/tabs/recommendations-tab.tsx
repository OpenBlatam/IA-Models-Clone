'use client';

/**
 * Recommendations tab component.
 * Displays track recommendations.
 */

import { type Track } from '@/lib/api/types';
import { Sparkles } from 'lucide-react';

interface RecommendationsTabProps {
  selectedTrack: Track | null;
  searchResults: Track[];
  analysisData: unknown;
  onTrackSelect: (track: Track) => void;
  onSearchResults: (results: Track[]) => void;
}

/**
 * Recommendations tab component.
 */
export default function RecommendationsTab({
  selectedTrack,
}: RecommendationsTabProps) {
  if (!selectedTrack) {
    return (
      <div className="bg-white/10 backdrop-blur-lg rounded-xl p-12 border border-white/20 text-center">
        <Sparkles className="w-16 h-16 text-purple-300 mx-auto mb-4" />
        <p className="text-gray-300 text-lg">
          Selecciona una canción para obtener recomendaciones
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <p className="text-white">Recommendations functionality coming soon...</p>
    </div>
  );
}

