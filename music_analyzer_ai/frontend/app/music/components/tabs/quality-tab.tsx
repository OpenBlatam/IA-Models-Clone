'use client';

/**
 * Quality tab component.
 * Displays quality analysis.
 */

import { type Track } from '@/lib/api/types';
import { Award } from 'lucide-react';

interface QualityTabProps {
  selectedTrack: Track | null;
  searchResults: Track[];
  analysisData: unknown;
  onTrackSelect: (track: Track) => void;
  onSearchResults: (results: Track[]) => void;
}

/**
 * Quality tab component.
 */
export default function QualityTab({ selectedTrack }: QualityTabProps) {
  if (!selectedTrack) {
    return (
      <div className="bg-white/10 backdrop-blur-lg rounded-xl p-12 border border-white/20 text-center">
        <Award className="w-16 h-16 text-purple-300 mx-auto mb-4" />
        <p className="text-gray-300 text-lg">
          Selecciona una canción para analizar su calidad
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <p className="text-white">Quality analysis functionality coming soon...</p>
    </div>
  );
}

