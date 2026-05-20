'use client';

/**
 * ML (Machine Learning) tab component.
 * Displays ML-based analysis results.
 */

import { type Track } from '@/lib/api/types';
import { Brain } from 'lucide-react';

interface MLTabProps {
  selectedTrack: Track | null;
  searchResults: Track[];
  analysisData: unknown;
  onTrackSelect: (track: Track) => void;
  onSearchResults: (results: Track[]) => void;
}

/**
 * ML tab component.
 */
export default function MLTab({ selectedTrack }: MLTabProps) {
  if (!selectedTrack) {
    return (
      <div className="bg-white/10 backdrop-blur-lg rounded-xl p-12 border border-white/20 text-center">
        <Brain className="w-16 h-16 text-purple-300 mx-auto mb-4" />
        <p className="text-gray-300 text-lg">
          Selecciona una canción para análisis con Machine Learning
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <p className="text-white">ML analysis functionality coming soon...</p>
    </div>
  );
}

