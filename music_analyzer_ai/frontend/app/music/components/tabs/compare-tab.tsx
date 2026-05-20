'use client';

/**
 * Compare tab component.
 * Handles track comparison functionality.
 */

import { type Track } from '@/lib/api/types';
import { GitCompare } from 'lucide-react';

interface CompareTabProps {
  selectedTrack: Track | null;
  searchResults: Track[];
  analysisData: unknown;
  onTrackSelect: (track: Track) => void;
  onSearchResults: (results: Track[]) => void;
}

/**
 * Compare tab component.
 */
export default function CompareTab({
  searchResults,
}: CompareTabProps) {
  if (searchResults.length === 0) {
    return (
      <div className="bg-white/10 backdrop-blur-lg rounded-xl p-12 border border-white/20 text-center">
        <GitCompare className="w-16 h-16 text-purple-300 mx-auto mb-4" />
        <p className="text-gray-300 text-lg">
          Busca canciones primero para poder compararlas
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <p className="text-white">Comparison functionality coming soon...</p>
    </div>
  );
}

