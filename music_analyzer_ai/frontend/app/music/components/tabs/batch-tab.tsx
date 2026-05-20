'use client';

/**
 * Batch analysis tab component.
 * Handles batch track analysis.
 */

import { type Track } from '@/lib/api/types';

interface BatchTabProps {
  selectedTrack: Track | null;
  searchResults: Track[];
  analysisData: unknown;
  onTrackSelect: (track: Track) => void;
  onSearchResults: (results: Track[]) => void;
}

/**
 * Batch analysis tab component.
 */
export default function BatchTab({ searchResults }: BatchTabProps) {
  return (
    <div className="space-y-6">
      <p className="text-white">Batch analysis functionality coming soon...</p>
    </div>
  );
}

