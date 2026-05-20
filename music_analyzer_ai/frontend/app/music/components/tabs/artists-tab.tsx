'use client';

/**
 * Artists tab component.
 * Displays artist analysis.
 */

import { type Track } from '@/lib/api/types';

interface ArtistsTabProps {
  selectedTrack: Track | null;
  searchResults: Track[];
  analysisData: unknown;
  onTrackSelect: (track: Track) => void;
  onSearchResults: (results: Track[]) => void;
}

/**
 * Artists tab component.
 */
export default function ArtistsTab() {
  return (
    <div className="space-y-6">
      <p className="text-white">Artists functionality coming soon...</p>
    </div>
  );
}

