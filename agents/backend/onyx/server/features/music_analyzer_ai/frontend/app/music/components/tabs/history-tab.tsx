'use client';

/**
 * History tab component.
 * Displays analysis history.
 */

import { type Track } from '@/lib/api/types';

interface HistoryTabProps {
  selectedTrack: Track | null;
  searchResults: Track[];
  analysisData: unknown;
  onTrackSelect: (track: Track) => void;
  onSearchResults: (results: Track[]) => void;
}

/**
 * History tab component.
 */
export default function HistoryTab() {
  return (
    <div className="space-y-6">
      <p className="text-white">History functionality coming soon...</p>
    </div>
  );
}

