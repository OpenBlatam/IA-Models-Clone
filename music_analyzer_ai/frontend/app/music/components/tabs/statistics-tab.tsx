'use client';

/**
 * Statistics tab component.
 * Displays music statistics.
 */

import { type Track } from '@/lib/api/types';

interface StatisticsTabProps {
  selectedTrack: Track | null;
  searchResults: Track[];
  analysisData: unknown;
  onTrackSelect: (track: Track) => void;
  onSearchResults: (results: Track[]) => void;
}

/**
 * Statistics tab component.
 */
export default function StatisticsTab() {
  return (
    <div className="space-y-6">
      <p className="text-white">Statistics functionality coming soon...</p>
    </div>
  );
}

