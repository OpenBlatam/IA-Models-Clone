'use client';

/**
 * Trends tab component.
 * Displays music trends.
 */

import { type Track } from '@/lib/api/types';

interface TrendsTabProps {
  selectedTrack: Track | null;
  searchResults: Track[];
  analysisData: unknown;
  onTrackSelect: (track: Track) => void;
  onSearchResults: (results: Track[]) => void;
}

/**
 * Trends tab component.
 */
export default function TrendsTab({ onTrackSelect }: TrendsTabProps) {
  return (
    <div className="space-y-6">
      <p className="text-white">Trends functionality coming soon...</p>
    </div>
  );
}

