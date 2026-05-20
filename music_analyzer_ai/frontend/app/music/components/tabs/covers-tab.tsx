'use client';

/**
 * Covers tab component.
 * Displays covers and remixes.
 */

import { type Track } from '@/lib/api/types';

interface CoversTabProps {
  selectedTrack: Track | null;
  searchResults: Track[];
  analysisData: unknown;
  onTrackSelect: (track: Track) => void;
  onSearchResults: (results: Track[]) => void;
}

/**
 * Covers tab component.
 */
export default function CoversTab({ selectedTrack, onTrackSelect }: CoversTabProps) {
  return (
    <div className="space-y-6">
      <p className="text-white">Covers and remixes functionality coming soon...</p>
    </div>
  );
}

