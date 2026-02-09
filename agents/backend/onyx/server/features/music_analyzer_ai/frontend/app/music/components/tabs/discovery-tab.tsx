'use client';

/**
 * Discovery tab component.
 * Displays music discovery features.
 */

import { type Track } from '@/lib/api/types';

interface DiscoveryTabProps {
  selectedTrack: Track | null;
  searchResults: Track[];
  analysisData: unknown;
  onTrackSelect: (track: Track) => void;
  onSearchResults: (results: Track[]) => void;
}

/**
 * Discovery tab component.
 */
export default function DiscoveryTab({ onTrackSelect }: DiscoveryTabProps) {
  return (
    <div className="space-y-6">
      <p className="text-white">Discovery functionality coming soon...</p>
    </div>
  );
}

