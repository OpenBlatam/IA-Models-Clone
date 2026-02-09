'use client';

/**
 * Collaborations tab component.
 * Displays artist collaborations.
 */

import { type Track } from '@/lib/api/types';

interface CollaborationsTabProps {
  selectedTrack: Track | null;
  searchResults: Track[];
  analysisData: unknown;
  onTrackSelect: (track: Track) => void;
  onSearchResults: (results: Track[]) => void;
}

/**
 * Collaborations tab component.
 */
export default function CollaborationsTab({ onTrackSelect }: CollaborationsTabProps) {
  return (
    <div className="space-y-6">
      <p className="text-white">Collaborations functionality coming soon...</p>
    </div>
  );
}

