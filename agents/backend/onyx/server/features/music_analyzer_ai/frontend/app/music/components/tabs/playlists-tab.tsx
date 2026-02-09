'use client';

/**
 * Playlists tab component.
 * Manages user playlists.
 */

import { type Track } from '@/lib/api/types';

interface PlaylistsTabProps {
  selectedTrack: Track | null;
  searchResults: Track[];
  analysisData: unknown;
  onTrackSelect: (track: Track) => void;
  onSearchResults: (results: Track[]) => void;
}

/**
 * Playlists tab component.
 */
export default function PlaylistsTab() {
  return (
    <div className="space-y-6">
      <p className="text-white">Playlists functionality coming soon...</p>
    </div>
  );
}

