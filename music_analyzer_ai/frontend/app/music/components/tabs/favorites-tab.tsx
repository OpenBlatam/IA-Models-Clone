'use client';

/**
 * Favorites tab component.
 * Displays user's favorite tracks.
 */

import { type Track } from '@/lib/api/types';

interface FavoritesTabProps {
  selectedTrack: Track | null;
  searchResults: Track[];
  analysisData: unknown;
  onTrackSelect: (track: Track) => void;
  onSearchResults: (results: Track[]) => void;
}

/**
 * Favorites tab component.
 */
export default function FavoritesTab({ onTrackSelect }: FavoritesTabProps) {
  return (
    <div className="space-y-6">
      <p className="text-white">Favorites functionality coming soon...</p>
    </div>
  );
}

