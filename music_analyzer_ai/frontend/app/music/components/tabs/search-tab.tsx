/**
 * Search tab component.
 * Handles track search functionality with validation and error handling.
 */

import { type Track } from '@/lib/api/types';
import { TrackSearch } from '@/components/music/TrackSearch';
import dynamic from 'next/dynamic';
import { Suspense } from 'react';
import { LoadingState } from '@/components/ui';

// Lazy load advanced search component
const MusicSearchAdvanced = dynamic(
  () =>
    import('@/components/music/MusicSearchAdvanced').then((mod) => ({
      default: mod.MusicSearchAdvanced,
    })),
  {
    ssr: false,
    loading: () => <LoadingState message="Cargando búsqueda avanzada..." />,
  }
);

interface SearchTabProps {
  selectedTrack: Track | null;
  searchResults: Track[];
  analysisData: unknown;
  onTrackSelect: (track: Track) => void;
  onSearchResults: (results: Track[]) => void;
}

/**
 * Search tab component.
 * Provides both basic and advanced search functionality.
 *
 * @param props - Component props
 * @returns Search tab component
 */
export default function SearchTab({
  onTrackSelect,
  onSearchResults,
}: SearchTabProps) {
  return (
    <div className="space-y-6">
      <Suspense fallback={<LoadingState message="Cargando..." />}>
        <MusicSearchAdvanced
          onTrackSelect={onTrackSelect}
          onResults={onSearchResults}
        />
      </Suspense>
      <TrackSearch
        onTrackSelect={onTrackSelect}
        onSearchResults={onSearchResults}
      />
    </div>
  );
}
