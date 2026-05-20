/**
 * Track list component.
 * Optimized list component for displaying multiple tracks.
 */

import { type Track } from '@/lib/api/types';
import { TrackCard } from './track-card';
import { LoadingState, ErrorMessage } from '@/components/ui';
import { cn } from '@/lib/utils';

interface TrackListProps {
  tracks: Track[];
  onTrackSelect?: (track: Track) => void;
  isLoading?: boolean;
  error?: Error | null;
  emptyMessage?: string;
  className?: string;
  size?: 'sm' | 'md' | 'lg';
  maxHeight?: string;
}

/**
 * Track list component.
 * Displays a list of tracks with loading and error states.
 *
 * @param props - Component props
 * @returns Track list component
 */
export function TrackList({
  tracks,
  onTrackSelect,
  isLoading = false,
  error = null,
  emptyMessage = 'No se encontraron canciones',
  className,
  size = 'md',
  maxHeight = 'max-h-96',
}: TrackListProps) {
  // Early return for loading state
  if (isLoading) {
    return (
      <div className={cn('py-8', className)}>
        <LoadingState message="Cargando canciones..." />
      </div>
    );
  }

  // Early return for error state
  if (error) {
    return (
      <div className={cn('py-4', className)}>
        <ErrorMessage
          message={`Error al cargar canciones: ${error.message}`}
          variant="banner"
        />
      </div>
    );
  }

  // Early return for empty state
  if (tracks.length === 0) {
    return (
      <div className={cn('text-center py-8 text-gray-300', className)}>
        {emptyMessage}
      </div>
    );
  }

  return (
    <div
      className={cn(
        'space-y-2 overflow-y-auto',
        maxHeight,
        className
      )}
      role="list"
      aria-label="Lista de canciones"
    >
      {tracks.map((track) => (
        <div key={track.id} role="listitem">
          <TrackCard
            track={track}
            onClick={onTrackSelect}
            size={size}
          />
        </div>
      ))}
    </div>
  );
}

