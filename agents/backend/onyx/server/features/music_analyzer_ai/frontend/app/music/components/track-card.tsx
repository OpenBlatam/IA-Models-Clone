/**
 * Track card component.
 * Reusable card component for displaying track information.
 * Optimized with memoization and proper accessibility.
 */

import { memo } from 'react';
import { type Track } from '@/lib/api/types';
import { TrackImage } from '@/components/ui';
import { cn } from '@/lib/utils';

interface TrackCardProps {
  track: Track;
  onClick?: (track: Track) => void;
  className?: string;
  showDetails?: boolean;
  size?: 'sm' | 'md' | 'lg';
}

/**
 * Size configuration for track card.
 */
const SIZE_CONFIG = {
  sm: {
    image: 'w-10 h-10',
    text: 'text-sm',
    padding: 'p-2',
    dimensions: { width: 40, height: 40 },
  },
  md: {
    image: 'w-12 h-12',
    text: 'text-base',
    padding: 'p-3',
    dimensions: { width: 48, height: 48 },
  },
  lg: {
    image: 'w-16 h-16',
    text: 'text-lg',
    padding: 'p-4',
    dimensions: { width: 64, height: 64 },
  },
} as const;

/**
 * Track card component.
 * Displays track information in a card format with optimized image.
 * Memoized to prevent unnecessary re-renders.
 *
 * @param props - Component props
 * @returns Track card component
 */
export const TrackCard = memo(function TrackCard({
  track,
  onClick,
  className,
  showDetails = true,
  size = 'md',
}: TrackCardProps) {
  const sizeConfig = SIZE_CONFIG[size];
  const { width, height } = sizeConfig.dimensions;

  /**
   * Handles card click.
   */
  const handleClick = () => {
    onClick?.(track);
  };

  /**
   * Generates accessible label for the track.
   */
  const ariaLabel = onClick
    ? `Seleccionar ${track.name} de ${track.artists.join(', ')}`
    : undefined;

  const Component = onClick ? 'button' : 'div';

  return (
    <Component
      onClick={handleClick}
      className={cn(
        'w-full flex items-center gap-3',
        sizeConfig.padding,
        'bg-white/5 hover:bg-white/10 rounded-lg transition-colors',
        'focus:outline-none focus:ring-2 focus:ring-purple-400 focus:ring-offset-2 focus:ring-offset-purple-900',
        onClick && 'cursor-pointer text-left',
        className
      )}
      type={onClick ? 'button' : undefined}
      aria-label={ariaLabel}
      aria-describedby={showDetails ? `track-${track.id}-description` : undefined}
    >
      <TrackImage
        src={track.images?.[0]?.url}
        alt={track.name}
        width={width}
        height={height}
        className={sizeConfig.image}
      />
      {showDetails && (
        <div className="flex-1 min-w-0" id={`track-${track.id}-description`}>
          <p
            className={cn(
              'text-white font-medium truncate',
              sizeConfig.text
            )}
          >
            {track.name}
          </p>
          <p className={cn('text-gray-300 truncate', sizeConfig.text)}>
            {track.artists.join(', ')}
          </p>
        </div>
      )}
    </Component>
  );
});

