/**
 * Skeleton loader component.
 * Provides loading placeholders for better UX.
 */

import { cn } from '@/lib/utils';

interface SkeletonProps {
  className?: string;
  width?: string | number;
  height?: string | number;
  variant?: 'text' | 'circular' | 'rectangular';
  animation?: 'pulse' | 'wave' | 'none';
}

/**
 * Skeleton loader component.
 * Displays a loading placeholder with customizable appearance.
 *
 * @param props - Component props
 * @returns Skeleton component
 */
export function Skeleton({
  className,
  width,
  height,
  variant = 'rectangular',
  animation = 'pulse',
}: SkeletonProps) {
  const variantClasses = {
    text: 'rounded',
    circular: 'rounded-full',
    rectangular: 'rounded-lg',
  };

  const animationClasses = {
    pulse: 'animate-pulse',
    wave: 'animate-shimmer',
    none: '',
  };

  const style: React.CSSProperties = {};
  if (width) {
    style.width = typeof width === 'number' ? `${width}px` : width;
  }
  if (height) {
    style.height = typeof height === 'number' ? `${height}px` : height;
  }

  return (
    <div
      className={cn(
        'bg-white/10',
        variantClasses[variant],
        animationClasses[animation],
        className
      )}
      style={style}
      aria-label="Cargando..."
      aria-busy="true"
      role="status"
    />
  );
}

/**
 * Text skeleton component.
 * Pre-configured for text loading states.
 */
export function SkeletonText({
  lines = 3,
  className,
}: {
  lines?: number;
  className?: string;
}) {
  return (
    <div className={cn('space-y-2', className)}>
      {Array.from({ length: lines }).map((_, i) => (
        <Skeleton
          key={i}
          variant="text"
          height={16}
          width={i === lines - 1 ? '75%' : '100%'}
        />
      ))}
    </div>
  );
}

/**
 * Card skeleton component.
 * Pre-configured for card loading states.
 */
export function SkeletonCard({ className }: { className?: string }) {
  return (
    <div className={cn('p-4 space-y-4', className)}>
      <Skeleton variant="rectangular" height={200} />
      <SkeletonText lines={2} />
    </div>
  );
}

