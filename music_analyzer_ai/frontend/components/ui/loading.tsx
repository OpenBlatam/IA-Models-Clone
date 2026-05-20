/**
 * Shared loading components.
 * Centralized loading states for consistent UX.
 */

import { Loader2 } from 'lucide-react';
import { cn } from '@/lib/utils';

interface LoadingSpinnerProps {
  size?: 'sm' | 'md' | 'lg';
  className?: string;
  text?: string;
}

/**
 * Loading spinner component.
 */
export function LoadingSpinner({
  size = 'md',
  className,
  text,
}: LoadingSpinnerProps) {
  const sizeClasses = {
    sm: 'w-4 h-4',
    md: 'w-6 h-6',
    lg: 'w-8 h-8',
  };

  return (
    <div className={cn('flex items-center justify-center gap-2', className)}>
      <Loader2
        className={cn(
          'text-purple-300 animate-spin',
          sizeClasses[size]
        )}
        aria-label="Cargando"
      />
      {text && <span className="text-gray-300">{text}</span>}
    </div>
  );
}

interface LoadingStateProps {
  message?: string;
  className?: string;
}

/**
 * Full loading state component.
 */
export function LoadingState({ message, className }: LoadingStateProps) {
  return (
    <div
      className={cn(
        'flex flex-col items-center justify-center py-12',
        className
      )}
    >
      <LoadingSpinner size="lg" />
      {message && (
        <p className="mt-4 text-gray-300 text-center">{message}</p>
      )}
    </div>
  );
}

interface SkeletonProps {
  className?: string;
  width?: string;
  height?: string;
}

// Skeleton moved to separate file for better organization

/**
 * Tab loading state component.
 */
export function TabLoadingState() {
  return <LoadingState message="Cargando contenido..." />;
}

