import { memo } from 'react';
import { cn } from '@/lib/utils';
import LoadingSpinner from './LoadingSpinner';

interface LoadingOverlayProps {
  isLoading: boolean;
  message?: string;
  className?: string;
  fullScreen?: boolean;
}

const LoadingOverlay = memo(({
  isLoading,
  message,
  className = '',
  fullScreen = false,
}: LoadingOverlayProps): JSX.Element => {
  if (!isLoading) {
    return null;
  }

  return (
    <div
      className={cn(
        'absolute inset-0 bg-white bg-opacity-75 backdrop-blur-sm',
        'flex items-center justify-center z-50',
        fullScreen && 'fixed',
        className
      )}
      role="status"
      aria-live="polite"
      aria-label={message || 'Loading'}
    >
      <div className="flex flex-col items-center gap-4">
        <LoadingSpinner />
        {message && <p className="text-gray-600">{message}</p>}
      </div>
    </div>
  );
});

LoadingOverlay.displayName = 'LoadingOverlay';

export default LoadingOverlay;



