'use client';

import { memo, useState, useCallback } from 'react';
import Image from 'next/image';
import { Camera as CameraIcon, AlertCircle } from 'lucide-react';
import { cn } from '@/lib/utils';

interface CameraFrameProps {
  frame: string | null;
  isStreaming: boolean;
  className?: string;
}

const CameraFrame = memo(({ frame, isStreaming, className }: CameraFrameProps): JSX.Element => {
  const [hasError, setHasError] = useState(false);
  const [isLoading, setIsLoading] = useState(true);

  const handleError = useCallback((): void => {
    setHasError(true);
    setIsLoading(false);
  }, []);

  const handleLoad = useCallback((): void => {
    setIsLoading(false);
    setHasError(false);
  }, []);

  if (!frame || hasError) {
    return (
      <div
        className={cn(
          'w-full h-full flex items-center justify-center bg-gray-900',
          className
        )}
        role="img"
        aria-label="Camera feed unavailable"
      >
        <div className="text-center">
          {hasError ? (
            <AlertCircle
              className="w-16 h-16 text-red-500 mx-auto mb-2"
              aria-hidden="true"
            />
          ) : (
            <CameraIcon
              className="w-16 h-16 text-gray-600 mx-auto mb-2"
              aria-hidden="true"
            />
          )}
          <p className="text-gray-400">
            {hasError ? 'Failed to load camera feed' : 'No camera feed'}
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className={cn('relative w-full h-full', className)}>
      {isLoading && (
        <div
          className="absolute inset-0 flex items-center justify-center bg-gray-900 z-10"
          role="status"
          aria-label="Loading camera feed"
        >
          <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-white"></div>
        </div>
      )}
      <Image
        src={frame}
        alt="Camera feed"
        fill
        className={cn('object-contain', isLoading && 'opacity-0')}
        unoptimized
        priority={isStreaming}
        onError={handleError}
        onLoad={handleLoad}
        aria-live="polite"
      />
    </div>
  );
});

CameraFrame.displayName = 'CameraFrame';

export default CameraFrame;

