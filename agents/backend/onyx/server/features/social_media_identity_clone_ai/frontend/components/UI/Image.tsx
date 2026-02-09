import { useState, useCallback } from 'react';
import { cn } from '@/lib/utils';
import Skeleton from './Skeleton';

interface ImageProps {
  src: string;
  alt: string;
  className?: string;
  fallback?: string;
  onError?: () => void;
}

const Image = ({ src, alt, className = '', fallback, onError }: ImageProps): JSX.Element => {
  const [isLoading, setIsLoading] = useState(true);
  const [hasError, setHasError] = useState(false);
  const [currentSrc, setCurrentSrc] = useState(src);

  const handleLoad = useCallback((): void => {
    setIsLoading(false);
  }, []);

  const handleError = useCallback((): void => {
    setIsLoading(false);
    setHasError(true);
    
    if (fallback && currentSrc !== fallback) {
      setCurrentSrc(fallback);
      setHasError(false);
      setIsLoading(true);
    } else if (onError) {
      onError();
    }
  }, [fallback, currentSrc, onError]);

  if (hasError && !fallback) {
    return (
      <div
        className={cn('flex items-center justify-center bg-gray-200', className)}
        role="img"
        aria-label={alt}
      >
        <span className="text-gray-400 text-sm">Image not available</span>
      </div>
    );
  }

  return (
    <div className={cn('relative', className)}>
      {isLoading && (
        <div className="absolute inset-0">
          <Skeleton className="w-full h-full" />
        </div>
      )}
      <img
        src={currentSrc}
        alt={alt}
        onLoad={handleLoad}
        onError={handleError}
        className={cn('w-full h-full object-cover', isLoading && 'opacity-0')}
        loading="lazy"
      />
    </div>
  );
};

export default Image;



