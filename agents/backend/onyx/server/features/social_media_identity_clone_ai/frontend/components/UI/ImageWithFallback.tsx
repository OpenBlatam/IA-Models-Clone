import { memo, useState, useCallback } from 'react';
import { useImage } from '@/lib/hooks';
import { cn } from '@/lib/utils';
import Skeleton from './Skeleton';

interface ImageWithFallbackProps {
  src: string;
  alt: string;
  fallback?: string;
  className?: string;
  onLoad?: () => void;
  onError?: () => void;
}

const ImageWithFallback = memo(({
  src,
  alt,
  fallback = '/placeholder.png',
  className = '',
  onLoad,
  onError,
}: ImageWithFallbackProps): JSX.Element => {
  const [imgSrc, setImgSrc] = useState(src);
  const { loading, error } = useImage(imgSrc);

  const handleError = useCallback(() => {
    if (imgSrc !== fallback) {
      setImgSrc(fallback);
    }
    if (onError) {
      onError();
    }
  }, [imgSrc, fallback, onError]);

  const handleLoad = useCallback(() => {
    if (onLoad) {
      onLoad();
    }
  }, [onLoad]);

  if (loading) {
    return <Skeleton className={cn('w-full h-full', className)} />;
  }

  if (error && imgSrc === fallback) {
    return (
      <div className={cn('flex items-center justify-center bg-gray-100', className)}>
        <span className="text-gray-400 text-sm">Image not available</span>
      </div>
    );
  }

  return (
    <img
      src={imgSrc}
      alt={alt}
      className={cn('object-cover', className)}
      onError={handleError}
      onLoad={handleLoad}
      loading="lazy"
      decoding="async"
    />
  );
});

ImageWithFallback.displayName = 'ImageWithFallback';

export default ImageWithFallback;

