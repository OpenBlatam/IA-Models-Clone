'use client';

import { useState, useEffect } from 'react';
import { cn } from '@/lib/utils';
import { optimizeImageUrl, getImagePlaceholder } from '@/lib/performance/optimize-images';
import { Skeleton } from '@/components/ui/skeleton';

interface OptimizedImageProps {
  src: string;
  alt: string;
  width?: number;
  height?: number;
  quality?: number;
  className?: string;
  objectFit?: 'contain' | 'cover' | 'fill' | 'none' | 'scale-down';
  priority?: boolean;
  onLoad?: () => void;
  onError?: () => void;
}

const OptimizedImage = ({
  src,
  alt,
  width,
  height,
  quality = 80,
  className,
  objectFit = 'cover',
  priority = false,
  onLoad,
  onError,
}: OptimizedImageProps) => {
  const [isLoading, setIsLoading] = useState(true);
  const [hasError, setHasError] = useState(false);
  const [imageSrc, setImageSrc] = useState<string>(getImagePlaceholder(width || 400, height || 300));

  useEffect(() => {
    if (!src) {
      setHasError(true);
      setIsLoading(false);
      return;
    }

    setIsLoading(true);
    setHasError(false);

    const optimizedSrc = optimizeImageUrl(src, width, quality);
    const img = new Image();

    img.onload = () => {
      setImageSrc(optimizedSrc);
      setIsLoading(false);
      if (onLoad) onLoad();
    };

    img.onerror = () => {
      setHasError(true);
      setIsLoading(false);
      if (onError) onError();
    };

    img.src = optimizedSrc;

    return () => {
      img.onload = null;
      img.onerror = null;
    };
  }, [src, width, quality, onLoad, onError]);

  if (hasError) {
    return (
      <div
        className={cn('bg-gray-200 flex items-center justify-center', className)}
        style={{ width, height }}
      >
        <span className="text-gray-400 text-sm">Error al cargar imagen</span>
      </div>
    );
  }

  return (
    <div className={cn('relative overflow-hidden', className)} style={{ width, height }}>
      {isLoading && (
        <Skeleton className="absolute inset-0" />
      )}
      <img
        src={imageSrc}
        alt={alt}
        className={cn(
          'transition-opacity duration-300',
          isLoading ? 'opacity-0' : 'opacity-100',
          `object-${objectFit}`
        )}
        style={{ width: '100%', height: '100%' }}
        loading={priority ? 'eager' : 'lazy'}
        decoding="async"
      />
    </div>
  );
};

export { OptimizedImage };

