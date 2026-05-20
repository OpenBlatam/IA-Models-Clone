import { useState, useEffect } from 'react';

interface UseImageLoaderOptions {
  src: string | null | undefined;
  onLoad?: () => void;
  onError?: (error: Error) => void;
}

export const useImageLoader = (options: UseImageLoaderOptions) => {
  const { src, onLoad, onError } = options;
  const [isLoading, setIsLoading] = useState(true);
  const [hasError, setHasError] = useState(false);
  const [imageSrc, setImageSrc] = useState<string | null>(null);

  useEffect(() => {
    if (!src) {
      setIsLoading(false);
      setHasError(false);
      setImageSrc(null);
      return;
    }

    setIsLoading(true);
    setHasError(false);

    const img = new Image();
    img.src = src;

    const handleLoad = (): void => {
      setIsLoading(false);
      setHasError(false);
      setImageSrc(src);
      onLoad?.();
    };

    const handleError = (): void => {
      setIsLoading(false);
      setHasError(true);
      setImageSrc(null);
      onError?.(new Error('Failed to load image'));
    };

    img.addEventListener('load', handleLoad);
    img.addEventListener('error', handleError);

    return () => {
      img.removeEventListener('load', handleLoad);
      img.removeEventListener('error', handleError);
    };
  }, [src, onLoad, onError]);

  return { isLoading, hasError, imageSrc };
};

