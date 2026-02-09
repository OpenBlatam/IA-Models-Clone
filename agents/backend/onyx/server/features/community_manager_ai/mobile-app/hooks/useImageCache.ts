import { useState, useEffect } from 'react';
import { Image } from 'expo-image';

interface ImageCacheOptions {
  uri: string;
  priority?: 'low' | 'normal' | 'high';
}

export function useImageCache({ uri, priority = 'normal' }: ImageCacheOptions) {
  const [isCached, setIsCached] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    if (!uri) return;

    // Prefetch image
    Image.prefetch(uri, { priority })
      .then(() => {
        setIsCached(true);
        setIsLoading(false);
      })
      .catch((err) => {
        setError(err);
        setIsLoading(false);
      });
  }, [uri, priority]);

  return { isCached, isLoading, error };
}


