import { useEffect } from 'react';
import { preloadImages } from '@/utils/image-cache';

export function useImagePreload(uris: string[]): void {
  useEffect(() => {
    if (uris.length > 0) {
      preloadImages(uris).catch((error) => {
        console.warn('Failed to preload images:', error);
      });
    }
  }, [uris]);
}

