import { Image } from 'react-native';
import { cacheManager } from './cache-manager';

interface ImageDimensions {
  width: number;
  height: number;
}

export async function getCachedImageDimensions(
  uri: string
): Promise<ImageDimensions | null> {
  const cacheKey = `image-dims-${uri}`;
  const cached = cacheManager.get<ImageDimensions>(cacheKey);
  
  if (cached) {
    return cached;
  }

  try {
    const dimensions = await new Promise<ImageDimensions>((resolve, reject) => {
      Image.getSize(
        uri,
        (width, height) => resolve({ width, height }),
        reject
      );
    });

    cacheManager.set(cacheKey, dimensions, 24 * 60 * 60 * 1000); // 24 hours
    return dimensions;
  } catch {
    return null;
  }
}

export function preloadImage(uri: string): Promise<void> {
  return new Promise((resolve, reject) => {
    Image.prefetch(uri)
      .then(() => resolve())
      .catch(reject);
  });
}

export function preloadImages(uris: string[]): Promise<void[]> {
  return Promise.all(uris.map((uri) => preloadImage(uri)));
}

