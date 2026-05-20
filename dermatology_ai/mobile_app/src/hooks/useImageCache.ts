import { useState, useEffect } from 'react';
import { Cache } from '../utils/cache';
import * as FileSystem from 'expo-file-system';

interface CachedImage {
  uri: string;
  localUri?: string;
  timestamp: number;
}

/**
 * Hook to cache and manage images locally
 */
export const useImageCache = (imageUri: string | null) => {
  const [cachedUri, setCachedUri] = useState<string | null>(null);
  const [isCaching, setIsCaching] = useState(false);

  useEffect(() => {
    if (!imageUri) {
      setCachedUri(null);
      return;
    }

    const cacheImage = async () => {
      try {
        setIsCaching(true);
        
        // Check cache first
        const cacheKey = `image_${imageUri}`;
        const cached = await Cache.get<CachedImage>(cacheKey);
        
        if (cached && cached.localUri) {
          // Check if local file still exists
          const fileInfo = await FileSystem.getInfoAsync(cached.localUri);
          if (fileInfo.exists) {
            setCachedUri(cached.localUri);
            setIsCaching(false);
            return;
          }
        }

        // Download and cache
        const filename = imageUri.split('/').pop() || 'image.jpg';
        const localUri = `${FileSystem.cacheDirectory}${filename}`;
        
        const downloadResult = await FileSystem.downloadAsync(imageUri, localUri);
        
        if (downloadResult.uri) {
          await Cache.set<CachedImage>(cacheKey, {
            uri: imageUri,
            localUri: downloadResult.uri,
            timestamp: Date.now(),
          });
          setCachedUri(downloadResult.uri);
        } else {
          setCachedUri(imageUri);
        }
      } catch (error) {
        console.error('Error caching image:', error);
        setCachedUri(imageUri); // Fallback to original URI
      } finally {
        setIsCaching(false);
      }
    };

    cacheImage();
  }, [imageUri]);

  return { cachedUri, isCaching };
};

