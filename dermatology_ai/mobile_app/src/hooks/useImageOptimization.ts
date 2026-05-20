import { useState, useCallback, useMemo } from 'react';
import { ImageStyle, StyleSheet } from 'react-native';
import FastImage, { Source } from 'react-native-fast-image';

interface UseImageOptimizationOptions {
  placeholder?: Source;
  fallback?: Source;
  priority?: 'low' | 'normal' | 'high';
  cache?: 'immutable' | 'web' | 'cacheOnly';
}

/**
 * Hook para optimizar carga de imágenes con react-native-fast-image
 */
export const useImageOptimization = (
  source: Source | string,
  options: UseImageOptimizationOptions = {}
) => {
  const { placeholder, fallback, priority = 'normal', cache = 'immutable' } =
    options;

  const [error, setError] = useState(false);
  const [loading, setLoading] = useState(true);

  const imageSource = useMemo(() => {
    if (typeof source === 'string') {
      return { uri: source, priority: FastImage.priority[priority] };
    }
    return { ...source, priority: FastImage.priority[priority] };
  }, [source, priority]);

  const handleLoadStart = useCallback(() => {
    setLoading(true);
    setError(false);
  }, []);

  const handleLoadEnd = useCallback(() => {
    setLoading(false);
  }, []);

  const handleError = useCallback(() => {
    setError(true);
    setLoading(false);
  }, []);

  const finalSource = useMemo(() => {
    if (error && fallback) {
      return fallback;
    }
    return imageSource;
  }, [error, fallback, imageSource]);

  return {
    source: finalSource,
    placeholder,
    loading,
    error,
    onLoadStart: handleLoadStart,
    onLoadEnd: handleLoadEnd,
    onError: handleError,
    resizeMode: FastImage.resizeMode.cover,
    cache: FastImage.cacheControl[cache],
  };
};

/**
 * Hook para precargar imágenes
 */
export const useImagePreloader = () => {
  const preloadImages = useCallback((sources: Source[]) => {
    FastImage.preload(sources);
  }, []);

  return { preloadImages };
};

