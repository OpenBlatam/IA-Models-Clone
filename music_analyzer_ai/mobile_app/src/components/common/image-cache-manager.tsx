import React, { memo, useCallback, useState, useEffect } from 'react';
import { Image, ImageProps } from 'expo-image';
import { View, StyleSheet, ActivityIndicator } from 'react-native';
import { COLORS } from '../../constants/config';

interface ImageCacheManagerProps extends Omit<ImageProps, 'source'> {
  uri: string;
  placeholder?: string;
  fallback?: string;
  cachePolicy?: 'memory' | 'disk' | 'none';
  priority?: 'low' | 'normal' | 'high';
  onLoadStart?: () => void;
  onLoadEnd?: () => void;
  onError?: (error: Error) => void;
}

function ImageCacheManagerComponent({
  uri,
  placeholder,
  fallback,
  cachePolicy = 'disk',
  priority = 'normal',
  onLoadStart,
  onLoadEnd,
  onError,
  style,
  ...props
}: ImageCacheManagerProps) {
  const [isLoading, setIsLoading] = useState(true);
  const [hasError, setHasError] = useState(false);
  const [currentUri, setCurrentUri] = useState(uri);

  useEffect(() => {
    setCurrentUri(uri);
    setHasError(false);
    setIsLoading(true);
  }, [uri]);

  const handleLoadStart = useCallback(() => {
    setIsLoading(true);
    setHasError(false);
    onLoadStart?.();
  }, [onLoadStart]);

  const handleLoadEnd = useCallback(() => {
    setIsLoading(false);
    onLoadEnd?.();
  }, [onLoadEnd]);

  const handleError = useCallback((error: Error) => {
    setIsLoading(false);
    setHasError(true);
    if (fallback && currentUri !== fallback) {
      setCurrentUri(fallback);
    } else {
      onError?.(error);
    }
  }, [fallback, currentUri, onError]);

  const imageSource = {
    uri: currentUri,
    cachePolicy,
    priority,
  };

  return (
    <View style={[styles.container, style]}>
      <Image
        source={imageSource}
        onLoadStart={handleLoadStart}
        onLoadEnd={handleLoadEnd}
        onError={handleError}
        placeholder={placeholder ? { blurhash: placeholder } : undefined}
        transition={200}
        contentFit="cover"
        {...props}
      />
      {isLoading && (
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="small" color={COLORS.primary} />
        </View>
      )}
      {hasError && !fallback && (
        <View style={styles.errorContainer}>
          {/* Error placeholder could be added here */}
        </View>
      )}
    </View>
  );
}

export const ImageCacheManager = memo(ImageCacheManagerComponent);

const styles = StyleSheet.create({
  container: {
    position: 'relative',
  },
  loadingContainer: {
    ...StyleSheet.absoluteFillObject,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'rgba(0, 0, 0, 0.1)',
  },
  errorContainer: {
    ...StyleSheet.absoluteFillObject,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'rgba(0, 0, 0, 0.05)',
  },
});

