import React, { useMemo } from 'react';
import { Image, ImageProps, StyleSheet, View } from 'react-native';
import { Image as ExpoImage } from 'expo-image';
import { COLORS } from '../../constants/config';

interface OptimizedImageProps extends Omit<ImageProps, 'source'> {
  uri: string;
  width?: number;
  height?: number;
  useExpoImage?: boolean;
  placeholder?: string;
  blurhash?: string;
}

/**
 * Optimized image component with WebP support and lazy loading
 * Uses expo-image for better performance when available
 */
export function OptimizedImage({
  uri,
  width,
  height,
  useExpoImage = true,
  placeholder,
  blurhash,
  style,
  ...props
}: OptimizedImageProps) {
  const imageSource = useMemo(() => {
    // Prefer WebP format when available
    const webpUri = uri.replace(/\.(jpg|jpeg|png)$/i, '.webp');
    return { uri: webpUri };
  }, [uri]);

  const imageStyle = useMemo(
    () => [
      styles.image,
      width && { width },
      height && { height },
      style,
    ],
    [width, height, style]
  );

  if (useExpoImage) {
    return (
      <ExpoImage
        source={imageSource}
        style={imageStyle}
        placeholder={blurhash || placeholder}
        contentFit="cover"
        transition={200}
        cachePolicy="memory-disk"
        {...props}
      />
    );
  }

  return (
    <Image
      source={imageSource}
      style={imageStyle}
      resizeMode="cover"
      {...props}
    />
  );
}

const styles = StyleSheet.create({
  image: {
    backgroundColor: COLORS.surfaceLight,
  },
});

