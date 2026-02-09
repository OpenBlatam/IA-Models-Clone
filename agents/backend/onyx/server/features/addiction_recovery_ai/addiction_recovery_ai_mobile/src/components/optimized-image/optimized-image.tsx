import React, { memo } from 'react';
import { ImageStyle, StyleProp } from 'react-native';
import { Image as ExpoImage, ImageProps as ExpoImageProps } from 'expo-image';

interface OptimizedImageProps extends Omit<ExpoImageProps, 'source' | 'style'> {
  source: string | { uri: string } | number;
  width?: number;
  height?: number;
  style?: StyleProp<ImageStyle>;
  placeholder?: string;
  contentFit?: 'cover' | 'contain' | 'fill' | 'none' | 'scaleDown';
  transition?: number;
}

function OptimizedImageComponent({
  source,
  width,
  height,
  style,
  placeholder,
  contentFit = 'cover',
  transition = 300,
  ...props
}: OptimizedImageProps): JSX.Element {
  const imageSource = typeof source === 'string' ? { uri: source } : source;

  return (
    <ExpoImage
      source={imageSource}
      style={[{ width, height }, style]}
      placeholder={placeholder}
      contentFit={contentFit}
      transition={transition}
      cachePolicy="memory-disk"
      {...props}
    />
  );
}

export const OptimizedImage = memo(OptimizedImageComponent);

