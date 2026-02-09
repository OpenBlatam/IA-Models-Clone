import { useState } from 'react';
import { View, StyleSheet, ViewStyle, ActivityIndicator } from 'react-native';
import { Image } from 'expo-image';

interface LazyImageProps {
  uri: string;
  style?: ViewStyle;
  width?: number;
  height?: number;
  placeholder?: string;
  fallback?: string;
  resizeMode?: 'cover' | 'contain' | 'stretch' | 'center';
  onLoadStart?: () => void;
  onLoadEnd?: () => void;
  onError?: () => void;
}

export function LazyImage({
  uri,
  style,
  width,
  height,
  placeholder,
  fallback,
  resizeMode = 'cover',
  onLoadStart,
  onLoadEnd,
  onError,
}: LazyImageProps) {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  const handleLoadStart = () => {
    setLoading(true);
    setError(false);
    onLoadStart?.();
  };

  const handleLoadEnd = () => {
    setLoading(false);
    onLoadEnd?.();
  };

  const handleError = () => {
    setError(true);
    setLoading(false);
    onError?.();
  };

  if (error && fallback) {
    return (
      <Image
        source={{ uri: fallback }}
        style={[style, { width, height }]}
        contentFit={resizeMode}
      />
    );
  }

  return (
    <View style={[style, { width, height }]}>
      <Image
        source={{ uri }}
        placeholder={placeholder}
        contentFit={resizeMode}
        transition={200}
        onLoadStart={handleLoadStart}
        onLoadEnd={handleLoadEnd}
        onError={handleError}
        style={StyleSheet.absoluteFill}
        cachePolicy="memory-disk"
      />
      {loading && (
        <View style={[StyleSheet.absoluteFill, styles.loader]}>
          <ActivityIndicator size="small" color="#0ea5e9" />
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  loader: {
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#f3f4f6',
  },
});

