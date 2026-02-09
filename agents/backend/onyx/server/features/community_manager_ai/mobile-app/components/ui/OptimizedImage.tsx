import { Image, ImageProps, StyleSheet, View, ActivityIndicator } from 'react-native';
import { useState } from 'react';
import { Image as ExpoImage } from 'expo-image';

interface OptimizedImageProps extends Omit<ImageProps, 'source'> {
  uri: string;
  placeholder?: string;
  fallback?: string;
}

export function OptimizedImage({
  uri,
  placeholder,
  fallback,
  style,
  ...props
}: OptimizedImageProps) {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  if (error && fallback) {
    return <Image source={{ uri: fallback }} style={style} {...props} />;
  }

  return (
    <View style={style}>
      <ExpoImage
        source={{ uri }}
        placeholder={placeholder}
        contentFit="cover"
        transition={200}
        onLoadStart={() => setLoading(true)}
        onLoadEnd={() => setLoading(false)}
        onError={() => {
          setError(true);
          setLoading(false);
        }}
        style={StyleSheet.absoluteFill}
        {...props}
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


