import React, { useState } from 'react';
import { View, Image, StyleSheet, ActivityIndicator, ImageStyle } from 'react-native';
import { useTheme } from '../contexts/ThemeContext';

interface ImageLoaderProps {
  uri: string;
  style?: ImageStyle;
  placeholder?: React.ReactNode;
  onLoad?: () => void;
  onError?: () => void;
}

export const ImageLoader: React.FC<ImageLoaderProps> = ({
  uri,
  style,
  placeholder,
  onLoad,
  onError,
}) => {
  const { theme } = useTheme();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  const handleLoad = () => {
    setLoading(false);
    onLoad?.();
  };

  const handleError = () => {
    setLoading(false);
    setError(true);
    onError?.();
  };

  if (error) {
    return (
      <View style={[styles.container, style, { backgroundColor: theme.surfaceVariant }]}>
        {placeholder || (
          <View style={[styles.placeholder, { backgroundColor: theme.surfaceVariant }]}>
            <ActivityIndicator size="small" color={theme.textTertiary} />
          </View>
        )}
      </View>
    );
  }

  return (
    <View style={[styles.container, style]}>
      {loading && (
        <View style={[styles.loader, { backgroundColor: theme.surfaceVariant }]}>
          <ActivityIndicator size="small" color={theme.primary} />
        </View>
      )}
      <Image
        source={{ uri }}
        style={[styles.image, style]}
        onLoad={handleLoad}
        onError={handleError}
        resizeMode="cover"
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    overflow: 'hidden',
  },
  image: {
    width: '100%',
    height: '100%',
  },
  loader: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    justifyContent: 'center',
    alignItems: 'center',
    zIndex: 1,
  },
  placeholder: {
    width: '100%',
    height: '100%',
    justifyContent: 'center',
    alignItems: 'center',
  },
});

