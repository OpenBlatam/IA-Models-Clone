import React, { useState } from 'react';
import { View, Image, StyleSheet, ActivityIndicator, ImageStyle, ViewStyle } from 'react-native';
import { useTheme } from '../contexts/ThemeContext';
import { spacing } from '../theme/colors';

interface LazyImageProps {
  uri: string;
  style?: ImageStyle;
  containerStyle?: ViewStyle;
  placeholder?: React.ReactNode;
  errorComponent?: React.ReactNode;
  resizeMode?: 'cover' | 'contain' | 'stretch' | 'center' | 'repeat';
}

export const LazyImage: React.FC<LazyImageProps> = ({
  uri,
  style,
  containerStyle,
  placeholder,
  errorComponent,
  resizeMode = 'cover',
}) => {
  const { theme } = useTheme();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  const defaultPlaceholder = (
    <View style={[styles.placeholder, { backgroundColor: theme.surfaceVariant }]}>
      <ActivityIndicator size="small" color={theme.primary} />
    </View>
  );

  const defaultErrorComponent = (
    <View style={[styles.errorContainer, { backgroundColor: theme.surfaceVariant }]}>
      <View style={[styles.errorIcon, { backgroundColor: theme.error + '20' }]}>
        <Text style={{ color: theme.error, fontSize: 24 }}>📷</Text>
      </View>
    </View>
  );

  return (
    <View style={[styles.container, containerStyle]}>
      {loading && (placeholder || defaultPlaceholder)}
      {error && (errorComponent || defaultErrorComponent)}
      <Image
        source={{ uri }}
        style={[styles.image, style, (loading || error) && styles.hidden]}
        resizeMode={resizeMode}
        onLoadStart={() => {
          setLoading(true);
          setError(false);
        }}
        onLoadEnd={() => setLoading(false)}
        onError={() => {
          setError(true);
          setLoading(false);
        }}
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    position: 'relative',
    overflow: 'hidden',
  },
  image: {
    width: '100%',
    height: '100%',
  },
  hidden: {
    opacity: 0,
    position: 'absolute',
  },
  placeholder: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    justifyContent: 'center',
    alignItems: 'center',
  },
  errorContainer: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    justifyContent: 'center',
    alignItems: 'center',
  },
  errorIcon: {
    width: 60,
    height: 60,
    borderRadius: 30,
    justifyContent: 'center',
    alignItems: 'center',
  },
});

