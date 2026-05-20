import React, { memo } from 'react';
import { View, ActivityIndicator, StyleSheet, ViewStyle, ImageStyle } from 'react-native';
import FastImage, { FastImageProps, Source } from 'react-native-fast-image';
import { useImageOptimization } from '../hooks/useImageOptimization';

interface OptimizedImageProps extends Omit<FastImageProps, 'source'> {
  source: Source | string;
  placeholder?: Source;
  fallback?: Source;
  showLoader?: boolean;
  containerStyle?: ViewStyle;
  imageStyle?: ImageStyle;
  priority?: 'low' | 'normal' | 'high';
}

const OptimizedImage: React.FC<OptimizedImageProps> = ({
  source,
  placeholder,
  fallback,
  showLoader = true,
  containerStyle,
  imageStyle,
  priority = 'normal',
  ...props
}) => {
  const {
    source: optimizedSource,
    loading,
    onLoadStart,
    onLoadEnd,
    onError,
    resizeMode,
    cache,
  } = useImageOptimization(source, {
    placeholder,
    fallback,
    priority,
    cache: 'immutable',
  });

  return (
    <View style={[styles.container, containerStyle]}>
      <FastImage
        source={optimizedSource}
        style={[styles.image, imageStyle]}
        resizeMode={resizeMode}
        onLoadStart={onLoadStart}
        onLoadEnd={onLoadEnd}
        onError={onError}
        {...props}
      />
      {loading && showLoader && (
        <View style={styles.loader}>
          <ActivityIndicator size="small" />
        </View>
      )}
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
  loader: {
    ...StyleSheet.absoluteFillObject,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'rgba(0, 0, 0, 0.1)',
  },
});

export default memo(OptimizedImage);

