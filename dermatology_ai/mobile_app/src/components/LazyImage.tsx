import React, { useState } from 'react';
import { Image, ImageProps, StyleSheet, View, ActivityIndicator } from 'react-native';
import { useTheme } from '../context/ThemeContext';

interface LazyImageProps extends Omit<ImageProps, 'source'> {
  source: { uri: string };
  placeholder?: React.ReactNode;
  errorComponent?: React.ReactNode;
  showLoader?: boolean;
}

const LazyImage: React.FC<LazyImageProps> = ({
  source,
  placeholder,
  errorComponent,
  showLoader = true,
  style,
  ...props
}) => {
  const { colors } = useTheme();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  const handleLoadEnd = () => {
    setLoading(false);
  };

  const handleError = () => {
    setLoading(false);
    setError(true);
  };

  if (error && errorComponent) {
    return <>{errorComponent}</>;
  }

  return (
    <View style={[styles.container, style]}>
      {loading && placeholder ? (
        placeholder
      ) : loading && showLoader ? (
        <View style={styles.loader}>
          <ActivityIndicator size="small" color={colors.primary} />
        </View>
      ) : null}
      <Image
        source={source}
        style={[styles.image, style, loading && styles.hidden]}
        onLoadEnd={handleLoadEnd}
        onError={handleError}
        {...props}
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    position: 'relative',
  },
  image: {
    width: '100%',
    height: '100%',
  },
  hidden: {
    opacity: 0,
  },
  loader: {
    ...StyleSheet.absoluteFillObject,
    justifyContent: 'center',
    alignItems: 'center',
  },
});

export default LazyImage;

