import React from 'react';
import { View, StyleSheet, ActivityIndicator } from 'react-native';
import { useTheme } from '@/contexts/theme-context';
import { Loading } from './loading';

interface LoadingOverlayProps {
  visible: boolean;
  message?: string;
  transparent?: boolean;
}

export function LoadingOverlay({
  visible,
  message,
  transparent = false,
}: LoadingOverlayProps) {
  const { colors } = useTheme();

  if (!visible) {
    return null;
  }

  return (
    <View
      style={[
        styles.overlay,
        {
          backgroundColor: transparent
            ? 'rgba(0, 0, 0, 0.5)'
            : colors.background,
        },
      ]}
    >
      <Loading message={message} />
    </View>
  );
}

const styles = StyleSheet.create({
  overlay: {
    ...StyleSheet.absoluteFillObject,
    justifyContent: 'center',
    alignItems: 'center',
    zIndex: 9999,
  },
});


