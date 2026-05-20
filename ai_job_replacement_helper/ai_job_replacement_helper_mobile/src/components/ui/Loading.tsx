import React, { memo } from 'react';
import { View, ActivityIndicator, Text, StyleSheet } from 'react-native';
import { useTheme } from '@/theme/theme';

export interface LoadingProps {
  message?: string;
  size?: 'small' | 'large';
  fullScreen?: boolean;
}

function LoadingComponent({ message, size = 'large', fullScreen = false }: LoadingProps) {
  const theme = useTheme();

  const containerStyle = fullScreen
    ? [styles.fullScreen, { backgroundColor: theme.colors.background }]
    : styles.container;

  return (
    <View style={containerStyle} accessibilityRole="progressbar" accessibilityLabel="Loading">
      <ActivityIndicator size={size} color={theme.colors.primary} />
      {message && (
        <Text style={[styles.message, { color: theme.colors.textSecondary }]}>{message}</Text>
      )}
    </View>
  );
}

export const Loading = memo(LoadingComponent);

const styles = StyleSheet.create({
  container: {
    padding: 24,
    alignItems: 'center',
    justifyContent: 'center',
  },
  fullScreen: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  message: {
    marginTop: 16,
    fontSize: 14,
    textAlign: 'center',
  },
});


