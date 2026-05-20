import React, { useRef, useState } from 'react';
import { View, StyleSheet, Animated, Text, ActivityIndicator } from 'react-native';
import { useTheme } from '../contexts/ThemeContext';
import { spacing, typography } from '../theme/colors';

interface PullToRefreshProps {
  refreshing: boolean;
  onRefresh: () => void;
  children: React.ReactNode;
  pullDistance?: number;
  refreshText?: string;
}

export const PullToRefresh: React.FC<PullToRefreshProps> = ({
  refreshing,
  onRefresh,
  children,
  pullDistance = 80,
  refreshText = 'Actualizando...',
}) => {
  const { theme } = useTheme();
  const [pullDistanceState, setPullDistanceState] = useState(0);
  const translateY = useRef(new Animated.Value(0)).current;

  const handleScroll = (event: any) => {
    const offsetY = event.nativeEvent.contentOffset.y;
    if (offsetY < 0) {
      const distance = Math.abs(offsetY);
      setPullDistanceState(distance);
      translateY.setValue(Math.min(distance, pullDistance));
    }
  };

  const handleRelease = () => {
    if (pullDistanceState >= pullDistance && !refreshing) {
      onRefresh();
    }
    Animated.spring(translateY, {
      toValue: 0,
      useNativeDriver: true,
    }).start();
    setPullDistanceState(0);
  };

  const opacity = translateY.interpolate({
    inputRange: [0, pullDistance],
    outputRange: [0, 1],
    extrapolate: 'clamp',
  });

  return (
    <View style={styles.container}>
      <Animated.View
        style={[
          styles.refreshIndicator,
          {
            opacity,
            transform: [{ translateY }],
          },
        ]}
      >
        {refreshing ? (
          <>
            <ActivityIndicator size="small" color={theme.primary} />
            <Text style={[styles.refreshText, { color: theme.text }]}>
              {refreshText}
            </Text>
          </>
        ) : (
          <Text style={[styles.pullText, { color: theme.textSecondary }]}>
            {pullDistanceState >= pullDistance ? 'Suelta para actualizar' : 'Tira para actualizar'}
          </Text>
        )}
      </Animated.View>
      {children}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  refreshIndicator: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: spacing.md,
    zIndex: 1000,
  },
  refreshText: {
    ...typography.bodySmall,
    marginTop: spacing.sm,
  },
  pullText: {
    ...typography.bodySmall,
  },
});

