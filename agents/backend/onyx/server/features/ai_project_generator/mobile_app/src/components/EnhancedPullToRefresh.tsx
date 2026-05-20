import React, { useRef, useEffect } from 'react';
import { Animated, View, Text, StyleSheet, ActivityIndicator } from 'react-native';
import { useTheme } from '../contexts/ThemeContext';
import { spacing, typography } from '../theme/colors';

interface EnhancedPullToRefreshProps {
  refreshing: boolean;
  onRefresh: () => void;
  children: React.ReactNode;
  pullDistance?: number;
  refreshThreshold?: number;
}

export const EnhancedPullToRefresh: React.FC<EnhancedPullToRefreshProps> = ({
  refreshing,
  onRefresh,
  children,
  pullDistance = 80,
  refreshThreshold = 60,
}) => {
  const { theme } = useTheme();
  const pullAnim = useRef(new Animated.Value(0)).current;
  const [canRefresh, setCanRefresh] = React.useState(false);

  useEffect(() => {
    if (refreshing) {
      Animated.spring(pullAnim, {
        toValue: pullDistance,
        useNativeDriver: true,
        tension: 50,
        friction: 7,
      }).start();
    } else {
      Animated.spring(pullAnim, {
        toValue: 0,
        useNativeDriver: true,
        tension: 50,
        friction: 7,
      }).start();
    }
  }, [refreshing, pullDistance]);

  const handleScroll = (event: any) => {
    const offsetY = event.nativeEvent.contentOffset.y;
    if (offsetY < -refreshThreshold && !canRefresh && !refreshing) {
      setCanRefresh(true);
    } else if (offsetY > -refreshThreshold && canRefresh) {
      setCanRefresh(false);
    }
  };

  const handleRelease = () => {
    if (canRefresh && !refreshing) {
      onRefresh();
    }
  };

  const opacity = pullAnim.interpolate({
    inputRange: [0, pullDistance],
    outputRange: [0, 1],
    extrapolate: 'clamp',
  });

  const rotate = pullAnim.interpolate({
    inputRange: [0, pullDistance],
    outputRange: ['0deg', '360deg'],
    extrapolate: 'clamp',
  });

  return (
    <View style={styles.container}>
      <Animated.View
        style={[
          styles.refreshIndicator,
          {
            opacity,
            transform: [
              { translateY: pullAnim },
              { rotate },
            ],
          },
        ]}
        pointerEvents="none"
      >
        {refreshing ? (
          <ActivityIndicator size="small" color={theme.primary} />
        ) : (
          <Text style={[styles.refreshText, { color: theme.primary }]}>
            ↓ Suelta para actualizar
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
    height: 80,
    justifyContent: 'center',
    alignItems: 'center',
    zIndex: 1000,
  },
  refreshText: {
    ...typography.caption,
    fontWeight: '600',
  },
});

