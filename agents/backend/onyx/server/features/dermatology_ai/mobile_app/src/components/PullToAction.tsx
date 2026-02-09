import React, { useState, useRef } from 'react';
import {
  View,
  ScrollView,
  StyleSheet,
  Animated,
  RefreshControl,
} from 'react-native';
import { useTheme } from '../context/ThemeContext';

interface PullToActionProps {
  children: React.ReactNode;
  onRefresh?: () => Promise<void> | void;
  onPullDown?: () => void;
  pullDownThreshold?: number;
  refreshControl?: boolean;
}

const PullToAction: React.FC<PullToActionProps> = ({
  children,
  onRefresh,
  onPullDown,
  pullDownThreshold = 100,
  refreshControl = true,
}) => {
  const { colors } = useTheme();
  const [refreshing, setRefreshing] = useState(false);
  const scrollY = useRef(new Animated.Value(0)).current;

  const handleRefresh = async () => {
    setRefreshing(true);
    try {
      await onRefresh?.();
    } finally {
      setRefreshing(false);
    }
  };

  const handleScroll = Animated.event(
    [{ nativeEvent: { contentOffset: { y: scrollY } } }],
    {
      useNativeDriver: false,
      listener: (event: any) => {
        const offset = event.nativeEvent.contentOffset.y;
        if (offset > pullDownThreshold && onPullDown) {
          onPullDown();
        }
      },
    }
  );

  return (
    <ScrollView
      style={styles.container}
      onScroll={handleScroll}
      scrollEventThrottle={16}
      refreshControl={
        refreshControl && onRefresh ? (
          <RefreshControl
            refreshing={refreshing}
            onRefresh={handleRefresh}
            tintColor={colors.primary}
            colors={[colors.primary]}
          />
        ) : undefined
      }
    >
      {children}
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
});

export default PullToAction;

