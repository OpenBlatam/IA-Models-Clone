import React, { ReactNode } from 'react';
import { RefreshControl, ScrollView, StyleSheet } from 'react-native';
import { COLORS } from '../../constants/config';

interface PullToRefreshProps {
  children: ReactNode;
  refreshing: boolean;
  onRefresh: () => void;
  style?: unknown;
}

export function PullToRefresh({
  children,
  refreshing,
  onRefresh,
  style,
}: PullToRefreshProps) {
  return (
    <ScrollView
      style={[styles.container, style]}
      refreshControl={
        <RefreshControl
          refreshing={refreshing}
          onRefresh={onRefresh}
          tintColor={COLORS.primary}
          colors={[COLORS.primary]}
        />
      }
    >
      {children}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: COLORS.background,
  },
});

