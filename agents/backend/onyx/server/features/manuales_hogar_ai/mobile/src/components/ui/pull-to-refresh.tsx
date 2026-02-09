/**
 * Pull to Refresh
 * ===============
 * Enhanced pull to refresh component
 */

import { View, Text, StyleSheet, RefreshControl } from 'react-native';
import { useApp } from '@/lib/context/app-context';

interface PullToRefreshProps {
  refreshing: boolean;
  onRefresh: () => void;
  children: React.ReactNode;
}

export function PullToRefresh({ refreshing, onRefresh, children }: PullToRefreshProps) {
  const { state } = useApp();
  const colors = state.colors;

  return (
    <RefreshControl
      refreshing={refreshing}
      onRefresh={onRefresh}
      tintColor={colors.tint}
      colors={[colors.tint]}
      progressBackgroundColor={colors.background}
    >
      {children}
    </RefreshControl>
  );
}



