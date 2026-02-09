import React, { useMemo, useCallback } from 'react';
import {
  View,
  Text,
  ScrollView,
  RefreshControl,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { ProgressCard, LoadingSpinner } from '@/components';
import { useDashboard } from '@/hooks/api';
import { useAuthStore } from '@/store/auth-store';
import { useColors } from '@/theme/colors';
import { format } from 'date-fns';
import { useDashboardStyles } from './dashboard-screen.styles';
import { DashboardHeader } from './dashboard-header';
import { DashboardCards } from './dashboard-cards';
import { DashboardSections } from './dashboard-sections';

export function DashboardScreen(): JSX.Element {
  const colors = useColors();
  const { user } = useAuthStore();
  const { data: dashboard, isLoading, refetch, isRefetching } = useDashboard(
    user?.user_id || null
  );
  const styles = useDashboardStyles(colors);

  const onRefresh = useCallback(() => {
    refetch();
  }, [refetch]);

  if (isLoading) {
    return <LoadingSpinner />;
  }

  if (!dashboard) {
    return (
      <SafeAreaView style={styles.container} edges={['top', 'bottom']}>
        <View style={styles.emptyContainer}>
          <Text style={styles.emptyText}>
            No hay datos disponibles
          </Text>
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container} edges={['top']}>
      <ScrollView
        style={styles.scrollView}
        refreshControl={
          <RefreshControl refreshing={isRefetching} onRefresh={onRefresh} />
        }
        contentContainerStyle={styles.scrollContent}
      >
        <DashboardHeader user={user} />
        <DashboardCards dashboard={dashboard} />
        <DashboardSections dashboard={dashboard} />
      </ScrollView>
    </SafeAreaView>
  );
}

