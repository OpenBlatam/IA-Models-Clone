import React, { useCallback, useMemo } from 'react';
import { View, Text, StyleSheet, ScrollView, RefreshControl, TouchableOpacity } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useQuery } from '@tanstack/react-query';
import { Ionicons } from '@expo/vector-icons';
import { useAuthStore } from '@/store/authStore';
import { useAppStore } from '@/store/appStore';
import { apiService } from '@/services/api';
import { useTheme } from '@/theme/theme';
import { Loading } from '@/components/ui/Loading';
import { Card } from '@/components/ui/Card';
import { ProgressCard } from '@/components/dashboard/ProgressCard';
import { StatisticsCard } from '@/components/dashboard/StatisticsCard';
import type { DashboardData } from '@/types';

export default function DashboardScreen() {
  const { user } = useAuthStore();
  const { setDashboardData, setGamificationProgress } = useAppStore();
  const theme = useTheme();

  const { data, isLoading, refetch, isRefetching } = useQuery<DashboardData>({
    queryKey: ['dashboard', user?.id],
    queryFn: async () => {
      if (!user?.id) throw new Error('No user ID');
      const response = await apiService.getDashboard(user.id);
      if (response.data) {
        setDashboardData(response.data);
        setGamificationProgress(response.data.gamification);
        return response.data;
      }
      throw new Error(response.error || 'Failed to load dashboard');
    },
    enabled: !!user?.id,
    staleTime: 5 * 60 * 1000, // 5 minutes
  });

  const onRefresh = useCallback(async () => {
    await refetch();
  }, [refetch]);

  const containerStyle = useMemo(
    () => [styles.container, { backgroundColor: theme.colors.background }],
    [theme.colors.background]
  );

  if (isLoading && !data) {
    return (
      <SafeAreaView style={containerStyle} edges={['top']}>
        <Loading fullScreen message="Loading dashboard..." />
      </SafeAreaView>
    );
  }

  const gamification = data?.gamification;
  const statistics = data?.statistics;

  return (
    <SafeAreaView style={containerStyle} edges={['top']}>
      <ScrollView
        style={styles.scrollView}
        refreshControl={
          <RefreshControl refreshing={isRefetching} onRefresh={onRefresh} tintColor={theme.colors.primary} />
        }
        showsVerticalScrollIndicator={false}
      >
        {/* Header */}
        <View style={[styles.header, { backgroundColor: theme.colors.card }]}>
          <View accessibilityRole="text">
            <Text style={[styles.greeting, { color: theme.colors.textSecondary }]}>
              Welcome back!
            </Text>
            <Text style={[styles.username, { color: theme.colors.text }]}>
              {user?.username || 'User'}
            </Text>
          </View>
          <TouchableOpacity
            style={styles.notificationButton}
            accessibilityRole="button"
            accessibilityLabel="Notifications"
            accessibilityHint="View your notifications"
          >
            <Ionicons name="notifications-outline" size={24} color={theme.colors.text} />
            {data?.notifications && data.notifications.filter((n) => !n.read).length > 0 && (
              <View style={[styles.badge, { backgroundColor: theme.colors.error }]}>
                <Text style={styles.badgeText}>
                  {data.notifications.filter((n) => !n.read).length}
                </Text>
              </View>
            )}
          </TouchableOpacity>
        </View>

        {/* Gamification Card */}
        {gamification && <ProgressCard progress={gamification} />}

        {/* Statistics */}
        {statistics && <StatisticsCard statistics={statistics} />}

        {/* Roadmap Progress */}
        {data?.roadmap && (
          <Card>
            <View style={styles.cardHeader}>
              <Text style={[styles.cardTitle, { color: theme.colors.text }]}>Roadmap Progress</Text>
              <Text style={[styles.percentage, { color: theme.colors.secondary }]}>
                {data.roadmap.progress_percentage.toFixed(0)}%
              </Text>
            </View>
            <View style={styles.progressContainer}>
              <View style={[styles.progressBar, { backgroundColor: theme.colors.border }]}>
                <View
                  style={[
                    styles.progressFill,
                    {
                      width: `${data.roadmap.progress_percentage}%`,
                      backgroundColor: theme.colors.secondary,
                    },
                  ]}
                  accessibilityRole="progressbar"
                  accessibilityValue={{
                    min: 0,
                    max: data.roadmap.total_steps,
                    now: data.roadmap.completed_steps,
                    text: `${data.roadmap.completed_steps} of ${data.roadmap.total_steps} steps completed`,
                  }}
                />
              </View>
              <Text style={[styles.progressText, { color: theme.colors.textSecondary }]}>
                {data.roadmap.completed_steps} of {data.roadmap.total_steps} steps completed
              </Text>
            </View>
          </Card>
        )}

        {/* Quick Actions */}
        <Card>
          <Text style={[styles.cardTitle, { color: theme.colors.text }]}>Quick Actions</Text>
          <View style={styles.actionsGrid}>
            {[
              { icon: 'search' as const, label: 'Search Jobs' },
              { icon: 'document-text' as const, label: 'Analyze CV' },
              { icon: 'chatbubbles' as const, label: 'AI Mentor' },
              { icon: 'videocam' as const, label: 'Practice Interview' },
            ].map((action, index) => (
              <TouchableOpacity
                key={index}
                style={[styles.actionButton, { backgroundColor: theme.colors.surface }]}
                accessibilityRole="button"
                accessibilityLabel={action.label}
              >
                <Ionicons name={action.icon} size={24} color={theme.colors.primary} />
                <Text style={[styles.actionText, { color: theme.colors.text }]}>
                  {action.label}
                </Text>
              </TouchableOpacity>
            ))}
          </View>
        </Card>

        {/* Recent Recommendations */}
        {data?.recommendations && data.recommendations.next_steps.length > 0 && (
          <Card>
            <Text style={[styles.cardTitle, { color: theme.colors.text }]}>
              Recommended Next Steps
            </Text>
            {data.recommendations.next_steps.slice(0, 3).map((step, index) => (
              <View
                key={index}
                style={styles.recommendationItem}
                accessibilityRole="text"
                accessibilityLabel={`Recommended step: ${step}`}
              >
                <Ionicons name="checkmark-circle" size={20} color={theme.colors.secondary} />
                <Text style={[styles.recommendationText, { color: theme.colors.text }]}>
                  {step}
                </Text>
              </View>
            ))}
          </Card>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  scrollView: {
    flex: 1,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 20,
  },
  greeting: {
    fontSize: 14,
  },
  username: {
    fontSize: 24,
    fontWeight: 'bold',
    marginTop: 4,
  },
  notificationButton: {
    position: 'relative',
    padding: 8,
  },
  badge: {
    position: 'absolute',
    top: 4,
    right: 4,
    borderRadius: 10,
    minWidth: 20,
    height: 20,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 4,
  },
  badgeText: {
    color: '#fff',
    fontSize: 12,
    fontWeight: 'bold',
  },
  cardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  cardTitle: {
    fontSize: 20,
    fontWeight: 'bold',
  },
  percentage: {
    fontSize: 18,
    fontWeight: 'bold',
  },
  progressContainer: {
    marginTop: 8,
  },
  progressBar: {
    height: 8,
    borderRadius: 4,
    overflow: 'hidden',
    marginBottom: 8,
  },
  progressFill: {
    height: '100%',
    borderRadius: 4,
  },
  progressText: {
    fontSize: 12,
  },
  actionsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
  },
  actionButton: {
    width: '48%',
    padding: 16,
    borderRadius: 12,
    marginBottom: 12,
    alignItems: 'center',
  },
  actionText: {
    fontSize: 12,
    marginTop: 8,
    fontWeight: '500',
  },
  recommendationItem: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },
  recommendationText: {
    fontSize: 14,
    marginLeft: 12,
    flex: 1,
  },
});

