import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  RefreshControl,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useRouter } from 'expo-router';
import { useAuthStore } from '@/store/auth-store';
import { useAnalytics, useQuota } from '@/hooks/use-analytics';
import { Button } from '@/components/ui/button';
import { Loading } from '@/components/ui/loading';

export default function HomeScreen() {
  const router = useRouter();
  const { user } = useAuthStore();
  const { data: analytics, isLoading: analyticsLoading, refetch: refetchAnalytics } = useAnalytics();
  const { data: quota, isLoading: quotaLoading, refetch: refetchQuota } = useQuota();
  const [refreshing, setRefreshing] = React.useState(false);

  const onRefresh = React.useCallback(async () => {
    setRefreshing(true);
    await Promise.all([refetchAnalytics(), refetchQuota()]);
    setRefreshing(false);
  }, [refetchAnalytics, refetchQuota]);

  if (analyticsLoading || quotaLoading) {
    return <Loading message="Loading..." />;
  }

  return (
    <SafeAreaView style={styles.container} edges={['top']}>
      <ScrollView
        style={styles.scrollView}
        contentContainerStyle={styles.content}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
        }
      >
        <View style={styles.header}>
          <Text style={styles.greeting}>Welcome back, {user?.email}</Text>
          <Text style={styles.subtitle}>Create amazing videos with AI</Text>
        </View>

        <View style={styles.statsContainer}>
          <View style={styles.statCard}>
            <Text style={styles.statValue}>
              {analytics?.metrics.total_videos || 0}
            </Text>
            <Text style={styles.statLabel}>Total Videos</Text>
          </View>
          <View style={styles.statCard}>
            <Text style={styles.statValue}>
              {analytics?.metrics.completed_videos || 0}
            </Text>
            <Text style={styles.statLabel}>Completed</Text>
          </View>
          {quota && (
            <View style={styles.statCard}>
              <Text style={styles.statValue}>
                {quota.videos_generated}/{quota.videos_limit}
              </Text>
              <Text style={styles.statLabel}>Quota</Text>
            </View>
          )}
        </View>

        <View style={styles.actionsContainer}>
          <Button
            title="Create New Video"
            onPress={() => router.push('/video-generation')}
            variant="primary"
            size="large"
            style={styles.actionButton}
          />
          <Button
            title="View My Videos"
            onPress={() => router.push('/(tabs)/my-videos')}
            variant="outline"
            size="large"
            style={styles.actionButton}
          />
          <Button
            title="Browse Templates"
            onPress={() => router.push('/(tabs)/templates')}
            variant="outline"
            size="large"
            style={styles.actionButton}
          />
        </View>

        {analytics && analytics.usage_statistics.videos_per_day.length > 0 && (
          <View style={styles.recentActivity}>
            <Text style={styles.sectionTitle}>Recent Activity</Text>
            <Text style={styles.activityText}>
              {analytics.usage_statistics.videos_per_day.length} videos generated this week
            </Text>
          </View>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#FFFFFF',
  },
  scrollView: {
    flex: 1,
  },
  content: {
    padding: 20,
  },
  header: {
    marginBottom: 24,
  },
  greeting: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#000000',
    marginBottom: 8,
  },
  subtitle: {
    fontSize: 16,
    color: '#666666',
  },
  statsContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 24,
  },
  statCard: {
    flex: 1,
    backgroundColor: '#F5F5F5',
    borderRadius: 12,
    padding: 16,
    marginHorizontal: 4,
    alignItems: 'center',
  },
  statValue: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#007AFF',
    marginBottom: 4,
  },
  statLabel: {
    fontSize: 12,
    color: '#666666',
    textAlign: 'center',
  },
  actionsContainer: {
    marginBottom: 24,
  },
  actionButton: {
    marginBottom: 12,
  },
  recentActivity: {
    backgroundColor: '#F5F5F5',
    borderRadius: 12,
    padding: 16,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#000000',
    marginBottom: 8,
  },
  activityText: {
    fontSize: 14,
    color: '#666666',
  },
});


