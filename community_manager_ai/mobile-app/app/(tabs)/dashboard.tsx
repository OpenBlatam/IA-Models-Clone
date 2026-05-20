import { View, Text, ScrollView, StyleSheet, RefreshControl } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useDashboardOverview, useUpcomingPosts, useDashboardEngagement } from '@/hooks/useApi';
import { ActivityIndicator } from 'react-native';
import { Ionicons } from '@expo/vector-icons';

export default function DashboardScreen() {
  const { data: overview, isLoading: overviewLoading, refetch: refetchOverview } = useDashboardOverview(7);
  const { data: upcomingPosts, isLoading: postsLoading, refetch: refetchPosts } = useUpcomingPosts(10);
  const { data: engagement, isLoading: engagementLoading, refetch: refetchEngagement } = useDashboardEngagement(7);

  const refreshing = overviewLoading || postsLoading || engagementLoading;

  const handleRefresh = () => {
    refetchOverview();
    refetchPosts();
    refetchEngagement();
  };

  if (overviewLoading && !overview) {
    return (
      <SafeAreaView style={styles.container}>
        <ActivityIndicator size="large" color="#0ea5e9" />
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container} edges={['top']}>
      <ScrollView
        style={styles.scrollView}
        refreshControl={<RefreshControl refreshing={refreshing} onRefresh={handleRefresh} />}
      >
        <View style={styles.content}>
          <Text style={styles.title}>Dashboard</Text>

          {/* Stats Grid */}
          <View style={styles.statsGrid}>
            <StatCard
              icon="document-text"
              label="Total Posts"
              value={overview?.total_posts || 0}
              color="#0ea5e9"
            />
            <StatCard
              icon="time"
              label="Scheduled"
              value={overview?.scheduled_posts || 0}
              color="#f59e0b"
            />
            <StatCard
              icon="checkmark-circle"
              label="Published"
              value={overview?.published_posts || 0}
              color="#10b981"
            />
            <StatCard
              icon="share-social"
              label="Platforms"
              value={overview?.connected_platforms || 0}
              color="#8b5cf6"
            />
          </View>

          {/* Engagement Card */}
          {engagement && (
            <View style={styles.card}>
              <Text style={styles.cardTitle}>Engagement (7 days)</Text>
              <Text style={styles.engagementValue}>{engagement.total_engagement.toLocaleString()}</Text>
              <Text style={styles.engagementRate}>
                Avg Rate: {(engagement.average_engagement_rate * 100).toFixed(2)}%
              </Text>
            </View>
          )}

          {/* Upcoming Posts */}
          <View style={styles.card}>
            <Text style={styles.cardTitle}>Upcoming Posts</Text>
            {postsLoading ? (
              <ActivityIndicator size="small" color="#0ea5e9" />
            ) : upcomingPosts && upcomingPosts.length > 0 ? (
              upcomingPosts.slice(0, 5).map((post) => (
                <View key={post.post_id} style={styles.postItem}>
                  <Text style={styles.postContent} numberOfLines={2}>
                    {post.content}
                  </Text>
                  <Text style={styles.postMeta}>
                    {post.platforms.join(', ')} • {post.scheduled_time ? new Date(post.scheduled_time).toLocaleDateString() : 'No date'}
                  </Text>
                </View>
              ))
            ) : (
              <Text style={styles.emptyText}>No upcoming posts</Text>
            )}
          </View>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

function StatCard({ icon, label, value, color }: { icon: string; label: string; value: number; color: string }) {
  return (
    <View style={[styles.statCard, { borderLeftColor: color }]}>
      <Ionicons name={icon as any} size={24} color={color} />
      <Text style={styles.statValue}>{value}</Text>
      <Text style={styles.statLabel}>{label}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  scrollView: {
    flex: 1,
  },
  content: {
    padding: 16,
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#1f2937',
    marginBottom: 20,
  },
  statsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 12,
    marginBottom: 20,
  },
  statCard: {
    flex: 1,
    minWidth: '45%',
    backgroundColor: '#fff',
    padding: 16,
    borderRadius: 12,
    borderLeftWidth: 4,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  statValue: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#1f2937',
    marginTop: 8,
  },
  statLabel: {
    fontSize: 12,
    color: '#6b7280',
    marginTop: 4,
  },
  card: {
    backgroundColor: '#fff',
    padding: 16,
    borderRadius: 12,
    marginBottom: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  cardTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#1f2937',
    marginBottom: 12,
  },
  engagementValue: {
    fontSize: 32,
    fontWeight: 'bold',
    color: '#0ea5e9',
    marginBottom: 4,
  },
  engagementRate: {
    fontSize: 14,
    color: '#6b7280',
  },
  postItem: {
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#f3f4f6',
  },
  postContent: {
    fontSize: 14,
    color: '#1f2937',
    marginBottom: 4,
  },
  postMeta: {
    fontSize: 12,
    color: '#6b7280',
  },
  emptyText: {
    fontSize: 14,
    color: '#9ca3af',
    fontStyle: 'italic',
    textAlign: 'center',
    padding: 20,
  },
});


