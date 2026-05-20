import { useState } from 'react';
import { View, Text, ScrollView, StyleSheet, RefreshControl, TouchableOpacity } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { usePlatformAnalytics, useBestPerformingPosts } from '@/hooks/useApi';
import { ActivityIndicator } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { PLATFORMS } from '@/utils/constants';

const PLATFORM_IDS = PLATFORMS.map((p) => p.id);

export default function AnalyticsScreen() {
  const [selectedPlatform, setSelectedPlatform] = useState<string>(PLATFORM_IDS[0]);
  const { data: analytics, isLoading, refetch } = usePlatformAnalytics(selectedPlatform, 7);
  const { data: bestPerforming } = useBestPerformingPosts(selectedPlatform, 5);

  return (
    <SafeAreaView style={styles.container} edges={['top']}>
      <ScrollView
        style={styles.scrollView}
        refreshControl={<RefreshControl refreshing={isLoading} onRefresh={refetch} />}
      >
        <View style={styles.content}>
          <Text style={styles.title}>Analytics</Text>

          {/* Platform Selector */}
          <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.platformSelector}>
            {PLATFORM_IDS.map((platformId) => {
              const platform = PLATFORMS.find((p) => p.id === platformId);
              return (
                <TouchableOpacity
                  key={platformId}
                  style={[
                    styles.platformButton,
                    selectedPlatform === platformId && styles.platformButtonActive,
                  ]}
                  onPress={() => setSelectedPlatform(platformId)}
                >
                  <Text
                    style={[
                      styles.platformButtonText,
                      selectedPlatform === platformId && styles.platformButtonTextActive,
                    ]}
                  >
                    {platform?.name || platformId}
                  </Text>
                </TouchableOpacity>
              );
            })}
          </ScrollView>

          {isLoading && !analytics ? (
            <ActivityIndicator size="large" color="#0ea5e9" style={styles.loader} />
          ) : analytics ? (
            <>
              {/* Stats Cards */}
              <View style={styles.statsGrid}>
                <StatCard
                  icon="document-text"
                  label="Total Posts"
                  value={analytics.total_posts}
                  color="#0ea5e9"
                />
                <StatCard
                  icon="heart"
                  label="Total Engagement"
                  value={analytics.total_engagement.toLocaleString()}
                  color="#e91e63"
                />
                <StatCard
                  icon="trending-up"
                  label="Avg Engagement Rate"
                  value={`${(analytics.average_engagement_rate * 100).toFixed(2)}%`}
                  color="#10b981"
                />
              </View>

              {/* Chart Placeholder */}
              <View style={styles.chartCard}>
                <Text style={styles.cardTitle}>Engagement Trends</Text>
                <View style={styles.chartPlaceholder}>
                  <Ionicons name="bar-chart" size={48} color="#d1d5db" />
                  <Text style={styles.chartPlaceholderText}>Chart visualization</Text>
                </View>
              </View>

              {/* Best Performing Posts */}
              {bestPerforming && bestPerforming.length > 0 && (
                <View style={styles.card}>
                  <Text style={styles.cardTitle}>Best Performing Posts</Text>
                  {bestPerforming.map((post, index) => (
                    <View key={post.post_id} style={styles.postItem}>
                      <View style={styles.postRank}>
                        <Text style={styles.rankNumber}>{index + 1}</Text>
                      </View>
                      <View style={styles.postContent}>
                        <Text style={styles.postText} numberOfLines={2}>
                          {post.content}
                        </Text>
                        <Text style={styles.postMeta}>
                          {post.platforms.join(', ')} • {post.status}
                        </Text>
                      </View>
                    </View>
                  ))}
                </View>
              )}
            </>
          ) : (
            <View style={styles.empty}>
              <Ionicons name="stats-chart-outline" size={64} color="#d1d5db" />
              <Text style={styles.emptyText}>No analytics data available</Text>
            </View>
          )}
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

function StatCard({ icon, label, value, color }: { icon: string; label: string; value: string | number; color: string }) {
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
  platformSelector: {
    marginBottom: 20,
  },
  platformButton: {
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 20,
    backgroundColor: '#f3f4f6',
    marginRight: 8,
  },
  platformButtonActive: {
    backgroundColor: '#0ea5e9',
  },
  platformButtonText: {
    fontSize: 14,
    color: '#6b7280',
  },
  platformButtonTextActive: {
    color: '#fff',
    fontWeight: '600',
  },
  loader: {
    marginTop: 40,
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
    fontSize: 20,
    fontWeight: 'bold',
    color: '#1f2937',
    marginTop: 8,
  },
  statLabel: {
    fontSize: 12,
    color: '#6b7280',
    marginTop: 4,
  },
  chartCard: {
    backgroundColor: '#fff',
    padding: 16,
    borderRadius: 12,
    marginBottom: 20,
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
  chartPlaceholder: {
    height: 200,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#f9fafb',
    borderRadius: 8,
  },
  chartPlaceholderText: {
    marginTop: 8,
    fontSize: 14,
    color: '#9ca3af',
  },
  card: {
    backgroundColor: '#fff',
    padding: 16,
    borderRadius: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  postItem: {
    flexDirection: 'row',
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#f3f4f6',
  },
  postRank: {
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: '#0ea5e9',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 12,
  },
  rankNumber: {
    color: '#fff',
    fontWeight: 'bold',
    fontSize: 14,
  },
  postContent: {
    flex: 1,
  },
  postText: {
    fontSize: 14,
    color: '#1f2937',
    marginBottom: 4,
  },
  postMeta: {
    fontSize: 12,
    color: '#6b7280',
  },
  empty: {
    alignItems: 'center',
    justifyContent: 'center',
    padding: 40,
    marginTop: 60,
  },
  emptyText: {
    fontSize: 16,
    color: '#9ca3af',
    marginTop: 16,
  },
});

