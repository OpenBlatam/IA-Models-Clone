import React, { useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  FlatList,
  TouchableOpacity,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useRouter } from 'expo-router';
import { useHistory, useHistoryStats } from '../hooks/use-music-api';
import { TrackCard } from '../components/music/track-card';
import { LoadingSpinner } from '../components/common/loading-spinner';
import { ErrorMessage } from '../components/common/error-message';
import { EmptyState } from '../components/common/empty-state';
import { COLORS, SPACING, TYPOGRAPHY } from '../constants/config';
import { useTranslation } from '../hooks/use-translation';
import { formatArtists } from '../utils/formatters';
import type { HistoryEntry } from '../types/api';

export default function HistoryPage() {
  const { t } = useTranslation();
  const router = useRouter();
  const { data: history, isLoading, error, refetch } = useHistory(50);
  const { data: stats } = useHistoryStats();

  const handleTrackPress = useCallback(
    (entry: HistoryEntry) => {
      router.push({
        pathname: '/analysis',
        params: {
          trackId: entry.track_id,
          trackName: entry.track_name,
          artists: JSON.stringify(entry.artists),
        },
      });
    },
    [router]
  );

  const renderHistoryItem = useCallback(
    ({ item }: { item: HistoryEntry }) => {
      const track = {
        id: item.track_id,
        name: item.track_name,
        artists: item.artists,
        album: '',
        duration_ms: 0,
        preview_url: null,
        external_urls: { spotify: '' },
        popularity: 0,
      };

      return (
        <TouchableOpacity onPress={() => handleTrackPress(item)}>
          <TrackCard track={track} onPress={() => handleTrackPress(item)} />
          <Text style={styles.date}>
            Analyzed: {new Date(item.analyzed_at).toLocaleDateString()}
          </Text>
        </TouchableOpacity>
      );
    },
    [handleTrackPress]
  );

  const keyExtractor = useCallback((item: HistoryEntry) => item.id, []);

  if (isLoading) {
    return (
      <SafeAreaView style={styles.container} edges={['top', 'bottom']}>
        <LoadingSpinner />
      </SafeAreaView>
    );
  }

  if (error) {
    return (
      <SafeAreaView style={styles.container} edges={['top', 'bottom']}>
        <ErrorMessage
          message={error.message || 'Failed to load history'}
          onRetry={() => refetch()}
        />
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container} edges={['top', 'bottom']}>
      {stats && (
        <View style={styles.statsContainer}>
          <View style={styles.statItem}>
            <Text style={styles.statValue}>{stats.stats.total_analyses}</Text>
            <Text style={styles.statLabel}>Total Analyses</Text>
          </View>
          <View style={styles.statItem}>
            <Text style={styles.statValue}>{stats.stats.unique_tracks}</Text>
            <Text style={styles.statLabel}>Unique Tracks</Text>
          </View>
        </View>
      )}

      {history && history.history.length === 0 ? (
        <EmptyState
          icon="📜"
          title="No History"
          message="Your analysis history will appear here"
        />
      ) : (
        <FlatList
          data={history?.history || []}
          keyExtractor={keyExtractor}
          renderItem={renderHistoryItem}
          contentContainerStyle={styles.listContent}
          removeClippedSubviews={true}
          maxToRenderPerBatch={10}
          windowSize={10}
        />
      )}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: COLORS.background,
  },
  statsContainer: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    padding: SPACING.md,
    backgroundColor: COLORS.surface,
    borderBottomWidth: 1,
    borderBottomColor: COLORS.surfaceLight,
  },
  statItem: {
    alignItems: 'center',
  },
  statValue: {
    ...TYPOGRAPHY.h2,
    color: COLORS.primary,
  },
  statLabel: {
    ...TYPOGRAPHY.caption,
    color: COLORS.textSecondary,
    marginTop: SPACING.xs,
  },
  listContent: {
    padding: SPACING.md,
  },
  date: {
    ...TYPOGRAPHY.caption,
    color: COLORS.textSecondary,
    marginTop: SPACING.xs,
    marginLeft: SPACING.md,
  },
});

