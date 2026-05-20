import React, { useCallback, useMemo } from 'react';
import {
  View,
  Text,
  StyleSheet,
  FlatList,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useRouter, useLocalSearchParams } from 'expo-router';
import { useTrackRecommendations } from '../hooks/use-music-analysis';
import { TrackCard } from '../components/music/track-card';
import { LoadingSpinner } from '../components/common/loading-spinner';
import { ErrorMessage } from '../components/common/error-message';
import { EmptyState } from '../components/common/empty-state';
import { COLORS, SPACING, TYPOGRAPHY } from '../constants/config';
import { useTranslation } from '../hooks/use-translation';
import type { Track } from '../types/api';

export default function RecommendationsPage() {
  const { t } = useTranslation();
  const router = useRouter();
  const params = useLocalSearchParams<{ trackId: string }>();
  const trackId = params.trackId || '';

  const {
    data,
    isLoading,
    error,
    refetch,
  } = useTrackRecommendations(trackId);

  const handleTrackPress = useCallback(
    (track: Track) => {
      router.push({
        pathname: '/analysis',
        params: {
          trackId: track.id,
          trackName: track.name,
          artists: JSON.stringify(track.artists),
        },
      });
    },
    [router]
  );

  const renderTrack = useCallback(
    ({ item }: { item: Track }) => (
      <TrackCard track={item} onPress={handleTrackPress} />
    ),
    [handleTrackPress]
  );

  const keyExtractor = useCallback((item: Track) => item.id, []);

  const hasRecommendations = useMemo(
    () => data && data.recommendations.length > 0,
    [data]
  );

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
          message={error.message || t('errors.networkError')}
          onRetry={() => refetch()}
          retryLabel={t('common.retry')}
        />
      </SafeAreaView>
    );
  }

  if (!hasRecommendations) {
    return (
      <SafeAreaView style={styles.container} edges={['top', 'bottom']}>
        <EmptyState
          icon="🎵"
          title={t('common.recommendations')}
          message="No recommendations available for this track."
        />
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container} edges={['top', 'bottom']}>
      <View style={styles.header}>
        <Text style={styles.title} accessibilityRole="header">
          {t('common.recommendations')}
        </Text>
        <Text style={styles.subtitle}>
          {data?.total || 0} recommendations found
        </Text>
      </View>
      <FlatList
        data={data?.recommendations || []}
        keyExtractor={keyExtractor}
        renderItem={renderTrack}
        contentContainerStyle={styles.listContent}
        removeClippedSubviews={true}
        maxToRenderPerBatch={10}
        windowSize={10}
        initialNumToRender={10}
      />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: COLORS.background,
  },
  header: {
    padding: SPACING.md,
    borderBottomWidth: 1,
    borderBottomColor: COLORS.surfaceLight,
  },
  title: {
    ...TYPOGRAPHY.h2,
    color: COLORS.text,
    marginBottom: SPACING.xs,
  },
  subtitle: {
    ...TYPOGRAPHY.bodySmall,
    color: COLORS.textSecondary,
  },
  listContent: {
    padding: SPACING.md,
  },
});

