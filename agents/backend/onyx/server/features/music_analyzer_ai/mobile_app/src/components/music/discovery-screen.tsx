import React, { useState, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  FlatList,
  TouchableOpacity,
  ScrollView,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useRouter } from 'expo-router';
import {
  useDiscoverUnderground,
  useDiscoverFresh,
  useDiscoverSimilarArtists,
} from '../../hooks/use-music-api';
import { TrackCard } from './track-card';
import { LoadingSpinner } from '../common/loading-spinner';
import { ErrorMessage } from '../common/error-message';
import { EmptyState } from '../common/empty-state';
import { AnimatedView } from '../common/animated-view';
import { COLORS, SPACING, TYPOGRAPHY, BORDER_RADIUS } from '../../constants/config';
import { useTranslation } from '../../hooks/use-translation';
import type { Track } from '../../types/api';

type DiscoveryType = 'underground' | 'fresh' | 'similar';

export function DiscoveryScreen() {
  const { t } = useTranslation();
  const router = useRouter();
  const [activeTab, setActiveTab] = useState<DiscoveryType>('underground');
  const [selectedArtistId, setSelectedArtistId] = useState<string>('');

  const {
    data: undergroundData,
    isLoading: isLoadingUnderground,
    error: undergroundError,
    refetch: refetchUnderground,
  } = useDiscoverUnderground();

  const {
    data: freshData,
    isLoading: isLoadingFresh,
    error: freshError,
    refetch: refetchFresh,
  } = useDiscoverFresh();

  const {
    data: similarData,
    isLoading: isLoadingSimilar,
    error: similarError,
    refetch: refetchSimilar,
  } = useDiscoverSimilarArtists(selectedArtistId);

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
      <AnimatedView delay={100}>
        <TrackCard track={item} onPress={handleTrackPress} />
      </AnimatedView>
    ),
    [handleTrackPress]
  );

  const keyExtractor = useCallback((item: Track) => item.id, []);

  const getCurrentData = () => {
    switch (activeTab) {
      case 'underground':
        return undergroundData;
      case 'fresh':
        return freshData;
      case 'similar':
        return similarData;
      default:
        return null;
    }
  };

  const getCurrentLoading = () => {
    switch (activeTab) {
      case 'underground':
        return isLoadingUnderground;
      case 'fresh':
        return isLoadingFresh;
      case 'similar':
        return isLoadingSimilar;
      default:
        return false;
    }
  };

  const getCurrentError = () => {
    switch (activeTab) {
      case 'underground':
        return undergroundError;
      case 'fresh':
        return freshError;
      case 'similar':
        return similarError;
      default:
        return null;
    }
  };

  const handleRetry = useCallback(() => {
    switch (activeTab) {
      case 'underground':
        refetchUnderground();
        break;
      case 'fresh':
        refetchFresh();
        break;
      case 'similar':
        refetchSimilar();
        break;
    }
  }, [activeTab, refetchUnderground, refetchFresh, refetchSimilar]);

  const currentData = getCurrentData();
  const isLoading = getCurrentLoading();
  const error = getCurrentError();

  return (
    <SafeAreaView style={styles.container} edges={['top', 'bottom']}>
      <View style={styles.header}>
        <Text style={styles.title} accessibilityRole="header">
          Discover Music
        </Text>
      </View>

      <View style={styles.tabs}>
        <TouchableOpacity
          style={[styles.tab, activeTab === 'underground' && styles.tabActive]}
          onPress={() => setActiveTab('underground')}
          accessibilityRole="button"
          accessibilityLabel="Underground music"
        >
          <Text
            style={[
              styles.tabText,
              activeTab === 'underground' && styles.tabTextActive,
            ]}
          >
            Underground
          </Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.tab, activeTab === 'fresh' && styles.tabActive]}
          onPress={() => setActiveTab('fresh')}
          accessibilityRole="button"
          accessibilityLabel="Fresh music"
        >
          <Text
            style={[styles.tabText, activeTab === 'fresh' && styles.tabTextActive]}
          >
            Fresh
          </Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.tab, activeTab === 'similar' && styles.tabActive]}
          onPress={() => setActiveTab('similar')}
          accessibilityRole="button"
          accessibilityLabel="Similar artists"
        >
          <Text
            style={[
              styles.tabText,
              activeTab === 'similar' && styles.tabTextActive,
            ]}
          >
            Similar
          </Text>
        </TouchableOpacity>
      </View>

      <View style={styles.content}>
        {isLoading && <LoadingSpinner />}
        {error && (
          <ErrorMessage
            message={error.message || 'Failed to load discovery'}
            onRetry={handleRetry}
            retryLabel={t('common.retry')}
          />
        )}
        {currentData && currentData.results.length === 0 && (
          <EmptyState
            icon="🔍"
            title="No results"
            message="Try a different discovery option"
          />
        )}
        {currentData && currentData.results.length > 0 && (
          <FlatList
            data={currentData.results}
            keyExtractor={keyExtractor}
            renderItem={renderTrack}
            contentContainerStyle={styles.listContent}
            removeClippedSubviews={true}
            maxToRenderPerBatch={10}
            windowSize={10}
          />
        )}
      </View>
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
    ...TYPOGRAPHY.h1,
    color: COLORS.text,
  },
  tabs: {
    flexDirection: 'row',
    padding: SPACING.sm,
    borderBottomWidth: 1,
    borderBottomColor: COLORS.surfaceLight,
  },
  tab: {
    flex: 1,
    padding: SPACING.sm,
    alignItems: 'center',
    borderRadius: BORDER_RADIUS.md,
    marginHorizontal: SPACING.xs,
  },
  tabActive: {
    backgroundColor: COLORS.primary,
  },
  tabText: {
    ...TYPOGRAPHY.body,
    color: COLORS.textSecondary,
  },
  tabTextActive: {
    color: COLORS.text,
    fontWeight: '600',
  },
  content: {
    flex: 1,
    padding: SPACING.md,
  },
  listContent: {
    paddingBottom: SPACING.lg,
  },
});

