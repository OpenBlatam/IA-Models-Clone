import React, { useCallback, useMemo } from 'react';
import {
  View,
  StyleSheet,
  FlatList,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useRouter } from 'expo-router';
import { useMusic } from '../../stores/music';
import { TrackCard } from '../../components/music/track-card';
import { EmptyState } from '../../components/common/empty-state';
import { COLORS, SPACING } from '../../constants/config';
import { useTranslation } from '../../hooks/use-translation';
import type { Track } from '../../types/api';

export default function FavoritesTab() {
  const { t } = useTranslation();
  const router = useRouter();
  const { favorites, addFavorite, removeFavorite } = useMusic();

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

  const handleToggleFavorite = useCallback(
    (track: Track) => {
      removeFavorite(track.id);
    },
    [removeFavorite]
  );

  const renderTrack = useCallback(
    ({ item }: { item: Track }) => (
      <TrackCard
        track={item}
        onPress={handleTrackPress}
        isFavorite={true}
        onToggleFavorite={handleToggleFavorite}
      />
    ),
    [handleTrackPress, handleToggleFavorite]
  );

  const keyExtractor = useCallback((item: Track) => item.id, []);

  const hasFavorites = useMemo(() => favorites.length > 0, [favorites]);

  if (!hasFavorites) {
    return (
      <SafeAreaView style={styles.container} edges={['top', 'bottom']}>
        <EmptyState
          icon="❤️"
          title={t('common.favorites')}
          message="You haven't added any favorites yet. Search for tracks and add them to your favorites!"
        />
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container} edges={['top', 'bottom']}>
      <FlatList
        data={favorites}
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
  listContent: {
    padding: SPACING.md,
  },
});

