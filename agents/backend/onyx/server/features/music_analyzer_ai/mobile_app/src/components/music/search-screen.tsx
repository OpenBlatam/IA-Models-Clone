import React, { useState, useCallback, useMemo } from 'react';
import {
  View,
  Text,
  TextInput,
  StyleSheet,
  FlatList,
  Keyboard,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useSearchTracks } from '../../hooks/use-music-analysis';
import { useDebounce } from '../../hooks/use-debounce';
import { TrackCard } from './track-card';
import { LoadingSpinner } from '../common/loading-spinner';
import { ErrorMessage } from '../common/error-message';
import { EmptyState } from '../common/empty-state';
import { COLORS, SPACING, TYPOGRAPHY, BORDER_RADIUS } from '../../constants/config';
import type { Track } from '../../types/api';
import { useMusic } from '../../stores/music';
import { useTranslation } from '../../hooks/use-translation';
import { logError } from '../../utils/error-handler';

interface SearchScreenProps {
  onTrackSelect: (track: Track) => void;
}

export function SearchScreen({ onTrackSelect }: SearchScreenProps) {
  const { t } = useTranslation();
  const [searchQuery, setSearchQuery] = useState('');
  const debouncedQuery = useDebounce(searchQuery, 500);
  const { addRecentSearch, recentSearches, favorites, addFavorite, removeFavorite } = useMusic();
  const {
    data,
    isLoading,
    error,
    refetch,
  } = useSearchTracks(debouncedQuery, 20);

  const handleTrackPress = useCallback(
    (track: Track) => {
      addRecentSearch(track);
      onTrackSelect(track);
      Keyboard.dismiss();
    },
    [addRecentSearch, onTrackSelect]
  );

  const handleToggleFavorite = useCallback(
    (track: Track) => {
      const isFavorite = favorites.some((f) => f.id === track.id);
      if (isFavorite) {
        removeFavorite(track.id);
      } else {
        addFavorite(track);
      }
    },
    [favorites, addFavorite, removeFavorite]
  );

  const handleSearchChange = useCallback((text: string) => {
    setSearchQuery(text);
  }, []);

  const handleRetry = useCallback(() => {
    if (error) {
      logError(error, 'SearchScreen');
    }
    refetch();
  }, [error, refetch]);

  const renderTrack = useCallback(
    ({ item }: { item: Track }) => {
      const isFavorite = favorites.some((f) => f.id === item.id);
      return (
        <TrackCard
          track={item}
          onPress={handleTrackPress}
          isFavorite={isFavorite}
          onToggleFavorite={handleToggleFavorite}
        />
      );
    },
    [handleTrackPress, handleToggleFavorite, favorites]
  );

  const keyExtractor = useCallback((item: Track) => item.id, []);

  const hasResults = useMemo(
    () => data && data.results.length > 0,
    [data]
  );

  const showRecentSearches = useMemo(
    () => !debouncedQuery && recentSearches.length > 0,
    [debouncedQuery, recentSearches]
  );

  return (
    <SafeAreaView style={styles.container} edges={['top']}>
      <View style={styles.header}>
        <Text style={styles.title} accessibilityRole="header">
          {t('search.title')}
        </Text>
        <TextInput
          style={styles.input}
          placeholder={t('search.placeholder')}
          placeholderTextColor={COLORS.textSecondary}
          value={searchQuery}
          onChangeText={handleSearchChange}
          autoCapitalize="none"
          autoCorrect={false}
          returnKeyType="search"
          accessibilityLabel={t('search.placeholder')}
          accessibilityHint="Enter song or artist name to search"
        />
      </View>

      <View style={styles.content}>
        {isLoading && <LoadingSpinner message="Searching tracks..." />}
        {error && (
          <ErrorMessage
            message={error.message || t('errors.networkError')}
            onRetry={handleRetry}
            retryLabel={t('common.retry')}
          />
        )}
        {showRecentSearches && (
          <View style={styles.section}>
            <Text style={styles.sectionTitle} accessibilityRole="header">
              {t('search.recentSearches')}
            </Text>
            <FlatList
              data={recentSearches}
              keyExtractor={keyExtractor}
              renderItem={renderTrack}
              contentContainerStyle={styles.listContent}
              keyboardShouldPersistTaps="handled"
              scrollEnabled={false}
              removeClippedSubviews={true}
            />
          </View>
        )}
        {data && data.results.length === 0 && debouncedQuery.length > 0 && (
          <EmptyState
            icon="🔍"
            title={t('search.noResults')}
            message={t('search.noResultsMessage')}
          />
        )}
        {hasResults && (
          <FlatList
            data={data.results}
            keyExtractor={keyExtractor}
            renderItem={renderTrack}
            contentContainerStyle={styles.listContent}
            keyboardShouldPersistTaps="handled"
            removeClippedSubviews={true}
            maxToRenderPerBatch={10}
            windowSize={10}
            initialNumToRender={10}
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
    backgroundColor: COLORS.surface,
    borderBottomWidth: 1,
    borderBottomColor: COLORS.surfaceLight,
  },
  title: {
    ...TYPOGRAPHY.h1,
    color: COLORS.text,
    marginBottom: SPACING.md,
  },
  input: {
    ...TYPOGRAPHY.body,
    backgroundColor: COLORS.surfaceLight,
    color: COLORS.text,
    padding: SPACING.md,
    borderRadius: BORDER_RADIUS.md,
    borderWidth: 1,
    borderColor: COLORS.surfaceLight,
  },
  content: {
    flex: 1,
    padding: SPACING.md,
  },
  listContent: {
    paddingBottom: SPACING.lg,
  },
  section: {
    marginBottom: SPACING.lg,
  },
  sectionTitle: {
    ...TYPOGRAPHY.h3,
    color: COLORS.text,
    marginBottom: SPACING.md,
  },
});

