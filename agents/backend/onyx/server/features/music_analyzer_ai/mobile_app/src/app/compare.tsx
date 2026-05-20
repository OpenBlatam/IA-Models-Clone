import React, { useState, useCallback, useMemo } from 'react';
import {
  View,
  Text,
  StyleSheet,
  FlatList,
  TouchableOpacity,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useRouter } from 'expo-router';
import { useCompareTracks } from '../hooks/use-music-api';
import { useSearchTracks } from '../hooks/use-music-analysis';
import { TrackCard } from '../components/music/track-card';
import { LoadingSpinner } from '../components/common/loading-spinner';
import { ErrorMessage } from '../components/common/error-message';
import { EmptyState } from '../components/common/empty-state';
import { COLORS, SPACING, TYPOGRAPHY, BORDER_RADIUS } from '../constants/config';
import { useTranslation } from '../hooks/use-translation';
import { useMusic } from '../stores/music';
import { useToast } from '../contexts/toast-context';
import type { Track } from '../types/api';

export default function ComparePage() {
  const { t } = useTranslation();
  const router = useRouter();
  const { showToast } = useToast();
  const { favorites, recentSearches } = useMusic();
  const [selectedTracks, setSelectedTracks] = useState<Track[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const { data: searchResults } = useSearchTracks(searchQuery, 10);
  const compareMutation = useCompareTracks();

  const availableTracks = useMemo(
    () => [...favorites, ...recentSearches],
    [favorites, recentSearches]
  );

  const handleTrackSelect = useCallback(
    (track: Track) => {
      if (selectedTracks.some((t) => t.id === track.id)) {
        setSelectedTracks((prev) => prev.filter((t) => t.id !== track.id));
        showToast('Track removed from comparison', 'info');
      } else if (selectedTracks.length < 5) {
        setSelectedTracks((prev) => [...prev, track]);
        showToast('Track added to comparison', 'success');
      } else {
        showToast('Maximum 5 tracks can be compared', 'warning');
      }
    },
    [selectedTracks, showToast]
  );

  const handleCompare = useCallback(() => {
    if (selectedTracks.length < 2) {
      showToast('Please select at least 2 tracks to compare', 'warning');
      return;
    }

    compareMutation.mutate(
      { track_ids: selectedTracks.map((t) => t.id) },
      {
        onSuccess: (data) => {
          showToast('Comparison completed!', 'success');
          router.push({
            pathname: '/comparison-results',
            params: {
              data: JSON.stringify(data),
            },
          });
        },
        onError: (error) => {
          showToast(error.message || 'Failed to compare tracks', 'error');
        },
      }
    );
  }, [selectedTracks, compareMutation, router, showToast]);

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

  const isSelected = useCallback(
    (trackId: string) => selectedTracks.some((t) => t.id === trackId),
    [selectedTracks]
  );

  return (
    <SafeAreaView style={styles.container} edges={['top', 'bottom']}>
      <View style={styles.header}>
        <Text style={styles.title} accessibilityRole="header">
          Compare Tracks
        </Text>
        <Text style={styles.subtitle}>
          Select 2-5 tracks to compare ({selectedTracks.length}/5)
        </Text>
      </View>

      {selectedTracks.length >= 2 && (
        <TouchableOpacity
          style={styles.compareButton}
          onPress={handleCompare}
          disabled={compareMutation.isPending}
          accessibilityRole="button"
        >
          <Text style={styles.compareButtonText}>
            {compareMutation.isPending ? 'Comparing...' : 'Compare'}
          </Text>
        </TouchableOpacity>
      )}

      {compareMutation.isError && (
        <ErrorMessage
          message={
            compareMutation.error?.message || 'Failed to compare tracks'
          }
          onRetry={() => compareMutation.reset()}
        />
      )}

      <View style={styles.content}>
        <Text style={styles.sectionTitle}>Selected Tracks</Text>
        {selectedTracks.length === 0 ? (
          <EmptyState
            icon="🎵"
            title="No tracks selected"
            message="Select tracks from below to compare"
          />
        ) : (
          <FlatList
            horizontal
            data={selectedTracks}
            keyExtractor={(item) => item.id}
            renderItem={({ item }) => (
              <View style={styles.selectedTrack}>
                <TrackCard
                  track={item}
                  onPress={handleTrackPress}
                  showPreview={false}
                />
                <TouchableOpacity
                  style={styles.removeButton}
                  onPress={() => handleTrackSelect(item)}
                >
                  <Text style={styles.removeButtonText}>✕</Text>
                </TouchableOpacity>
              </View>
            )}
            contentContainerStyle={styles.selectedList}
          />
        )}

        <Text style={styles.sectionTitle}>Available Tracks</Text>
        {availableTracks.length === 0 ? (
          <EmptyState
            icon="🔍"
            title="No tracks available"
            message="Search for tracks first to compare them"
          />
        ) : (
          <FlatList
            data={availableTracks}
            keyExtractor={(item) => item.id}
            renderItem={({ item }) => (
              <View
                style={[
                  styles.trackWrapper,
                  isSelected(item.id) && styles.trackSelected,
                ]}
              >
                <TrackCard
                  track={item}
                  onPress={() => handleTrackSelect(item)}
                  showPreview={false}
                />
                {isSelected(item.id) && (
                  <View style={styles.checkmark}>
                    <Text style={styles.checkmarkText}>✓</Text>
                  </View>
                )}
              </View>
            )}
            contentContainerStyle={styles.listContent}
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
    marginBottom: SPACING.xs,
  },
  subtitle: {
    ...TYPOGRAPHY.bodySmall,
    color: COLORS.textSecondary,
  },
  compareButton: {
    backgroundColor: COLORS.primary,
    padding: SPACING.md,
    margin: SPACING.md,
    borderRadius: BORDER_RADIUS.md,
    alignItems: 'center',
  },
  compareButtonText: {
    ...TYPOGRAPHY.h3,
    color: COLORS.text,
    fontWeight: '600',
  },
  content: {
    flex: 1,
    padding: SPACING.md,
  },
  sectionTitle: {
    ...TYPOGRAPHY.h3,
    color: COLORS.text,
    marginBottom: SPACING.md,
    marginTop: SPACING.md,
  },
  selectedList: {
    paddingBottom: SPACING.md,
  },
  selectedTrack: {
    marginRight: SPACING.md,
    width: 200,
  },
  removeButton: {
    position: 'absolute',
    top: SPACING.xs,
    right: SPACING.xs,
    backgroundColor: COLORS.error,
    borderRadius: BORDER_RADIUS.full,
    width: 24,
    height: 24,
    justifyContent: 'center',
    alignItems: 'center',
  },
  removeButtonText: {
    color: COLORS.text,
    fontSize: 16,
    fontWeight: '600',
  },
  listContent: {
    paddingBottom: SPACING.lg,
  },
  trackWrapper: {
    position: 'relative',
    marginBottom: SPACING.sm,
  },
  trackSelected: {
    opacity: 0.7,
  },
  checkmark: {
    position: 'absolute',
    top: SPACING.sm,
    right: SPACING.sm,
    backgroundColor: COLORS.success,
    borderRadius: BORDER_RADIUS.full,
    width: 32,
    height: 32,
    justifyContent: 'center',
    alignItems: 'center',
  },
  checkmarkText: {
    color: COLORS.text,
    fontSize: 20,
    fontWeight: '600',
  },
});

