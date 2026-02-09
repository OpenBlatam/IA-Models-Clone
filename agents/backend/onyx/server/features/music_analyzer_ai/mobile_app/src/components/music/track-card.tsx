import React, { memo, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Pressable,
} from 'react-native';
import Animated, {
  useAnimatedStyle,
  useSharedValue,
  withSpring,
  withTiming,
} from 'react-native-reanimated';
import { Image } from 'expo-image';
import { COLORS, SPACING, TYPOGRAPHY, BORDER_RADIUS } from '../../constants/config';
import type { Track } from '../../types/api';
import { formatArtists, formatDuration } from '../../utils/formatters';
import { useTranslation } from '../../hooks/use-translation';
import { useHapticFeedback } from '../../hooks/use-haptic-feedback';

interface TrackCardProps {
  track: Track;
  onPress: (track: Track) => void;
  showPreview?: boolean;
  isFavorite?: boolean;
  onToggleFavorite?: (track: Track) => void;
}

const AnimatedPressable = Animated.createAnimatedComponent(Pressable);

function TrackCardComponent({
  track,
  onPress,
  showPreview = false,
  isFavorite = false,
  onToggleFavorite,
}: TrackCardProps) {
  const { t } = useTranslation();
  const haptics = useHapticFeedback();
  const scale = useSharedValue(1);
  const opacity = useSharedValue(1);

  const handlePress = useCallback(() => {
    haptics.light();
    onPress(track);
  }, [onPress, track, haptics]);

  const handleLongPress = useCallback(() => {
    if (onToggleFavorite) {
      haptics.medium();
      onToggleFavorite(track);
    }
  }, [onToggleFavorite, track, haptics]);

  const handlePressIn = useCallback(() => {
    haptics.selection();
    scale.value = withSpring(0.97);
    opacity.value = withTiming(0.8);
  }, [scale, opacity, haptics]);

  const handlePressOut = useCallback(() => {
    scale.value = withSpring(1);
    opacity.value = withTiming(1);
  }, [scale, opacity]);

  const animatedStyle = useAnimatedStyle(() => ({
    transform: [{ scale: scale.value }],
    opacity: opacity.value,
  }));

  return (
    <AnimatedPressable
      style={[styles.container, animatedStyle]}
      onPress={handlePress}
      onLongPress={handleLongPress}
      onPressIn={handlePressIn}
      onPressOut={handlePressOut}
      accessibilityRole="button"
      accessibilityLabel={`${track.name} by ${formatArtists(track.artists)}`}
      accessibilityHint="Double tap to view analysis"
    >
      {track.preview_url && showPreview ? (
        <Image
          source={{ uri: track.preview_url }}
          style={styles.image}
          contentFit="cover"
          transition={200}
          placeholder={{ blurhash: 'LGF5]+Yk^6#M@-5c,1J5@[or[Q6.' }}
        />
      ) : (
        <View style={styles.placeholder} accessibilityLabel="Track placeholder">
          <Text style={styles.placeholderText}>🎵</Text>
        </View>
      )}
      <View style={styles.content}>
        <Text style={styles.title} numberOfLines={1}>
          {track.name}
        </Text>
        <Text style={styles.artists} numberOfLines={1}>
          {formatArtists(track.artists)}
        </Text>
        {track.album && (
          <Text style={styles.album} numberOfLines={1}>
            {track.album}
          </Text>
        )}
        <View style={styles.footer}>
          <Text style={styles.duration} accessibilityLabel={t('track.duration')}>
            {formatDuration(track.duration_ms)}
          </Text>
          {track.popularity > 0 && (
            <Text
              style={styles.popularity}
              accessibilityLabel={`${t('track.popularity')}: ${track.popularity}`}
            >
              ⭐ {track.popularity}
            </Text>
          )}
          {onToggleFavorite && (
            <TouchableOpacity
              onPress={handleLongPress}
              style={styles.favoriteButton}
              accessibilityRole="button"
              accessibilityLabel={
                isFavorite ? 'Remove from favorites' : 'Add to favorites'
              }
            >
              <Text style={styles.favoriteIcon}>
                {isFavorite ? '❤️' : '🤍'}
              </Text>
            </TouchableOpacity>
          )}
        </View>
      </View>
    </AnimatedPressable>
  );
}

export const TrackCard = memo(TrackCardComponent);

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    backgroundColor: COLORS.surface,
    borderRadius: BORDER_RADIUS.md,
    padding: SPACING.md,
    marginBottom: SPACING.sm,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  image: {
    width: 60,
    height: 60,
    borderRadius: BORDER_RADIUS.sm,
    marginRight: SPACING.md,
  },
  placeholder: {
    width: 60,
    height: 60,
    borderRadius: BORDER_RADIUS.sm,
    backgroundColor: COLORS.surfaceLight,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: SPACING.md,
  },
  placeholderText: {
    fontSize: 24,
  },
  content: {
    flex: 1,
    justifyContent: 'space-between',
  },
  title: {
    ...TYPOGRAPHY.h3,
    color: COLORS.text,
    marginBottom: SPACING.xs,
  },
  artists: {
    ...TYPOGRAPHY.bodySmall,
    color: COLORS.textSecondary,
    marginBottom: SPACING.xs,
  },
  album: {
    ...TYPOGRAPHY.caption,
    color: COLORS.textSecondary,
    marginBottom: SPACING.xs,
  },
  footer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginTop: SPACING.xs,
  },
  duration: {
    ...TYPOGRAPHY.caption,
    color: COLORS.textSecondary,
  },
  popularity: {
    ...TYPOGRAPHY.caption,
    color: COLORS.warning,
  },
  favoriteButton: {
    marginLeft: SPACING.sm,
    padding: SPACING.xs,
  },
  favoriteIcon: {
    fontSize: 20,
  },
});

