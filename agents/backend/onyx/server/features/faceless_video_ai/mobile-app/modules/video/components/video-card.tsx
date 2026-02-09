import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet, Image } from 'react-native';
import { useRouter } from 'expo-router';
import { useTheme } from '@/contexts/theme-context';
import { formatDate, formatDuration } from '@/utils/format';
import { VIDEO_STATUS_COLORS, VIDEO_STATUS_LABELS } from '@/utils/constants';
import type { VideoGenerationResponse } from '@/types/api';

interface VideoCardProps {
  video: VideoGenerationResponse;
  onPress?: () => void;
  onLongPress?: () => void;
}

export function VideoCard({ video, onPress, onLongPress }: VideoCardProps) {
  const router = useRouter();
  const { colors } = useTheme();

  const handlePress = () => {
    if (onPress) {
      onPress();
    } else {
      router.push(`/video-detail?videoId=${video.video_id}`);
    }
  };

  const statusColor = VIDEO_STATUS_COLORS[video.status] || colors.text;
  const statusLabel = VIDEO_STATUS_LABELS[video.status] || video.status;

  return (
    <TouchableOpacity
      style={[styles.card, { backgroundColor: colors.card }]}
      onPress={handlePress}
      onLongPress={onLongPress}
      activeOpacity={0.7}
    >
      {video.thumbnail_url && (
        <Image
          source={{ uri: video.thumbnail_url }}
          style={styles.thumbnail}
          resizeMode="cover"
        />
      )}

      <View style={styles.content}>
        <View style={styles.header}>
          <Text style={[styles.title, { color: colors.text }]} numberOfLines={1}>
            Video {video.video_id.substring(0, 8)}
          </Text>
          <View style={[styles.statusBadge, { backgroundColor: `${statusColor}20` }]}>
            <Text style={[styles.statusText, { color: statusColor }]}>
              {statusLabel}
            </Text>
          </View>
        </View>

        {video.progress && (
          <View style={styles.progressContainer}>
            <View style={[styles.progressBar, { backgroundColor: colors.border }]}>
              <View
                style={[
                  styles.progressFill,
                  { width: `${video.progress.progress}%`, backgroundColor: statusColor },
                ]}
              />
            </View>
            <Text style={[styles.progressText, { color: colors.textSecondary }]}>
              {video.progress.progress.toFixed(0)}%
            </Text>
          </View>
        )}

        <View style={styles.footer}>
          {video.duration && (
            <Text style={[styles.meta, { color: colors.textSecondary }]}>
              {formatDuration(video.duration)}
            </Text>
          )}
          {video.created_at && (
            <Text style={[styles.meta, { color: colors.textSecondary }]}>
              {formatDate(video.created_at, 'short')}
            </Text>
          )}
        </View>
      </View>
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  card: {
    borderRadius: 12,
    overflow: 'hidden',
    marginBottom: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  thumbnail: {
    width: '100%',
    height: 200,
    backgroundColor: '#E5E5EA',
  },
  content: {
    padding: 16,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  title: {
    fontSize: 16,
    fontWeight: '600',
    flex: 1,
  },
  statusBadge: {
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 12,
  },
  statusText: {
    fontSize: 12,
    fontWeight: '600',
    textTransform: 'capitalize',
  },
  progressContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
  },
  progressBar: {
    flex: 1,
    height: 4,
    borderRadius: 2,
    marginRight: 8,
    overflow: 'hidden',
  },
  progressFill: {
    height: '100%',
    borderRadius: 2,
  },
  progressText: {
    fontSize: 12,
    minWidth: 40,
  },
  footer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  meta: {
    fontSize: 12,
  },
});

