import React from 'react';
import { FlatList, FlatListProps, StyleSheet, View, Text } from 'react-native';
import { VideoCard } from './video-card';
import { Loading } from '@/components/ui/loading';
import { useTheme } from '@/contexts/theme-context';
import type { VideoGenerationResponse } from '@/types/api';

interface VideoListProps extends Omit<FlatListProps<VideoGenerationResponse>, 'data' | 'renderItem'> {
  videos: VideoGenerationResponse[];
  isLoading?: boolean;
  onVideoPress?: (video: VideoGenerationResponse) => void;
  onVideoLongPress?: (video: VideoGenerationResponse) => void;
  emptyMessage?: string;
}

export function VideoList({
  videos,
  isLoading = false,
  onVideoPress,
  onVideoLongPress,
  emptyMessage = 'No videos yet',
  ...props
}: VideoListProps) {
  const { colors } = useTheme();

  if (isLoading) {
    return <Loading message="Loading videos..." />;
  }

  if (videos.length === 0) {
    return (
      <View style={styles.emptyContainer}>
        <Text style={[styles.emptyText, { color: colors.textSecondary }]}>{emptyMessage}</Text>
      </View>
    );
  }

  return (
    <FlatList
      data={videos}
      renderItem={({ item }) => (
        <VideoCard
          video={item}
          onPress={() => onVideoPress?.(item)}
          onLongPress={() => onVideoLongPress?.(item)}
        />
      )}
      keyExtractor={(item) => item.video_id}
      contentContainerStyle={styles.listContent}
      {...props}
    />
  );
}

const styles = StyleSheet.create({
  listContent: {
    padding: 16,
  },
  emptyContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 32,
  },
  emptyText: {
    fontSize: 16,
    color: '#666666',
  },
});

