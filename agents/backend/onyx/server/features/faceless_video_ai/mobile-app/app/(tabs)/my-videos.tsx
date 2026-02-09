import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  FlatList,
  TouchableOpacity,
  RefreshControl,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useRouter } from 'expo-router';
import { useVideoStatus } from '@/hooks/use-video-generation';
import { VideoStatus } from '@/types/api';
import { format } from 'date-fns';

// This is a placeholder - in a real app, you'd fetch a list of videos
const MOCK_VIDEOS: string[] = [];

export default function MyVideosScreen() {
  const router = useRouter();
  const [refreshing, setRefreshing] = React.useState(false);

  const onRefresh = React.useCallback(() => {
    setRefreshing(true);
    // In a real app, refetch videos list
    setTimeout(() => setRefreshing(false), 1000);
  }, []);

  const renderVideoItem = ({ item: videoId }: { item: string }) => {
    const { data: video } = useVideoStatus(videoId, true);

    if (!video) {
      return null;
    }

    const getStatusColor = (status: VideoStatus) => {
      switch (status) {
        case VideoStatus.COMPLETED:
          return '#34C759';
        case VideoStatus.FAILED:
          return '#FF3B30';
        case VideoStatus.PROCESSING:
        case VideoStatus.GENERATING_IMAGES:
        case VideoStatus.GENERATING_AUDIO:
        case VideoStatus.ADDING_SUBTITLES:
        case VideoStatus.COMPOSITING:
          return '#007AFF';
        default:
          return '#FF9500';
      }
    };

    return (
      <TouchableOpacity
        style={styles.videoCard}
        onPress={() => router.push(`/video-detail?videoId=${videoId}`)}
      >
        <View style={styles.videoHeader}>
          <Text style={styles.videoId} numberOfLines={1}>
            {videoId.substring(0, 8)}...
          </Text>
          <View
            style={[
              styles.statusBadge,
              { backgroundColor: getStatusColor(video.status) + '20' },
            ]}
          >
            <Text
              style={[styles.statusText, { color: getStatusColor(video.status) }]}
            >
              {video.status}
            </Text>
          </View>
        </View>

        {video.progress && (
          <View style={styles.progressContainer}>
            <View style={styles.progressBar}>
              <View
                style={[
                  styles.progressFill,
                  { width: `${video.progress.progress}%` },
                ]}
              />
            </View>
            <Text style={styles.progressText}>
              {video.progress.progress.toFixed(0)}%
            </Text>
          </View>
        )}

        {video.duration && (
          <Text style={styles.duration}>Duration: {video.duration.toFixed(2)}s</Text>
        )}

        {video.created_at && (
          <Text style={styles.date}>
            {format(new Date(video.created_at), 'MMM dd, yyyy HH:mm')}
          </Text>
        )}
      </TouchableOpacity>
    );
  };

  return (
    <SafeAreaView style={styles.container} edges={['top']}>
      <View style={styles.header}>
        <Text style={styles.title}>My Videos</Text>
        <TouchableOpacity onPress={() => router.push('/video-generation')}>
          <Text style={styles.createButton}>+ Create</Text>
        </TouchableOpacity>
      </View>

      {MOCK_VIDEOS.length === 0 ? (
        <View style={styles.emptyContainer}>
          <Text style={styles.emptyText}>No videos yet</Text>
          <Text style={styles.emptySubtext}>
            Create your first video to get started
          </Text>
          <TouchableOpacity
            style={styles.emptyButton}
            onPress={() => router.push('/video-generation')}
          >
            <Text style={styles.emptyButtonText}>Create Video</Text>
          </TouchableOpacity>
        </View>
      ) : (
        <FlatList
          data={MOCK_VIDEOS}
          renderItem={renderVideoItem}
          keyExtractor={(item) => item}
          contentContainerStyle={styles.listContent}
          refreshControl={
            <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
          }
        />
      )}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#FFFFFF',
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 20,
    borderBottomWidth: 1,
    borderBottomColor: '#E5E5EA',
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#000000',
  },
  createButton: {
    fontSize: 16,
    color: '#007AFF',
    fontWeight: '600',
  },
  listContent: {
    padding: 20,
  },
  videoCard: {
    backgroundColor: '#F5F5F5',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
  },
  videoHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  videoId: {
    fontSize: 16,
    fontWeight: '600',
    color: '#000000',
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
    backgroundColor: '#E5E5EA',
    borderRadius: 2,
    marginRight: 8,
    overflow: 'hidden',
  },
  progressFill: {
    height: '100%',
    backgroundColor: '#007AFF',
    borderRadius: 2,
  },
  progressText: {
    fontSize: 12,
    color: '#666666',
    minWidth: 40,
  },
  duration: {
    fontSize: 14,
    color: '#666666',
    marginBottom: 4,
  },
  date: {
    fontSize: 12,
    color: '#8E8E93',
  },
  emptyContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  emptyText: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#000000',
    marginBottom: 8,
  },
  emptySubtext: {
    fontSize: 16,
    color: '#666666',
    textAlign: 'center',
    marginBottom: 24,
  },
  emptyButton: {
    backgroundColor: '#007AFF',
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 8,
  },
  emptyButtonText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '600',
  },
});


