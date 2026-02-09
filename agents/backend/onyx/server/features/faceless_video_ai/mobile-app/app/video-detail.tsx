import React, { useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  ActivityIndicator,
  Alert,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useLocalSearchParams, useRouter } from 'expo-router';
import { useVideoStatus, useDownloadVideo, useDeleteVideo } from '@/hooks/use-video-generation';
import { Button } from '@/components/ui/button';
import { VideoStatus } from '@/types/api';
import * as FileSystem from 'expo-file-system';
import * as Sharing from 'expo-sharing';

export default function VideoDetailScreen() {
  const { videoId } = useLocalSearchParams<{ videoId: string }>();
  const router = useRouter();
  const { data: video, isLoading, error } = useVideoStatus(videoId || '', !!videoId);
  const downloadVideo = useDownloadVideo();
  const deleteVideo = useDeleteVideo();

  const handleDownload = async () => {
    if (!videoId) return;

    try {
      const result = await downloadVideo.mutateAsync({
        videoId,
        onProgress: (progress) => {
          console.log(`Download progress: ${progress}%`);
        },
      });

      if (await Sharing.isAvailableAsync()) {
        await Sharing.shareAsync(result.uri);
      } else {
        Alert.alert('Success', 'Video downloaded successfully');
      }
    } catch (error: any) {
      Alert.alert('Error', error.detail || 'Failed to download video');
    }
  };

  const handleDelete = async () => {
    if (!videoId) return;

    Alert.alert(
      'Delete Video',
      'Are you sure you want to delete this video?',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Delete',
          style: 'destructive',
          onPress: async () => {
            try {
              await deleteVideo.mutateAsync(videoId);
              router.back();
            } catch (error: any) {
              Alert.alert('Error', error.detail || 'Failed to delete video');
            }
          },
        },
      ]
    );
  };

  if (isLoading) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color="#007AFF" />
          <Text style={styles.loadingText}>Loading video details...</Text>
        </View>
      </SafeAreaView>
    );
  }

  if (error || !video) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.errorContainer}>
          <Text style={styles.errorText}>
            {error ? 'Failed to load video' : 'Video not found'}
          </Text>
          <Button title="Go Back" onPress={() => router.back()} variant="primary" />
        </View>
      </SafeAreaView>
    );
  }

  const isProcessing =
    video.status === VideoStatus.PENDING ||
    video.status === VideoStatus.PROCESSING ||
    video.status === VideoStatus.GENERATING_IMAGES ||
    video.status === VideoStatus.GENERATING_AUDIO ||
    video.status === VideoStatus.ADDING_SUBTITLES ||
    video.status === VideoStatus.COMPOSITING;

  return (
    <SafeAreaView style={styles.container} edges={['top', 'bottom']}>
      <ScrollView style={styles.scrollView} contentContainerStyle={styles.content}>
        <Text style={styles.title}>Video Details</Text>

        <View style={styles.statusContainer}>
          <Text style={styles.statusLabel}>Status:</Text>
          <Text style={[styles.statusValue, styles[`status_${video.status}`]]}>
            {video.status}
          </Text>
        </View>

        {isProcessing && (
          <View style={styles.progressContainer}>
            <View style={styles.progressBar}>
              <View
                style={[styles.progressFill, { width: `${video.progress.progress}%` }]}
              />
            </View>
            <Text style={styles.progressText}>
              {video.progress.progress.toFixed(1)}% - {video.progress.current_step}
            </Text>
            {video.progress.estimated_time_remaining && (
              <Text style={styles.timeRemaining}>
                Estimated time remaining: {Math.round(video.progress.estimated_time_remaining)}s
              </Text>
            )}
          </View>
        )}

        {video.status === VideoStatus.COMPLETED && (
          <View style={styles.infoContainer}>
            {video.duration && (
              <View style={styles.infoRow}>
                <Text style={styles.infoLabel}>Duration:</Text>
                <Text style={styles.infoValue}>{video.duration.toFixed(2)}s</Text>
              </View>
            )}
            {video.file_size && (
              <View style={styles.infoRow}>
                <Text style={styles.infoLabel}>File Size:</Text>
                <Text style={styles.infoValue}>
                  {(video.file_size / 1024 / 1024).toFixed(2)} MB
                </Text>
              </View>
            )}
            {video.created_at && (
              <View style={styles.infoRow}>
                <Text style={styles.infoLabel}>Created:</Text>
                <Text style={styles.infoValue}>
                  {new Date(video.created_at).toLocaleString()}
                </Text>
              </View>
            )}
          </View>
        )}

        {video.status === VideoStatus.FAILED && video.error && (
          <View style={styles.errorContainer}>
            <Text style={styles.errorTitle}>Generation Failed</Text>
            <Text style={styles.errorMessage}>{video.error}</Text>
          </View>
        )}

        <View style={styles.actionsContainer}>
          {video.status === VideoStatus.COMPLETED && (
            <Button
              title="Download Video"
              onPress={handleDownload}
              variant="primary"
              size="large"
              style={styles.actionButton}
              loading={downloadVideo.isPending}
            />
          )}
          <Button
            title="Delete Video"
            onPress={handleDelete}
            variant="danger"
            size="large"
            style={styles.actionButton}
            loading={deleteVideo.isPending}
          />
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#FFFFFF',
  },
  scrollView: {
    flex: 1,
  },
  content: {
    padding: 20,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    marginTop: 16,
    fontSize: 16,
    color: '#666666',
  },
  errorContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  errorText: {
    fontSize: 16,
    color: '#FF3B30',
    marginBottom: 16,
    textAlign: 'center',
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    marginBottom: 24,
    color: '#000000',
  },
  statusContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 24,
  },
  statusLabel: {
    fontSize: 16,
    fontWeight: '600',
    marginRight: 8,
    color: '#000000',
  },
  statusValue: {
    fontSize: 16,
    fontWeight: '600',
    textTransform: 'capitalize',
  },
  status_pending: { color: '#FF9500' },
  status_processing: { color: '#007AFF' },
  status_generating_images: { color: '#007AFF' },
  status_generating_audio: { color: '#007AFF' },
  status_adding_subtitles: { color: '#007AFF' },
  status_compositing: { color: '#007AFF' },
  status_completed: { color: '#34C759' },
  status_failed: { color: '#FF3B30' },
  progressContainer: {
    marginBottom: 24,
  },
  progressBar: {
    height: 8,
    backgroundColor: '#E5E5EA',
    borderRadius: 4,
    overflow: 'hidden',
    marginBottom: 8,
  },
  progressFill: {
    height: '100%',
    backgroundColor: '#007AFF',
    borderRadius: 4,
  },
  progressText: {
    fontSize: 14,
    color: '#666666',
    marginBottom: 4,
  },
  timeRemaining: {
    fontSize: 12,
    color: '#8E8E93',
  },
  infoContainer: {
    backgroundColor: '#F5F5F5',
    borderRadius: 12,
    padding: 16,
    marginBottom: 24,
  },
  infoRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 12,
  },
  infoLabel: {
    fontSize: 14,
    color: '#666666',
  },
  infoValue: {
    fontSize: 14,
    fontWeight: '600',
    color: '#000000',
  },
  errorTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#FF3B30',
    marginBottom: 8,
  },
  errorMessage: {
    fontSize: 14,
    color: '#666666',
  },
  actionsContainer: {
    marginTop: 24,
  },
  actionButton: {
    marginBottom: 12,
  },
});


