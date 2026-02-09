import { useState } from 'react';
import {
  View,
  Text,
  ScrollView,
  StyleSheet,
  TouchableOpacity,
  RefreshControl,
  Alert,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useRouter } from 'expo-router';
import { usePosts, usePublishPost, useCancelPost } from '@/hooks/useApi';
import { ActivityIndicator } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import type { Post } from '@/types';
import { format } from 'date-fns';
import { EmptyState } from '@/components/ui/EmptyState';

export default function PostsScreen() {
  const router = useRouter();
  const [statusFilter, setStatusFilter] = useState<string | undefined>(undefined);
  const { data: posts, isLoading, refetch } = usePosts(statusFilter);
  const publishPost = usePublishPost();
  const cancelPost = useCancelPost();

  const handlePublish = async (postId: string) => {
    Alert.alert('Publish Post', 'Are you sure you want to publish this post now?', [
      { text: 'Cancel', style: 'cancel' },
      {
        text: 'Publish',
        onPress: async () => {
          try {
            await publishPost.mutateAsync(postId);
            Alert.alert('Success', 'Post published successfully');
            refetch();
          } catch (error) {
            Alert.alert('Error', 'Failed to publish post');
          }
        },
      },
    ]);
  };

  const handleCancel = async (postId: string) => {
    Alert.alert('Cancel Post', 'Are you sure you want to cancel this post?', [
      { text: 'No', style: 'cancel' },
      {
        text: 'Yes',
        style: 'destructive',
        onPress: async () => {
          try {
            await cancelPost.mutateAsync(postId);
            Alert.alert('Success', 'Post cancelled');
            refetch();
          } catch (error) {
            Alert.alert('Error', 'Failed to cancel post');
          }
        },
      },
    ]);
  };

  return (
    <SafeAreaView style={styles.container} edges={['top']}>
      <View style={styles.header}>
        <TouchableOpacity
          style={styles.addButton}
          onPress={() => router.push('/posts/create')}
        >
          <Ionicons name="add" size={24} color="#fff" />
        </TouchableOpacity>
        <View style={styles.filters}>
          <FilterButton
            label="All"
            active={statusFilter === undefined}
            onPress={() => setStatusFilter(undefined)}
          />
          <FilterButton
            label="Scheduled"
            active={statusFilter === 'scheduled'}
            onPress={() => setStatusFilter('scheduled')}
          />
          <FilterButton
            label="Published"
            active={statusFilter === 'published'}
            onPress={() => setStatusFilter('published')}
          />
        </View>
      </View>

      <ScrollView
        style={styles.scrollView}
        refreshControl={<RefreshControl refreshing={isLoading} onRefresh={refetch} />}
      >
        {isLoading && !posts ? (
          <ActivityIndicator size="large" color="#0ea5e9" style={styles.loader} />
        ) : posts && posts.length > 0 ? (
          posts.map((post) => (
            <PostCard
              key={post.post_id}
              post={post}
              onPublish={() => handlePublish(post.post_id)}
              onCancel={() => handleCancel(post.post_id)}
            />
          ))
        ) : (
          <EmptyState
            icon="document-text-outline"
            title="No posts found"
            message="Create your first post to get started"
            actionLabel="Create Post"
            onAction={() => router.push('/posts/create')}
          />
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

function FilterButton({ label, active, onPress }: { label: string; active: boolean; onPress: () => void }) {
  return (
    <TouchableOpacity
      style={[styles.filterButton, active && styles.filterButtonActive]}
      onPress={onPress}
    >
      <Text style={[styles.filterText, active && styles.filterTextActive]}>{label}</Text>
    </TouchableOpacity>
  );
}

function PostCard({
  post,
  onPublish,
  onCancel,
}: {
  post: Post;
  onPublish: () => void;
  onCancel: () => void;
}) {
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'published':
        return '#10b981';
      case 'scheduled':
        return '#f59e0b';
      case 'cancelled':
        return '#ef4444';
      default:
        return '#6b7280';
    }
  };

  return (
    <View style={styles.card}>
      <View style={styles.cardHeader}>
        <View style={[styles.statusBadge, { backgroundColor: getStatusColor(post.status) }]}>
          <Text style={styles.statusText}>{post.status.toUpperCase()}</Text>
        </View>
        {post.scheduled_time && (
          <Text style={styles.date}>
            {format(new Date(post.scheduled_time), 'MMM dd, yyyy HH:mm')}
          </Text>
        )}
      </View>

      <Text style={styles.content} numberOfLines={3}>
        {post.content}
      </Text>

      <View style={styles.platforms}>
        {post.platforms.map((platform) => (
          <View key={platform} style={styles.platformTag}>
            <Text style={styles.platformText}>{platform}</Text>
          </View>
        ))}
      </View>

      {post.tags && post.tags.length > 0 && (
        <View style={styles.tags}>
          {post.tags.map((tag) => (
            <Text key={tag} style={styles.tag}>
              #{tag}
            </Text>
          ))}
        </View>
      )}

      {post.status === 'scheduled' && (
        <View style={styles.actions}>
          <TouchableOpacity style={styles.actionButton} onPress={onPublish}>
            <Ionicons name="send" size={16} color="#10b981" />
            <Text style={styles.actionText}>Publish Now</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.actionButton} onPress={onCancel}>
            <Ionicons name="close-circle" size={16} color="#ef4444" />
            <Text style={styles.actionText}>Cancel</Text>
          </TouchableOpacity>
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  header: {
    backgroundColor: '#fff',
    padding: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#e5e7eb',
  },
  addButton: {
    backgroundColor: '#0ea5e9',
    width: 48,
    height: 48,
    borderRadius: 24,
    justifyContent: 'center',
    alignItems: 'center',
    alignSelf: 'flex-end',
    marginBottom: 12,
  },
  filters: {
    flexDirection: 'row',
    gap: 8,
  },
  filterButton: {
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 20,
    backgroundColor: '#f3f4f6',
  },
  filterButtonActive: {
    backgroundColor: '#0ea5e9',
  },
  filterText: {
    fontSize: 14,
    color: '#6b7280',
  },
  filterTextActive: {
    color: '#fff',
    fontWeight: '600',
  },
  scrollView: {
    flex: 1,
  },
  loader: {
    marginTop: 40,
  },
  card: {
    backgroundColor: '#fff',
    margin: 16,
    padding: 16,
    borderRadius: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  cardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  statusBadge: {
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 4,
  },
  statusText: {
    fontSize: 10,
    fontWeight: '600',
    color: '#fff',
  },
  date: {
    fontSize: 12,
    color: '#6b7280',
  },
  content: {
    fontSize: 14,
    color: '#1f2937',
    marginBottom: 12,
    lineHeight: 20,
  },
  platforms: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
    marginBottom: 8,
  },
  platformTag: {
    backgroundColor: '#e0f2fe',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 4,
  },
  platformText: {
    fontSize: 12,
    color: '#0369a1',
    fontWeight: '500',
  },
  tags: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
    marginTop: 8,
  },
  tag: {
    fontSize: 12,
    color: '#6b7280',
  },
  actions: {
    flexDirection: 'row',
    gap: 12,
    marginTop: 12,
    paddingTop: 12,
    borderTopWidth: 1,
    borderTopColor: '#f3f4f6',
  },
  actionButton: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 6,
    backgroundColor: '#f9fafb',
  },
  actionText: {
    fontSize: 12,
    fontWeight: '500',
  },
});

