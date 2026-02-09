import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Dimensions,
  ActivityIndicator,
  Alert,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { Ionicons } from '@expo/vector-icons';
import { GestureDetector, Gesture } from 'react-native-gesture-handler';
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  withSpring,
  runOnJS,
} from 'react-native-reanimated';
import { useAuthStore } from '@/store/authStore';
import { apiService } from '@/services/api';
import type { Job } from '@/types';

const { width: SCREEN_WIDTH } = Dimensions.get('window');
const CARD_WIDTH = SCREEN_WIDTH - 40;
const SWIPE_THRESHOLD = 100;

export default function JobsScreen() {
  const { user } = useAuthStore();
  const queryClient = useQueryClient();
  const [currentIndex, setCurrentIndex] = useState(0);
  const [searchParams, setSearchParams] = useState({
    keywords: '',
    location: '',
  });

  const { data, isLoading, refetch } = useQuery<{ jobs: Job[]; total: number }>({
    queryKey: ['jobs', user?.id, searchParams],
    queryFn: async () => {
      if (!user?.id) throw new Error('No user ID');
      const response = await apiService.searchJobs(user.id, {
        keywords: searchParams.keywords || undefined,
        location: searchParams.location || undefined,
        limit: 20,
      });
      if (response.data) {
        return response.data;
      }
      throw new Error(response.error || 'Failed to load jobs');
    },
    enabled: !!user?.id,
  });

  const swipeMutation = useMutation({
    mutationFn: async ({ jobId, action }: { jobId: string; action: 'like' | 'dislike' | 'save' }) => {
      if (!user?.id) throw new Error('No user ID');
      return await apiService.swipeJob(user.id, { job_id: jobId, action });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['jobs', user?.id] });
      setCurrentIndex((prev) => prev + 1);
    },
  });

  const applyMutation = useMutation({
    mutationFn: async (jobId: string) => {
      if (!user?.id) throw new Error('No user ID');
      return await apiService.applyToJob(user.id, jobId);
    },
    onSuccess: () => {
      Alert.alert('Success', 'Application submitted successfully!');
      queryClient.invalidateQueries({ queryKey: ['jobs', user?.id] });
    },
  });

  const jobs = data?.jobs || [];
  const currentJob = jobs[currentIndex];

  const handleSwipe = (action: 'like' | 'dislike' | 'save') => {
    if (!currentJob) return;
    swipeMutation.mutate({ jobId: currentJob.id, action });
  };

  const handleApply = () => {
    if (!currentJob) return;
    Alert.alert(
      'Apply to Job',
      `Apply to ${currentJob.title} at ${currentJob.company}?`,
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Apply',
          onPress: () => applyMutation.mutate(currentJob.id),
        },
      ]
    );
  };

  if (isLoading && jobs.length === 0) {
    return (
      <SafeAreaView style={styles.container} edges={['top']}>
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color="#007AFF" />
          <Text style={styles.loadingText}>Loading jobs...</Text>
        </View>
      </SafeAreaView>
    );
  }

  if (jobs.length === 0 || currentIndex >= jobs.length) {
    return (
      <SafeAreaView style={styles.container} edges={['top']}>
        <View style={styles.emptyContainer}>
          <Ionicons name="briefcase-outline" size={64} color="#ccc" />
          <Text style={styles.emptyText}>No more jobs available</Text>
          <TouchableOpacity style={styles.refreshButton} onPress={() => refetch()}>
            <Text style={styles.refreshButtonText}>Refresh</Text>
          </TouchableOpacity>
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container} edges={['top']}>
      <View style={styles.header}>
        <Text style={styles.headerTitle}>Find Your Dream Job</Text>
        <Text style={styles.headerSubtitle}>
          {currentIndex + 1} of {jobs.length}
        </Text>
      </View>

      <View style={styles.cardContainer}>
        <JobCard job={currentJob} onSwipe={handleSwipe} />
      </View>

      <View style={styles.actionsContainer}>
        <TouchableOpacity
          style={[styles.actionButton, styles.dislikeButton]}
          onPress={() => handleSwipe('dislike')}
        >
          <Ionicons name="close" size={32} color="#fff" />
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.actionButton, styles.saveButton]}
          onPress={() => handleSwipe('save')}
        >
          <Ionicons name="bookmark" size={24} color="#fff" />
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.actionButton, styles.likeButton]}
          onPress={() => handleSwipe('like')}
        >
          <Ionicons name="heart" size={32} color="#fff" />
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.actionButton, styles.applyButton]}
          onPress={handleApply}
        >
          <Ionicons name="send" size={24} color="#fff" />
        </TouchableOpacity>
      </View>
    </SafeAreaView>
  );
}

interface JobCardProps {
  job: Job;
  onSwipe: (action: 'like' | 'dislike' | 'save') => void;
}

function JobCard({ job, onSwipe }: JobCardProps) {
  const translateX = useSharedValue(0);
  const translateY = useSharedValue(0);
  const scale = useSharedValue(1);

  const panGesture = Gesture.Pan()
    .onUpdate((e) => {
      translateX.value = e.translationX;
      translateY.value = e.translationY;
      scale.value = 1 - Math.abs(e.translationX) / 1000;
    })
    .onEnd((e) => {
      if (Math.abs(e.translationX) > SWIPE_THRESHOLD) {
        const action = e.translationX > 0 ? 'like' : 'dislike';
        runOnJS(onSwipe)(action);
        translateX.value = withSpring(e.translationX > 0 ? SCREEN_WIDTH : -SCREEN_WIDTH);
      } else {
        translateX.value = withSpring(0);
        translateY.value = withSpring(0);
        scale.value = withSpring(1);
      }
    });

  const animatedStyle = useAnimatedStyle(() => ({
    transform: [
      { translateX: translateX.value },
      { translateY: translateY.value },
      { scale: scale.value },
    ],
  }));

  return (
    <GestureDetector gesture={panGesture}>
      <Animated.View style={[styles.card, animatedStyle]}>
        <View style={styles.cardHeader}>
          <View style={styles.companyInfo}>
            <View style={styles.companyLogo}>
              <Text style={styles.companyInitial}>
                {job.company.charAt(0).toUpperCase()}
              </Text>
            </View>
            <View>
              <Text style={styles.companyName}>{job.company}</Text>
              <Text style={styles.jobLocation}>
                <Ionicons name="location" size={14} color="#666" /> {job.location}
              </Text>
            </View>
          </View>
          {job.match_score && (
            <View style={styles.matchBadge}>
              <Text style={styles.matchScore}>{job.match_score}%</Text>
              <Text style={styles.matchLabel}>Match</Text>
            </View>
          )}
        </View>

        <Text style={styles.jobTitle}>{job.title}</Text>

        {job.salary_range && (
          <View style={styles.salaryContainer}>
            <Ionicons name="cash" size={16} color="#4ECDC4" />
            <Text style={styles.salaryText}>{job.salary_range}</Text>
          </View>
        )}

        <Text style={styles.jobDescription} numberOfLines={4}>
          {job.description}
        </Text>

        {job.required_skills && job.required_skills.length > 0 && (
          <View style={styles.skillsContainer}>
            <Text style={styles.skillsTitle}>Required Skills:</Text>
            <View style={styles.skillsList}>
              {job.required_skills.slice(0, 5).map((skill, index) => (
                <View key={index} style={styles.skillTag}>
                  <Text style={styles.skillText}>{skill}</Text>
                </View>
              ))}
            </View>
          </View>
        )}

        {job.match_reasons && job.match_reasons.length > 0 && (
          <View style={styles.matchReasonsContainer}>
            <Text style={styles.matchReasonsTitle}>Why this matches:</Text>
            {job.match_reasons.slice(0, 3).map((reason, index) => (
              <View key={index} style={styles.matchReasonItem}>
                <Ionicons name="checkmark-circle" size={16} color="#4ECDC4" />
                <Text style={styles.matchReasonText}>{reason}</Text>
              </View>
            ))}
          </View>
        )}
      </Animated.View>
    </GestureDetector>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    marginTop: 16,
    fontSize: 16,
    color: '#666',
  },
  emptyContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  emptyText: {
    fontSize: 18,
    color: '#666',
    marginTop: 16,
    textAlign: 'center',
  },
  refreshButton: {
    marginTop: 24,
    backgroundColor: '#007AFF',
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 12,
  },
  refreshButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  header: {
    padding: 20,
    backgroundColor: '#fff',
    borderBottomWidth: 1,
    borderBottomColor: '#e0e0e0',
  },
  headerTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#333',
  },
  headerSubtitle: {
    fontSize: 14,
    color: '#666',
    marginTop: 4,
  },
  cardContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  card: {
    width: CARD_WIDTH,
    backgroundColor: '#fff',
    borderRadius: 20,
    padding: 24,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.2,
    shadowRadius: 12,
    elevation: 8,
  },
  cardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: 16,
  },
  companyInfo: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
  },
  companyLogo: {
    width: 50,
    height: 50,
    borderRadius: 25,
    backgroundColor: '#007AFF',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 12,
  },
  companyInitial: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#fff',
  },
  companyName: {
    fontSize: 18,
    fontWeight: '600',
    color: '#333',
  },
  jobLocation: {
    fontSize: 14,
    color: '#666',
    marginTop: 4,
  },
  matchBadge: {
    backgroundColor: '#4ECDC4',
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 12,
    alignItems: 'center',
  },
  matchScore: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#fff',
  },
  matchLabel: {
    fontSize: 10,
    color: '#fff',
  },
  jobTitle: {
    fontSize: 22,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 12,
  },
  salaryContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 16,
  },
  salaryText: {
    fontSize: 16,
    color: '#4ECDC4',
    fontWeight: '600',
    marginLeft: 8,
  },
  jobDescription: {
    fontSize: 14,
    color: '#666',
    lineHeight: 20,
    marginBottom: 16,
  },
  skillsContainer: {
    marginTop: 16,
    paddingTop: 16,
    borderTopWidth: 1,
    borderTopColor: '#e0e0e0',
  },
  skillsTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#333',
    marginBottom: 8,
  },
  skillsList: {
    flexDirection: 'row',
    flexWrap: 'wrap',
  },
  skillTag: {
    backgroundColor: '#f0f0f0',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 12,
    marginRight: 8,
    marginBottom: 8,
  },
  skillText: {
    fontSize: 12,
    color: '#333',
  },
  matchReasonsContainer: {
    marginTop: 16,
    paddingTop: 16,
    borderTopWidth: 1,
    borderTopColor: '#e0e0e0',
  },
  matchReasonsTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#333',
    marginBottom: 8,
  },
  matchReasonItem: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
  },
  matchReasonText: {
    fontSize: 12,
    color: '#666',
    marginLeft: 8,
    flex: 1,
  },
  actionsContainer: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    alignItems: 'center',
    padding: 20,
    backgroundColor: '#fff',
    borderTopWidth: 1,
    borderTopColor: '#e0e0e0',
  },
  actionButton: {
    width: 56,
    height: 56,
    borderRadius: 28,
    justifyContent: 'center',
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.2,
    shadowRadius: 4,
    elevation: 4,
  },
  dislikeButton: {
    backgroundColor: '#FF3B30',
  },
  saveButton: {
    backgroundColor: '#FFD700',
  },
  likeButton: {
    backgroundColor: '#4ECDC4',
  },
  applyButton: {
    backgroundColor: '#007AFF',
  },
});


