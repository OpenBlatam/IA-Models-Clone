import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  ActivityIndicator,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useQuery } from '@tanstack/react-query';
import { Ionicons } from '@expo/vector-icons';
import { useAuthStore } from '@/store/authStore';
import { apiService } from '@/services/api';
import type { Roadmap, Step } from '@/types';

export default function RoadmapScreen() {
  const { user } = useAuthStore();

  const { data, isLoading } = useQuery<Roadmap>({
    queryKey: ['roadmap', user?.id],
    queryFn: async () => {
      if (!user?.id) throw new Error('No user ID');
      const response = await apiService.getRoadmap(user.id);
      if (response.data) {
        return response.data;
      }
      throw new Error(response.error || 'Failed to load roadmap');
    },
    enabled: !!user?.id,
  });

  const handleStepPress = async (step: Step) => {
    if (!user?.id) return;

    if (step.status === 'not_started') {
      await apiService.startStep(user.id, step.id);
    }
    // Navigate to step details (would be implemented with navigation)
  };

  if (isLoading) {
    return (
      <SafeAreaView style={styles.container} edges={['top']}>
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color="#007AFF" />
        </View>
      </SafeAreaView>
    );
  }

  const steps = data?.steps || [];
  const progress = data?.progress_percentage || 0;

  return (
    <SafeAreaView style={styles.container} edges={['top']}>
      <View style={styles.header}>
        <Text style={styles.title}>Your Career Roadmap</Text>
        <Text style={styles.subtitle}>
          {data?.completed_steps || 0} of {data?.total_steps || 0} steps completed
        </Text>
      </View>

      {/* Progress Bar */}
      <View style={styles.progressContainer}>
        <View style={styles.progressBar}>
          <View style={[styles.progressFill, { width: `${progress}%` }]} />
        </View>
        <Text style={styles.progressText}>{progress.toFixed(0)}% Complete</Text>
      </View>

      <ScrollView style={styles.scrollView} contentContainerStyle={styles.scrollContent}>
        {steps.map((step, index) => (
          <StepCard
            key={step.id}
            step={step}
            index={index}
            onPress={() => handleStepPress(step)}
          />
        ))}
      </ScrollView>
    </SafeAreaView>
  );
}

interface StepCardProps {
  step: Step;
  index: number;
  onPress: () => void;
}

function StepCard({ step, index, onPress }: StepCardProps) {
  const getStatusIcon = () => {
    switch (step.status) {
      case 'completed':
        return <Ionicons name="checkmark-circle" size={24} color="#4ECDC4" />;
      case 'in_progress':
        return <Ionicons name="time" size={24} color="#FFD700" />;
      default:
        return <Ionicons name="ellipse-outline" size={24} color="#ccc" />;
    }
  };

  const getStatusColor = () => {
    switch (step.status) {
      case 'completed':
        return '#4ECDC4';
      case 'in_progress':
        return '#FFD700';
      default:
        return '#e0e0e0';
    }
  };

  return (
    <TouchableOpacity
      style={[styles.stepCard, { borderLeftColor: getStatusColor() }]}
      onPress={onPress}
      activeOpacity={0.7}
    >
      <View style={styles.stepHeader}>
        <View style={styles.stepNumber}>
          <Text style={styles.stepNumberText}>{index + 1}</Text>
        </View>
        <View style={styles.stepInfo}>
          <Text style={styles.stepTitle}>{step.title}</Text>
          <Text style={styles.stepCategory}>{step.category}</Text>
        </View>
        {getStatusIcon()}
      </View>

      <Text style={styles.stepDescription}>{step.description}</Text>

      {step.resources && step.resources.length > 0 && (
        <View style={styles.resourcesContainer}>
          <Text style={styles.resourcesTitle}>Resources:</Text>
          {step.resources.slice(0, 3).map((resource, idx) => (
            <View key={idx} style={styles.resourceItem}>
              <Ionicons
                name={
                  resource.type === 'video'
                    ? 'videocam'
                    : resource.type === 'article'
                    ? 'document-text'
                    : 'link'
                }
                size={16}
                color="#007AFF"
              />
              <Text style={styles.resourceText}>{resource.title}</Text>
            </View>
          ))}
        </View>
      )}

      {step.status === 'completed' && step.completed_at && (
        <Text style={styles.completedDate}>
          Completed on {new Date(step.completed_at).toLocaleDateString()}
        </Text>
      )}
    </TouchableOpacity>
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
  header: {
    backgroundColor: '#fff',
    padding: 20,
    borderBottomWidth: 1,
    borderBottomColor: '#e0e0e0',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#333',
  },
  subtitle: {
    fontSize: 14,
    color: '#666',
    marginTop: 4,
  },
  progressContainer: {
    backgroundColor: '#fff',
    padding: 20,
    borderBottomWidth: 1,
    borderBottomColor: '#e0e0e0',
  },
  progressBar: {
    height: 8,
    backgroundColor: '#e0e0e0',
    borderRadius: 4,
    overflow: 'hidden',
    marginBottom: 8,
  },
  progressFill: {
    height: '100%',
    backgroundColor: '#4ECDC4',
    borderRadius: 4,
  },
  progressText: {
    fontSize: 12,
    color: '#666',
    textAlign: 'right',
  },
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    padding: 16,
  },
  stepCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
    borderLeftWidth: 4,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 2,
  },
  stepHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },
  stepNumber: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: '#f0f0f0',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 12,
  },
  stepNumberText: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
  },
  stepInfo: {
    flex: 1,
  },
  stepTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#333',
  },
  stepCategory: {
    fontSize: 12,
    color: '#666',
    marginTop: 4,
  },
  stepDescription: {
    fontSize: 14,
    color: '#666',
    lineHeight: 20,
    marginBottom: 12,
  },
  resourcesContainer: {
    marginTop: 12,
    paddingTop: 12,
    borderTopWidth: 1,
    borderTopColor: '#e0e0e0',
  },
  resourcesTitle: {
    fontSize: 12,
    fontWeight: '600',
    color: '#333',
    marginBottom: 8,
  },
  resourceItem: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
  },
  resourceText: {
    fontSize: 12,
    color: '#007AFF',
    marginLeft: 8,
  },
  completedDate: {
    fontSize: 12,
    color: '#4ECDC4',
    marginTop: 8,
    fontStyle: 'italic',
  },
});


