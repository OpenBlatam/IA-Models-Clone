import React, { memo } from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { GestureDetector, Gesture } from 'react-native-gesture-handler';
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  withSpring,
  runOnJS,
} from 'react-native-reanimated';
import { useTheme } from '@/theme/theme';
import { Card } from '@/components/ui/Card';
import type { JobCardProps } from '../types';
import { SWIPE_THRESHOLD } from '../constants';
import { Dimensions } from 'react-native';

const { width: SCREEN_WIDTH } = Dimensions.get('window');
const CARD_WIDTH = SCREEN_WIDTH - 40;

function JobCardComponent({ job, onSwipe, onApply }: JobCardProps) {
  const theme = useTheme();
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
        runOnJS(onSwipe)({ jobId: job.id, action });
        translateX.value = withSpring(e.translationX > 0 ? CARD_WIDTH : -CARD_WIDTH);
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
      <Animated.View style={animatedStyle}>
        <Card style={styles.card}>
          <View style={styles.header}>
            <View style={styles.companyInfo}>
              <View style={[styles.companyLogo, { backgroundColor: theme.colors.primary }]}>
                <Text style={styles.companyInitial}>{job.company.charAt(0).toUpperCase()}</Text>
              </View>
              <View>
                <Text style={[styles.companyName, { color: theme.colors.text }]}>
                  {job.company}
                </Text>
                <Text style={[styles.location, { color: theme.colors.textSecondary }]}>
                  <Ionicons name="location" size={14} color={theme.colors.textSecondary} />{' '}
                  {job.location}
                </Text>
              </View>
            </View>
            {job.match_score && (
              <View style={[styles.matchBadge, { backgroundColor: theme.colors.secondary }]}>
                <Text style={styles.matchScore}>{job.match_score}%</Text>
                <Text style={styles.matchLabel}>Match</Text>
              </View>
            )}
          </View>

          <Text style={[styles.title, { color: theme.colors.text }]}>{job.title}</Text>

          {job.salary_range && (
            <View style={styles.salaryContainer}>
              <Ionicons name="cash" size={16} color={theme.colors.secondary} />
              <Text style={[styles.salary, { color: theme.colors.secondary }]}>
                {job.salary_range}
              </Text>
            </View>
          )}

          <Text style={[styles.description, { color: theme.colors.textSecondary }]} numberOfLines={4}>
            {job.description}
          </Text>

          {job.required_skills && job.required_skills.length > 0 && (
            <View style={styles.skillsContainer}>
              <Text style={[styles.skillsTitle, { color: theme.colors.text }]}>
                Required Skills:
              </Text>
              <View style={styles.skillsList}>
                {job.required_skills.slice(0, 5).map((skill, index) => (
                  <View
                    key={index}
                    style={[styles.skillTag, { backgroundColor: theme.colors.surface }]}
                  >
                    <Text style={[styles.skillText, { color: theme.colors.text }]}>{skill}</Text>
                  </View>
                ))}
              </View>
            </View>
          )}
        </Card>
      </Animated.View>
    </GestureDetector>
  );
}

export const JobCard = memo(JobCardComponent);

const styles = StyleSheet.create({
  card: {
    width: CARD_WIDTH,
  },
  header: {
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
  },
  location: {
    fontSize: 14,
    marginTop: 4,
  },
  matchBadge: {
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
  title: {
    fontSize: 22,
    fontWeight: 'bold',
    marginBottom: 12,
  },
  salaryContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 16,
  },
  salary: {
    fontSize: 16,
    fontWeight: '600',
    marginLeft: 8,
  },
  description: {
    fontSize: 14,
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
    marginBottom: 8,
  },
  skillsList: {
    flexDirection: 'row',
    flexWrap: 'wrap',
  },
  skillTag: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 12,
    marginRight: 8,
    marginBottom: 8,
  },
  skillText: {
    fontSize: 12,
  },
});

