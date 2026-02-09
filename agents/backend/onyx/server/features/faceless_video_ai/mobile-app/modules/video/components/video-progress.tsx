import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { useTheme } from '@/contexts/theme-context';
import { VIDEO_STATUS_COLORS } from '@/utils/constants';
import type { GenerationProgress } from '@/types/api';

interface VideoProgressProps {
  progress: GenerationProgress;
  showDetails?: boolean;
}

export function VideoProgress({ progress, showDetails = true }: VideoProgressProps) {
  const { colors } = useTheme();
  const statusColor = VIDEO_STATUS_COLORS[progress.status] || colors.primary;

  return (
    <View style={styles.container}>
      <View style={styles.progressHeader}>
        <Text style={[styles.progressText, { color: colors.text }]}>
          {progress.progress.toFixed(1)}%
        </Text>
        {progress.estimated_time_remaining && (
          <Text style={[styles.timeText, { color: colors.textSecondary }]}>
            {Math.round(progress.estimated_time_remaining)}s remaining
          </Text>
        )}
      </View>

      <View style={[styles.progressBar, { backgroundColor: colors.border }]}>
        <View
          style={[
            styles.progressFill,
            {
              width: `${progress.progress}%`,
              backgroundColor: statusColor,
            },
          ]}
        />
      </View>

      {showDetails && (
        <View style={styles.details}>
          <Text style={[styles.stepText, { color: colors.textSecondary }]}>
            {progress.current_step}
          </Text>
          <Text style={[styles.stepsText, { color: colors.textSecondary }]}>
            Step {progress.completed_steps} of {progress.total_steps}
          </Text>
        </View>
      )}

      {progress.message && (
        <Text style={[styles.message, { color: colors.textSecondary }]}>
          {progress.message}
        </Text>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    marginVertical: 16,
  },
  progressHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  progressText: {
    fontSize: 18,
    fontWeight: '600',
  },
  timeText: {
    fontSize: 12,
  },
  progressBar: {
    height: 8,
    borderRadius: 4,
    overflow: 'hidden',
    marginBottom: 8,
  },
  progressFill: {
    height: '100%',
    borderRadius: 4,
  },
  details: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 4,
  },
  stepText: {
    fontSize: 14,
    flex: 1,
  },
  stepsText: {
    fontSize: 12,
  },
  message: {
    fontSize: 12,
    marginTop: 4,
    fontStyle: 'italic',
  },
});

