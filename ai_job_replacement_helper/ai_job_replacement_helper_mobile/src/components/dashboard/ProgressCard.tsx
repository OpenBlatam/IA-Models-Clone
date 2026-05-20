import React, { memo, useMemo } from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useTheme } from '@/theme/theme';
import { Card } from '@/components/ui/Card';
import type { GamificationProgress } from '@/types';

interface ProgressCardProps {
  progress: GamificationProgress;
}

function ProgressCardComponent({ progress }: ProgressCardProps) {
  const theme = useTheme();

  const progressPercentage = useMemo(
    () => (progress.xp / progress.xp_to_next_level) * 100,
    [progress.xp, progress.xp_to_next_level]
  );

  return (
    <Card accessibilityLabel="Your progress card">
      <View style={styles.header}>
        <Text style={[styles.title, { color: theme.colors.text }]}>Your Progress</Text>
        <View style={[styles.levelBadge, { backgroundColor: theme.colors.primary }]}>
          <Text style={styles.levelText}>Level {progress.level}</Text>
        </View>
      </View>

      <View style={styles.progressContainer}>
        <View style={[styles.progressBar, { backgroundColor: theme.colors.border }]}>
          <View
            style={[
              styles.progressFill,
              {
                width: `${progressPercentage}%`,
                backgroundColor: theme.colors.primary,
              },
            ]}
            accessibilityRole="progressbar"
            accessibilityValue={{
              min: 0,
              max: progress.xp_to_next_level,
              now: progress.xp,
              text: `${progress.xp} of ${progress.xp_to_next_level} XP`,
            }}
          />
        </View>
        <Text style={[styles.progressText, { color: theme.colors.textSecondary }]}>
          {progress.xp} / {progress.xp_to_next_level} XP
        </Text>
      </View>

      <View style={styles.statsRow}>
        <View style={styles.statItem} accessibilityRole="text">
          <Ionicons name="trophy" size={20} color={theme.colors.accent} />
          <Text style={[styles.statValue, { color: theme.colors.text }]}>{progress.points}</Text>
          <Text style={[styles.statLabel, { color: theme.colors.textSecondary }]}>Points</Text>
        </View>
        <View style={styles.statItem} accessibilityRole="text">
          <Ionicons name="flame" size={20} color={theme.colors.error} />
          <Text style={[styles.statValue, { color: theme.colors.text }]}>{progress.streak}</Text>
          <Text style={[styles.statLabel, { color: theme.colors.textSecondary }]}>Day Streak</Text>
        </View>
        <View style={styles.statItem} accessibilityRole="text">
          <Ionicons name="medal" size={20} color={theme.colors.secondary} />
          <Text style={[styles.statValue, { color: theme.colors.text }]}>
            {progress.badges.length}
          </Text>
          <Text style={[styles.statLabel, { color: theme.colors.textSecondary }]}>Badges</Text>
        </View>
      </View>
    </Card>
  );
}

export const ProgressCard = memo(ProgressCardComponent);

const styles = StyleSheet.create({
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  title: {
    fontSize: 20,
    fontWeight: 'bold',
  },
  levelBadge: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 12,
  },
  levelText: {
    color: '#fff',
    fontSize: 12,
    fontWeight: '600',
  },
  progressContainer: {
    marginTop: 8,
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
  progressText: {
    fontSize: 12,
  },
  statsRow: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    marginTop: 16,
  },
  statItem: {
    alignItems: 'center',
  },
  statValue: {
    fontSize: 18,
    fontWeight: 'bold',
    marginTop: 4,
  },
  statLabel: {
    fontSize: 12,
    marginTop: 4,
  },
});


