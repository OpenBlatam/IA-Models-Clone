import React from 'react';
import { View, Text, ScrollView, Alert } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { ProgressCard, LoadingSpinner, Button } from '@/components';
import { useProgress, useStats, useLogEntry } from '@/hooks/api';
import { useAuthStore } from '@/store/auth-store';
import { useColors } from '@/theme/colors';
import { useProgressStyles } from './progress-screen.styles';
import { ProgressHeader } from './progress-header';
import { ProgressLogForm } from './progress-log-form';
import { ProgressCards } from './progress-cards';
import { ProgressStats } from './progress-stats';

export function ProgressScreen(): JSX.Element {
  const colors = useColors();
  const { user } = useAuthStore();
  const { data: progress, isLoading: progressLoading } = useProgress(
    user?.user_id || null
  );
  const { data: stats, isLoading: statsLoading } = useStats(
    user?.user_id || null
  );
  const styles = useProgressStyles(colors);

  if (progressLoading || statsLoading) {
    return <LoadingSpinner />;
  }

  return (
    <SafeAreaView style={styles.container} edges={['top', 'bottom']}>
      <ScrollView style={styles.scrollView}>
        <ProgressHeader />
        <ProgressLogForm />
        {progress && <ProgressCards progress={progress} />}
        {stats && <ProgressStats stats={stats} />}
      </ScrollView>
    </SafeAreaView>
  );
}

