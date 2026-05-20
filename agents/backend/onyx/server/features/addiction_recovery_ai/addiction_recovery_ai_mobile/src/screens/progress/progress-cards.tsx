import React from 'react';
import { View } from 'react-native';
import { ProgressCard } from '@/components';
import { useColors } from '@/theme/colors';
import type { ProgressResponse } from '@/types';
import { useProgressStyles } from './progress-screen.styles';

interface ProgressCardsProps {
  progress: ProgressResponse;
}

export function ProgressCards({ progress }: ProgressCardsProps): JSX.Element {
  const colors = useColors();
  const styles = useProgressStyles(colors);

  return (
    <View style={styles.cardsContainer}>
      <ProgressCard
        title="Días Sobrio"
        value={progress.days_sober}
        color={colors.success}
      />
      <ProgressCard
        title="Racha Actual"
        value={`${progress.streak_days} días`}
        color={colors.primary}
      />
      <ProgressCard
        title="Racha Más Larga"
        value={`${progress.longest_streak} días`}
        color={colors.secondary}
      />
      <ProgressCard
        title="Progreso"
        value={`${progress.progress_percentage.toFixed(0)}%`}
        color={colors.warning}
      />
    </View>
  );
}

