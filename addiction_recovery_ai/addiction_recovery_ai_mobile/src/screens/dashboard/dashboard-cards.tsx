import React, { useMemo } from 'react';
import { View, Text } from 'react-native';
import { ProgressCard } from '@/components';
import { useColors } from '@/theme/colors';
import type { DashboardResponse } from '@/types';
import { useDashboardStyles } from './dashboard-screen.styles';

interface DashboardCardsProps {
  dashboard: DashboardResponse;
}

export function DashboardCards({ dashboard }: DashboardCardsProps): JSX.Element {
  const colors = useColors();
  const styles = useDashboardStyles(colors);

  const riskLevelText = useMemo(() => {
    const level = dashboard.risk_level;
    const map: Record<string, string> = {
      low: 'Bajo',
      moderate: 'Moderado',
      high: 'Alto',
      critical: 'Crítico',
    };
    return map[level] || level;
  }, [dashboard.risk_level]);

  const riskLevelColor = useMemo(() => {
    const level = dashboard.risk_level;
    const colorMap: Record<string, string> = {
      low: colors.success,
      moderate: colors.warning,
      high: colors.error,
      critical: colors.textSecondary,
    };
    return colorMap[level] || colors.text;
  }, [dashboard.risk_level, colors]);

  return (
    <View style={styles.cardsContainer}>
      <ProgressCard
        title="Días Sobrio"
        value={dashboard.days_sober}
        subtitle="Sigue así"
        color={colors.success}
        accessibilityLabel={`Días sobrio: ${dashboard.days_sober}`}
      />

      <ProgressCard
        title="Racha Actual"
        value={dashboard.current_streak}
        subtitle="días consecutivos"
        color={colors.primary}
        accessibilityLabel={`Racha actual: ${dashboard.current_streak} días`}
      />

      <ProgressCard
        title="Progreso"
        value={`${dashboard.progress_percentage.toFixed(0)}%`}
        subtitle="Completado"
        color={colors.secondary}
        accessibilityLabel={`Progreso: ${dashboard.progress_percentage.toFixed(0)} por ciento`}
      />

      <View style={styles.riskCard}>
        <Text style={styles.riskTitle}>
          Nivel de Riesgo
        </Text>
        <Text
          style={[styles.riskValue, { color: riskLevelColor }]}
          accessibilityRole="text"
        >
          {riskLevelText}
        </Text>
      </View>
    </View>
  );
}

