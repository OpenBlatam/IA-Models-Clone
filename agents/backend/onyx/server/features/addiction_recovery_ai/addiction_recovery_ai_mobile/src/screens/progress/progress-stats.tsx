import React from 'react';
import { View, Text } from 'react-native';
import { useColors } from '@/theme/colors';
import type { StatsResponse } from '@/types';
import { useProgressStyles } from './progress-screen.styles';

interface ProgressStatsProps {
  stats: StatsResponse;
}

export function ProgressStats({ stats }: ProgressStatsProps): JSX.Element {
  const colors = useColors();
  const styles = useProgressStyles(colors);

  return (
    <View style={styles.statsSection}>
      <Text style={styles.sectionTitle}>Estadísticas</Text>
      <View style={styles.statRow}>
        <Text style={styles.statLabel}>Días Totales:</Text>
        <Text style={styles.statValue}>{stats.total_days}</Text>
      </View>
      <View style={styles.statRow}>
        <Text style={styles.statLabel}>Recaídas:</Text>
        <Text style={styles.statValue}>{stats.relapse_count}</Text>
      </View>
      <View style={styles.statRow}>
        <Text style={styles.statLabel}>Ansias Promedio:</Text>
        <Text style={styles.statValue}>
          {stats.average_cravings.toFixed(1)}/10
        </Text>
      </View>
      {stats.most_common_triggers.length > 0 && (
        <View style={styles.triggersSection}>
          <Text style={styles.statLabel}>Triggers Más Comunes:</Text>
          {stats.most_common_triggers.map((trigger, index) => (
            <Text key={index} style={styles.triggerItem}>
              • {trigger}
            </Text>
          ))}
        </View>
      )}
    </View>
  );
}

