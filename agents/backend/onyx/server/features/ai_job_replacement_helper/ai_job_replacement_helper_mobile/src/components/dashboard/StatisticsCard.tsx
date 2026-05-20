import React, { memo } from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { useTheme } from '@/theme/theme';
import { Card } from '@/components/ui/Card';

interface Statistics {
  applications_sent: number;
  interviews_completed: number;
  skills_learned: number;
  days_active: number;
}

interface StatisticsCardProps {
  statistics: Statistics;
}

function StatisticsCardComponent({ statistics }: StatisticsCardProps) {
  const theme = useTheme();

  const stats = [
    { label: 'Applications', value: statistics.applications_sent || 0, color: theme.colors.primary },
    {
      label: 'Interviews',
      value: statistics.interviews_completed || 0,
      color: theme.colors.secondary,
    },
    { label: 'Skills', value: statistics.skills_learned || 0, color: theme.colors.accent },
    { label: 'Days Active', value: statistics.days_active || 0, color: theme.colors.success },
  ];

  return (
    <Card accessibilityLabel="Statistics card">
      <Text style={[styles.title, { color: theme.colors.text }]}>Statistics</Text>
      <View style={styles.statsGrid}>
        {stats.map((stat, index) => (
          <View
            key={index}
            style={[styles.statBox, { backgroundColor: theme.colors.surface }]}
            accessibilityRole="text"
            accessibilityLabel={`${stat.label}: ${stat.value}`}
          >
            <Text style={[styles.statBoxValue, { color: stat.color }]}>{stat.value}</Text>
            <Text style={[styles.statBoxLabel, { color: theme.colors.textSecondary }]}>
              {stat.label}
            </Text>
          </View>
        ))}
      </View>
    </Card>
  );
}

export const StatisticsCard = memo(StatisticsCardComponent);

const styles = StyleSheet.create({
  title: {
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 16,
  },
  statsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
  },
  statBox: {
    width: '48%',
    padding: 16,
    borderRadius: 12,
    marginBottom: 12,
    alignItems: 'center',
  },
  statBoxValue: {
    fontSize: 24,
    fontWeight: 'bold',
  },
  statBoxLabel: {
    fontSize: 12,
    marginTop: 4,
  },
});


