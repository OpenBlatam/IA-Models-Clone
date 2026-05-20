import React from 'react';
import { View, Text, StyleSheet, ScrollView } from 'react-native';
import { useTheme } from '../contexts/ThemeContext';
import { SimpleChart } from './SimpleChart';
import { StatCard } from './StatCard';
import { spacing, borderRadius, typography } from '../theme/colors';
import { formatNumber, formatDuration } from '../utils/format';

interface DataPoint {
  label: string;
  value: number;
  color?: string;
}

interface DataVisualizationProps {
  title?: string;
  data: DataPoint[];
  total?: number;
  showChart?: boolean;
  showStats?: boolean;
  chartOrientation?: 'horizontal' | 'vertical';
}

export const DataVisualization: React.FC<DataVisualizationProps> = ({
  title,
  data,
  total,
  showChart = true,
  showStats = true,
  chartOrientation = 'horizontal',
}) => {
  const { theme } = useTheme();

  const maxValue = Math.max(...data.map((d) => d.value), 1);
  const percentages = data.map((d) => ({
    ...d,
    percentage: total ? (d.value / total) * 100 : (d.value / maxValue) * 100,
  }));

  return (
    <View style={[styles.container, { backgroundColor: theme.surface, borderColor: theme.border }]}>
      {title && (
        <Text style={[styles.title, { color: theme.text }]}>{title}</Text>
      )}

      {showStats && (
        <View style={styles.statsContainer}>
          {data.map((point, index) => (
            <View key={index} style={styles.statItem}>
              <View
                style={[
                  styles.statIndicator,
                  { backgroundColor: point.color || theme.primary },
                ]}
              />
              <View style={styles.statInfo}>
                <Text style={[styles.statLabel, { color: theme.textSecondary }]}>
                  {point.label}
                </Text>
                <Text style={[styles.statValue, { color: theme.text }]}>
                  {formatNumber(point.value)}
                  {total && (
                    <Text style={[styles.statPercentage, { color: theme.textTertiary }]}>
                      {' '}({percentages[index].percentage.toFixed(1)}%)
                    </Text>
                  )}
                </Text>
              </View>
            </View>
          ))}
        </View>
      )}

      {showChart && (
        <View style={styles.chartContainer}>
          <SimpleChart
            data={data}
            orientation={chartOrientation}
            showValues={true}
          />
        </View>
      )}

      {total && (
        <View style={[styles.totalContainer, { borderTopColor: theme.border }]}>
          <Text style={[styles.totalLabel, { color: theme.textSecondary }]}>Total:</Text>
          <Text style={[styles.totalValue, { color: theme.text }]}>{formatNumber(total)}</Text>
        </View>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    borderRadius: borderRadius.lg,
    padding: spacing.xl,
    borderWidth: 1,
    margin: spacing.md,
  },
  title: {
    ...typography.h3,
    marginBottom: spacing.lg,
  },
  statsContainer: {
    marginBottom: spacing.lg,
  },
  statItem: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: spacing.md,
  },
  statIndicator: {
    width: 12,
    height: 12,
    borderRadius: 6,
    marginRight: spacing.md,
  },
  statInfo: {
    flex: 1,
  },
  statLabel: {
    ...typography.bodySmall,
    marginBottom: spacing.xs,
  },
  statValue: {
    ...typography.body,
    fontWeight: '600',
  },
  statPercentage: {
    ...typography.caption,
  },
  chartContainer: {
    marginTop: spacing.md,
  },
  totalContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingTop: spacing.md,
    marginTop: spacing.md,
    borderTopWidth: 1,
  },
  totalLabel: {
    ...typography.body,
    fontWeight: '600',
  },
  totalValue: {
    ...typography.h3,
  },
});

