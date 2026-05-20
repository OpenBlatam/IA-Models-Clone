import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { colors, spacing, borderRadius, typography } from '../theme/colors';

interface DataPoint {
  label: string;
  value: number;
  color?: string;
}

interface SimpleChartProps {
  data: DataPoint[];
  title?: string;
  maxValue?: number;
  showValues?: boolean;
  orientation?: 'horizontal' | 'vertical';
}

export const SimpleChart: React.FC<SimpleChartProps> = ({
  data,
  title,
  maxValue,
  showValues = true,
  orientation = 'horizontal',
}) => {
  const max = maxValue || Math.max(...data.map((d) => d.value), 1);

  if (orientation === 'vertical') {
    return (
      <View style={styles.container}>
        {title && <Text style={styles.title}>{title}</Text>}
        <View style={styles.verticalChart}>
          {data.map((point, index) => (
            <View key={index} style={styles.verticalBarContainer}>
              <View style={styles.verticalBarWrapper}>
                <View
                  style={[
                    styles.verticalBar,
                    {
                      height: `${(point.value / max) * 100}%`,
                      backgroundColor: point.color || colors.primary,
                    },
                  ]}
                />
              </View>
              <Text style={styles.verticalLabel} numberOfLines={1}>
                {point.label}
              </Text>
              {showValues && (
                <Text style={styles.verticalValue}>{point.value}</Text>
              )}
            </View>
          ))}
        </View>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      {title && <Text style={styles.title}>{title}</Text>}
      <View style={styles.chart}>
        {data.map((point, index) => (
          <View key={index} style={styles.barContainer}>
            <View style={styles.barInfo}>
              <Text style={styles.label} numberOfLines={1}>
                {point.label}
              </Text>
              {showValues && (
                <Text style={styles.value}>{point.value}</Text>
              )}
            </View>
            <View style={styles.barWrapper}>
              <View
                style={[
                  styles.bar,
                  {
                    width: `${(point.value / max) * 100}%`,
                    backgroundColor: point.color || colors.primary,
                  },
                ]}
              />
            </View>
          </View>
        ))}
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    marginVertical: spacing.md,
  },
  title: {
    ...typography.h3,
    color: colors.text,
    marginBottom: spacing.md,
  },
  chart: {
    gap: spacing.md,
  },
  barContainer: {
    marginBottom: spacing.sm,
  },
  barInfo: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: spacing.xs,
  },
  label: {
    ...typography.bodySmall,
    color: colors.text,
    flex: 1,
  },
  value: {
    ...typography.bodySmall,
    fontWeight: '600',
    color: colors.text,
    marginLeft: spacing.sm,
  },
  barWrapper: {
    height: 8,
    backgroundColor: colors.surfaceVariant,
    borderRadius: borderRadius.full,
    overflow: 'hidden',
  },
  bar: {
    height: '100%',
    borderRadius: borderRadius.full,
  },
  verticalChart: {
    flexDirection: 'row',
    alignItems: 'flex-end',
    height: 200,
    gap: spacing.sm,
    paddingHorizontal: spacing.sm,
  },
  verticalBarContainer: {
    flex: 1,
    alignItems: 'center',
  },
  verticalBarWrapper: {
    width: '100%',
    height: 150,
    backgroundColor: colors.surfaceVariant,
    borderRadius: borderRadius.sm,
    overflow: 'hidden',
    justifyContent: 'flex-end',
    marginBottom: spacing.xs,
  },
  verticalBar: {
    width: '100%',
    borderRadius: borderRadius.sm,
  },
  verticalLabel: {
    ...typography.caption,
    color: colors.textSecondary,
    textAlign: 'center',
    fontSize: 10,
  },
  verticalValue: {
    ...typography.caption,
    color: colors.text,
    fontWeight: '600',
    marginTop: spacing.xs,
  },
});

