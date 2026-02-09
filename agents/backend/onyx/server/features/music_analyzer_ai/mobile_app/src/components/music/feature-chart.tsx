import React from 'react';
import { View, Text, StyleSheet, Dimensions } from 'react-native';
import { VictoryBar, VictoryChart, VictoryTheme, VictoryAxis } from 'victory-native';
import { COLORS, SPACING, TYPOGRAPHY } from '../../constants/config';
import type { TechnicalAnalysis } from '../../types/api';

interface FeatureChartProps {
  features: TechnicalAnalysis;
}

const { width } = Dimensions.get('window');

export function FeatureChart({ features }: FeatureChartProps) {
  const data = [
    { feature: 'Energy', value: features.energy },
    { feature: 'Dance', value: features.danceability },
    { feature: 'Valence', value: features.valence },
    { feature: 'Acoustic', value: features.acousticness },
    { feature: 'Instrumental', value: features.instrumentalness },
    { feature: 'Liveness', value: features.liveness },
    { feature: 'Speech', value: features.speechiness },
  ];

  return (
    <View style={styles.container}>
      <Text style={styles.title} accessibilityRole="header">
        Audio Features Chart
      </Text>
      <View style={styles.chartContainer}>
        <VictoryChart
          theme={VictoryTheme.material}
          width={width - SPACING.xl * 2}
          height={250}
          padding={{ left: 50, right: 20, top: 20, bottom: 50 }}
        >
          <VictoryAxis
            style={{
              axis: { stroke: COLORS.textSecondary },
              tickLabels: { fill: COLORS.textSecondary, fontSize: 10 },
            }}
          />
          <VictoryAxis
            dependentAxis
            style={{
              axis: { stroke: COLORS.textSecondary },
              tickLabels: { fill: COLORS.textSecondary, fontSize: 10 },
            }}
          />
          <VictoryBar
            data={data}
            x="feature"
            y="value"
            style={{
              data: {
                fill: COLORS.primary,
                width: 20,
              },
            }}
            animate={{
              duration: 1000,
              onLoad: { duration: 500 },
            }}
          />
        </VictoryChart>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    marginBottom: SPACING.lg,
  },
  title: {
    ...TYPOGRAPHY.h3,
    color: COLORS.text,
    marginBottom: SPACING.md,
  },
  chartContainer: {
    backgroundColor: COLORS.surface,
    borderRadius: 8,
    padding: SPACING.sm,
    alignItems: 'center',
  },
});

