import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { VisualizationCard } from './visualization-card';
import { COLORS, SPACING, TYPOGRAPHY } from '../../constants/config';
import type { TechnicalAnalysis } from '../../types/api';

interface AudioFeaturesChartProps {
  features: TechnicalAnalysis;
}

export function AudioFeaturesChart({ features }: AudioFeaturesChartProps) {
  return (
    <View style={styles.container}>
      <Text style={styles.title} accessibilityRole="header">
        Audio Features
      </Text>
      <View style={styles.grid}>
        <View style={styles.row}>
          <VisualizationCard
            title="Energy"
            value={features.energy}
            color={COLORS.error}
          />
          <VisualizationCard
            title="Danceability"
            value={features.danceability}
            color={COLORS.success}
          />
        </View>
        <View style={styles.row}>
          <VisualizationCard
            title="Valence"
            value={features.valence}
            color={COLORS.warning}
          />
          <VisualizationCard
            title="Acousticness"
            value={features.acousticness}
            color={COLORS.info}
          />
        </View>
        <View style={styles.row}>
          <VisualizationCard
            title="Instrumentalness"
            value={features.instrumentalness}
            color={COLORS.secondary}
          />
          <VisualizationCard
            title="Liveness"
            value={features.liveness}
            color={COLORS.primary}
          />
        </View>
        <View style={styles.row}>
          <VisualizationCard
            title="Speechiness"
            value={features.speechiness}
            color={COLORS.primaryDark}
          />
        </View>
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
  grid: {
    gap: SPACING.sm,
  },
  row: {
    flexDirection: 'row',
    gap: SPACING.sm,
  },
});

