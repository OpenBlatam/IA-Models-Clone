import React, { useMemo } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useLocalSearchParams } from 'expo-router';
import { Card } from '../common/card';
import { VisualizationCard } from './visualization-card';
import { COLORS, SPACING, TYPOGRAPHY, BORDER_RADIUS } from '../../constants/config';
import { useTranslation } from '../../hooks/use-translation';
import type { TrackComparison } from '../../types/api';

export function ComparisonResultsScreen() {
  const { t } = useTranslation();
  const params = useLocalSearchParams<{ data: string }>();
  
  const comparison: TrackComparison | null = useMemo(() => {
    if (!params.data) return null;
    try {
      return JSON.parse(params.data);
    } catch {
      return null;
    }
  }, [params.data]);

  if (!comparison) {
    return (
      <SafeAreaView style={styles.container} edges={['top', 'bottom']}>
        <View style={styles.errorContainer}>
          <Text style={styles.errorText}>Invalid comparison data</Text>
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container} edges={['top', 'bottom']}>
      <ScrollView
        style={styles.scrollView}
        contentContainerStyle={styles.content}
        showsVerticalScrollIndicator={false}
      >
        <View style={styles.header}>
          <Text style={styles.title} accessibilityRole="header">
            Comparison Results
          </Text>
          <Text style={styles.summary}>{comparison.summary}</Text>
        </View>

        {comparison.tracks.map((track, index) => (
          <Card key={track.track_id} style={styles.trackCard} delay={index * 100}>
            <Text style={styles.trackName}>{track.track_name}</Text>
            <Text style={styles.artists}>
              {track.artists.join(', ')}
            </Text>

            <View style={styles.similaritiesContainer}>
              <Text style={styles.sectionTitle}>Similarities</Text>
              <View style={styles.metricsRow}>
                <VisualizationCard
                  title="Key"
                  value={track.similarities.key}
                  color={COLORS.primary}
                />
                <VisualizationCard
                  title="Tempo"
                  value={track.similarities.tempo}
                  color={COLORS.success}
                />
                <VisualizationCard
                  title="Energy"
                  value={track.similarities.energy}
                  color={COLORS.error}
                />
                <VisualizationCard
                  title="Overall"
                  value={track.similarities.overall}
                  color={COLORS.warning}
                />
              </View>
            </View>

            {track.differences.length > 0 && (
              <View style={styles.differencesContainer}>
                <Text style={styles.sectionTitle}>Differences</Text>
                {track.differences.map((diff, diffIndex) => (
                  <Text key={diffIndex} style={styles.difference}>
                    • {diff}
                  </Text>
                ))}
              </View>
            )}
          </Card>
        ))}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: COLORS.background,
  },
  scrollView: {
    flex: 1,
  },
  content: {
    padding: SPACING.md,
  },
  errorContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: SPACING.lg,
  },
  errorText: {
    ...TYPOGRAPHY.body,
    color: COLORS.error,
  },
  header: {
    marginBottom: SPACING.lg,
    paddingBottom: SPACING.md,
    borderBottomWidth: 1,
    borderBottomColor: COLORS.surfaceLight,
  },
  title: {
    ...TYPOGRAPHY.h1,
    color: COLORS.text,
    marginBottom: SPACING.sm,
  },
  summary: {
    ...TYPOGRAPHY.body,
    color: COLORS.textSecondary,
  },
  trackCard: {
    marginBottom: SPACING.md,
  },
  trackName: {
    ...TYPOGRAPHY.h2,
    color: COLORS.text,
    marginBottom: SPACING.xs,
  },
  artists: {
    ...TYPOGRAPHY.body,
    color: COLORS.textSecondary,
    marginBottom: SPACING.md,
  },
  similaritiesContainer: {
    marginTop: SPACING.md,
  },
  sectionTitle: {
    ...TYPOGRAPHY.h3,
    color: COLORS.text,
    marginBottom: SPACING.sm,
  },
  metricsRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: SPACING.sm,
  },
  differencesContainer: {
    marginTop: SPACING.md,
    paddingTop: SPACING.md,
    borderTopWidth: 1,
    borderTopColor: COLORS.surfaceLight,
  },
  difference: {
    ...TYPOGRAPHY.bodySmall,
    color: COLORS.textSecondary,
    marginBottom: SPACING.xs,
  },
});

