import React, { useCallback, useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useRouter } from 'expo-router';
import { useAnalyzeTrackById } from '../../hooks/use-music-analysis';
import { LoadingSpinner } from '../common/loading-spinner';
import { ErrorMessage } from '../common/error-message';
import { VisualizationCard } from './visualization-card';
import { AudioFeaturesChart } from './audio-features-chart';
import { FeatureChart } from './feature-chart';
import { AudioWaveform } from './audio-waveform';
import { ExportModal } from './export-modal';
import { AnimatedView } from '../common/animated-view';
import { useToast } from '../../contexts/toast-context';
import { COLORS, SPACING, TYPOGRAPHY, BORDER_RADIUS } from '../../constants/config';
import { formatBPM, formatPercentage, formatKeySignature } from '../../utils/formatters';
import { useTranslation } from '../../hooks/use-translation';
import { logError } from '../../utils/error-handler';
import type { Track } from '../../types/api';

interface AnalysisScreenProps {
  track: Track;
}

export function AnalysisScreen({ track }: AnalysisScreenProps) {
  const { t } = useTranslation();
  const router = useRouter();
  const { showToast } = useToast();
  const [showExportModal, setShowExportModal] = useState(false);
  const {
    data: analysis,
    isLoading,
    error,
    refetch,
  } = useAnalyzeTrackById(track.id, true);

  const handleRetry = useCallback(() => {
    if (error) {
      logError(error, 'AnalysisScreen');
    }
    refetch();
  }, [error, refetch]);

  const handleViewRecommendations = useCallback(() => {
    router.push({
      pathname: '/recommendations',
      params: { trackId: track.id },
    });
  }, [router, track.id]);

  const handleExport = useCallback(() => {
    setShowExportModal(true);
  }, []);

  const handleExportSuccess = useCallback(() => {
    showToast('Analysis exported successfully!', 'success');
  }, [showToast]);

  if (isLoading) {
    return (
      <SafeAreaView style={styles.container} edges={['top', 'bottom']}>
        <LoadingSpinner message="Analyzing track..." />
      </SafeAreaView>
    );
  }

  if (error) {
    return (
      <SafeAreaView style={styles.container} edges={['top', 'bottom']}>
        <ErrorMessage
          message={error.message || t('errors.serverError')}
          onRetry={handleRetry}
          retryLabel={t('common.retry')}
        />
      </SafeAreaView>
    );
  }

  if (!analysis) {
    return null;
  }

  return (
    <SafeAreaView style={styles.container} edges={['top', 'bottom']}>
      <ScrollView
        style={styles.scrollView}
        contentContainerStyle={styles.content}
        showsVerticalScrollIndicator={false}
      >
        <View style={styles.header}>
          <Text style={styles.trackTitle}>{track.name}</Text>
          <Text style={styles.artists}>
            {track.artists.join(', ')}
          </Text>
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle} accessibilityRole="header">
            {t('analysis.musicalAnalysis')}
          </Text>
          <View style={styles.card}>
            <View style={styles.row}>
              <Text style={styles.label}>{t('analysis.keySignature')}:</Text>
              <Text style={styles.value}>
                {formatKeySignature(
                  analysis.musical_analysis.root_note,
                  analysis.musical_analysis.mode
                )}
              </Text>
            </View>
            <View style={styles.row}>
              <Text style={styles.label}>{t('analysis.tempo')}:</Text>
              <Text style={styles.value}>
                {formatBPM(analysis.musical_analysis.tempo.bpm)} -{' '}
                {analysis.musical_analysis.tempo.category}
              </Text>
            </View>
            <View style={styles.row}>
              <Text style={styles.label}>{t('analysis.timeSignature')}:</Text>
              <Text style={styles.value}>
                {analysis.musical_analysis.time_signature}
              </Text>
            </View>
            <View style={styles.row}>
              <Text style={styles.label}>{t('analysis.scale')}:</Text>
              <Text style={styles.value}>
                {analysis.musical_analysis.scale.name}
              </Text>
            </View>
          </View>
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle} accessibilityRole="header">
            {t('analysis.technicalFeatures')}
          </Text>
          <AudioFeaturesChart features={analysis.technical_analysis} />
          <FeatureChart features={analysis.technical_analysis} />
          <View style={styles.waveformContainer}>
            <AudioWaveform bars={30} height={60} animated={true} />
          </View>
        </View>

        <View style={styles.actionsContainer}>
          <TouchableOpacity
            style={styles.actionButton}
            onPress={handleViewRecommendations}
            accessibilityRole="button"
            accessibilityLabel={t('common.recommendations')}
          >
            <Text style={styles.actionButtonText}>
              {t('common.recommendations')} →
            </Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[styles.actionButton, styles.exportButton]}
            onPress={handleExport}
            accessibilityRole="button"
            accessibilityLabel="Export analysis"
          >
            <Text style={styles.actionButtonText}>📤 Export</Text>
          </TouchableOpacity>
        </View>

        {analysis.coaching && (
          <View style={styles.section}>
            <Text style={styles.sectionTitle} accessibilityRole="header">
              {t('analysis.coaching')}
            </Text>
            {analysis.coaching.learning_path.map((step, index) => (
              <View key={index} style={styles.card}>
                <Text style={styles.stepTitle}>
                  {t('analysis.step')} {step.step}: {step.title}
                </Text>
                <Text style={styles.stepDescription}>{step.description}</Text>
                {step.exercises.length > 0 && (
                  <View style={styles.exercises}>
                    {step.exercises.map((exercise, exIndex) => (
                      <Text key={exIndex} style={styles.exercise}>
                        • {exercise}
                      </Text>
                    ))}
                  </View>
                )}
              </View>
            ))}
            {analysis.coaching.tips.length > 0 && (
              <View style={styles.card}>
                <Text style={styles.tipsTitle}>{t('analysis.tips')}</Text>
                {analysis.coaching.tips.map((tip, index) => (
                  <Text key={index} style={styles.tip}>
                    • {tip}
                  </Text>
                ))}
              </View>
            )}
          </View>
        )}
      </ScrollView>

      <ExportModal
        visible={showExportModal}
        trackId={track.id}
        trackName={track.name}
        onClose={() => setShowExportModal(false)}
      />
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
  header: {
    marginBottom: SPACING.lg,
    paddingBottom: SPACING.md,
    borderBottomWidth: 1,
    borderBottomColor: COLORS.surfaceLight,
  },
  trackTitle: {
    ...TYPOGRAPHY.h1,
    color: COLORS.text,
    marginBottom: SPACING.xs,
  },
  artists: {
    ...TYPOGRAPHY.body,
    color: COLORS.textSecondary,
  },
  section: {
    marginBottom: SPACING.lg,
  },
  sectionTitle: {
    ...TYPOGRAPHY.h2,
    color: COLORS.text,
    marginBottom: SPACING.md,
  },
  card: {
    backgroundColor: COLORS.surface,
    borderRadius: BORDER_RADIUS.md,
    padding: SPACING.md,
    marginBottom: SPACING.sm,
  },
  row: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: SPACING.sm,
  },
  label: {
    ...TYPOGRAPHY.body,
    color: COLORS.textSecondary,
  },
  value: {
    ...TYPOGRAPHY.body,
    color: COLORS.text,
    fontWeight: '600',
  },
  stepTitle: {
    ...TYPOGRAPHY.h3,
    color: COLORS.primary,
    marginBottom: SPACING.xs,
  },
  stepDescription: {
    ...TYPOGRAPHY.body,
    color: COLORS.text,
    marginBottom: SPACING.sm,
  },
  exercises: {
    marginTop: SPACING.sm,
  },
  exercise: {
    ...TYPOGRAPHY.bodySmall,
    color: COLORS.textSecondary,
    marginBottom: SPACING.xs,
  },
  tipsTitle: {
    ...TYPOGRAPHY.h3,
    color: COLORS.text,
    marginBottom: SPACING.sm,
  },
  tip: {
    ...TYPOGRAPHY.body,
    color: COLORS.text,
    marginBottom: SPACING.xs,
  },
  actionsContainer: {
    flexDirection: 'row',
    gap: SPACING.sm,
    margin: SPACING.md,
  },
  actionButton: {
    flex: 1,
    backgroundColor: COLORS.primary,
    padding: SPACING.md,
    borderRadius: BORDER_RADIUS.md,
    alignItems: 'center',
  },
  exportButton: {
    backgroundColor: COLORS.secondary,
  },
  actionButtonText: {
    ...TYPOGRAPHY.body,
    color: COLORS.text,
    fontWeight: '600',
  },
  waveformContainer: {
    marginTop: SPACING.md,
    padding: SPACING.md,
    backgroundColor: COLORS.surface,
    borderRadius: BORDER_RADIUS.md,
  },
});

