import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
} from 'react-native';
import { useNavigation } from '@react-navigation/native';
import { useStatsQuery, useQueueStatusQuery } from '../hooks/useProjectsQuery';
import { LoadingSpinner } from '../components/LoadingSpinner';
import { ErrorMessage } from '../components/ErrorMessage';
import { StatCard } from '../components/StatCard';
import { ProgressBar } from '../components/ProgressBar';
import { SimpleChart } from '../components/SimpleChart';
import { DataVisualization } from '../components/DataVisualization';
import { useTheme } from '../contexts/ThemeContext';
import { useAnalytics } from '../hooks/useAnalytics';
import { spacing, borderRadius, typography } from '../theme/colors';
import { formatDuration } from '../utils/format';

export const HomeScreen: React.FC = () => {
  const navigation = useNavigation();
  const { theme } = useTheme();
  const analytics = useAnalytics();
  const {
    data: stats,
    isLoading: statsLoading,
    error: statsError,
    refetch: refetchStats,
  } = useStatsQuery();
  const {
    data: queueStatus,
    isLoading: queueLoading,
    error: queueError,
    refetch: refetchQueue,
  } = useQueueStatusQuery();

  React.useEffect(() => {
    analytics.trackScreenView('HomeScreen');
  }, []);

  const isLoading = statsLoading || queueLoading;
  const error = statsError || queueError;

  const handleRefresh = () => {
    refetchStats();
    refetchQueue();
  };

  if (isLoading && !stats && !queueStatus) {
    return <LoadingSpinner message="Cargando estadísticas..." />;
  }

  if (error) {
    return <ErrorMessage error={error} onRetry={handleRefresh} />;
  }

  const totalProjects = stats?.total_projects || 0;
  const completedProjects = stats?.completed_projects || 0;
  const failedProjects = stats?.failed_projects || 0;
  const successRate =
    totalProjects > 0 ? (completedProjects / totalProjects) * 100 : 0;

  return (
    <ScrollView style={[styles.container, { backgroundColor: theme.background }]}>
      <View style={[styles.header, { backgroundColor: theme.surface, borderBottomColor: theme.border }]}>
        <Text style={[styles.title, { color: theme.text }]}>AI Project Generator</Text>
        <Text style={[styles.subtitle, { color: theme.textSecondary }]}>Panel de Control</Text>
      </View>

      <View style={styles.statsContainer}>
        <StatCard
          label="Total Proyectos"
          value={totalProjects}
          icon="📊"
          color={theme.primary}
        />
        <StatCard
          label="Completados"
          value={completedProjects}
          icon="✅"
          color={theme.success}
        />
        <StatCard
          label="Fallidos"
          value={failedProjects}
          icon="❌"
          color={theme.error}
        />
      </View>

      {totalProjects > 0 && (
        <View style={[styles.progressContainer, { backgroundColor: theme.surface, shadowColor: theme.shadow }]}>
          <ProgressBar
            progress={completedProjects}
            total={totalProjects}
            label="Tasa de Éxito"
            color={theme.success}
          />
        </View>
      )}

      {queueStatus && (
        <View style={[styles.queueContainer, { backgroundColor: theme.surface, shadowColor: theme.shadow }]}>
          <Text style={[styles.sectionTitle, { color: theme.text }]}>Estado de la Cola</Text>
          <DataVisualization
            data={[
              { label: 'En Cola', value: queueStatus.queued, color: theme.status.queued },
              { label: 'Procesando', value: queueStatus.processing, color: theme.status.processing },
              { label: 'Completados', value: queueStatus.completed, color: theme.status.completed },
              { label: 'Fallidos', value: queueStatus.failed, color: theme.status.failed },
            ]}
            total={queueStatus.queued + queueStatus.processing + queueStatus.completed + queueStatus.failed}
            showChart={true}
            showStats={true}
            chartOrientation="horizontal"
          />
        </View>
      )}

      {stats && (stats.average_generation_time || stats.uptime) && (
        <View style={[styles.metricsContainer, { backgroundColor: theme.surface, shadowColor: theme.shadow }]}>
          <Text style={[styles.sectionTitle, { color: theme.text }]}>Métricas</Text>
          {stats.average_generation_time && (
            <View style={styles.metricRow}>
              <Text style={[styles.metricLabel, { color: theme.textSecondary }]}>Tiempo promedio:</Text>
              <Text style={[styles.metricValue, { color: theme.text }]}>
                {formatDuration(stats.average_generation_time)}
              </Text>
            </View>
          )}
          {stats.uptime && (
            <View style={styles.metricRow}>
              <Text style={[styles.metricLabel, { color: theme.textSecondary }]}>Uptime:</Text>
              <Text style={[styles.metricValue, { color: theme.text }]}>
                {formatDuration(stats.uptime)}
              </Text>
            </View>
          )}
        </View>
      )}

      <View style={styles.actionsContainer}>
        <TouchableOpacity
          style={[styles.actionButton, { backgroundColor: theme.primary, shadowColor: theme.shadow }]}
          onPress={() => {
            analytics.trackUserAction('navigate_to_generate');
            navigation.navigate('Generate' as never);
          }}
        >
          <Text style={styles.actionButtonIcon}>🚀</Text>
          <Text style={[styles.actionButtonText, { color: theme.surface }]}>Generar Proyecto</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.actionButton, styles.secondaryButton, { backgroundColor: theme.surface, borderColor: theme.primary, shadowColor: theme.shadow }]}
          onPress={() => {
            analytics.trackUserAction('navigate_to_projects');
            navigation.navigate('Projects' as never);
          }}
        >
          <Text style={styles.actionButtonIcon}>📋</Text>
          <Text style={[styles.actionButtonText, styles.secondaryButtonText, { color: theme.primary }]}>
            Ver Proyectos
          </Text>
        </TouchableOpacity>
      </View>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  header: {
    padding: spacing.xl,
    borderBottomWidth: 1,
  },
  title: {
    ...typography.h1,
    marginBottom: spacing.xs,
  },
  subtitle: {
    ...typography.body,
  },
  statsContainer: {
    flexDirection: 'row',
    padding: spacing.lg,
    gap: spacing.md,
  },
  progressContainer: {
    marginHorizontal: spacing.lg,
    marginBottom: spacing.lg,
    padding: spacing.lg,
    borderRadius: borderRadius.lg,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  queueContainer: {
    margin: spacing.lg,
    borderRadius: borderRadius.lg,
    padding: spacing.lg,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  sectionTitle: {
    ...typography.h3,
    marginBottom: spacing.md,
  },
  queueStats: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: spacing.md,
  },
  queueItem: {
    flex: 1,
    minWidth: '45%',
    alignItems: 'center',
    padding: spacing.md,
    backgroundColor: colors.surfaceVariant,
    borderRadius: borderRadius.md,
  },
  queueIndicator: {
    width: 12,
    height: 12,
    borderRadius: 6,
    marginBottom: spacing.xs,
  },
  queueIndicatorQueued: {
    backgroundColor: colors.status.queued,
  },
  queueIndicatorProcessing: {
    backgroundColor: colors.status.processing,
  },
  queueIndicatorCompleted: {
    backgroundColor: colors.status.completed,
  },
  queueIndicatorFailed: {
    backgroundColor: colors.status.failed,
  },
  queueValue: {
    ...typography.h3,
    color: colors.text,
    marginBottom: spacing.xs,
  },
  queueLabel: {
    ...typography.caption,
    color: colors.textSecondary,
  },
  metricsContainer: {
    margin: spacing.lg,
    borderRadius: borderRadius.lg,
    padding: spacing.lg,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  metricRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: spacing.sm,
  },
  metricLabel: {
    ...typography.body,
  },
  metricValue: {
    ...typography.body,
    fontWeight: '600',
  },
  actionsContainer: {
    padding: spacing.lg,
    gap: spacing.md,
  },
  actionButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: borderRadius.lg,
    padding: spacing.lg,
    gap: spacing.sm,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  secondaryButton: {
    borderWidth: 2,
  },
  actionButtonIcon: {
    fontSize: 20,
  },
  actionButtonText: {
    ...typography.body,
    fontWeight: '600',
  },
  secondaryButtonText: {
  },
});
