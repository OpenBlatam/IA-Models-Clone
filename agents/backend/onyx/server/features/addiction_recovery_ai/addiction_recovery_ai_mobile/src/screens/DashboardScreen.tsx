import React, { useMemo, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  RefreshControl,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { ProgressCard, LoadingSpinner } from '@/components';
import { useDashboard } from '@/hooks/useApi';
import { useAuthStore } from '@/store/auth-store';
import { useColors } from '@/theme/colors';
import { format } from 'date-fns';

export function DashboardScreen(): JSX.Element {
  const colors = useColors();
  const { user } = useAuthStore();
  const { data: dashboard, isLoading, refetch, isRefetching } = useDashboard(
    user?.user_id || null
  );

  const onRefresh = useCallback(() => {
    refetch();
  }, [refetch]);

  const riskLevelText = useMemo(() => {
    if (!dashboard) return '';
    const level = dashboard.risk_level;
    const map: Record<string, string> = {
      low: 'Bajo',
      moderate: 'Moderado',
      high: 'Alto',
      critical: 'Crítico',
    };
    return map[level] || level;
  }, [dashboard]);

  const riskLevelColor = useMemo(() => {
    if (!dashboard) return colors.text;
    const level = dashboard.risk_level;
    const colorMap: Record<string, string> = {
      low: colors.success,
      moderate: colors.warning,
      high: colors.error,
      critical: colors.textSecondary,
    };
    return colorMap[level] || colors.text;
  }, [dashboard, colors]);

  if (isLoading) {
    return <LoadingSpinner />;
  }

  if (!dashboard) {
    return (
      <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]} edges={['top', 'bottom']}>
        <View style={styles.emptyContainer}>
          <Text style={[styles.emptyText, { color: colors.textSecondary }]}>
            No hay datos disponibles
          </Text>
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]} edges={['top']}>
      <ScrollView
        style={styles.scrollView}
        refreshControl={
          <RefreshControl refreshing={isRefetching} onRefresh={onRefresh} />
        }
        contentContainerStyle={styles.scrollContent}
      >
        <View style={[styles.header, { backgroundColor: colors.surface, borderBottomColor: colors.border }]}>
          <Text
            style={[styles.greeting, { color: colors.text }]}
            accessibilityRole="text"
          >
            Hola, {user?.name || user?.user_id}
          </Text>
          <Text
            style={[styles.date, { color: colors.textSecondary }]}
            accessibilityRole="text"
          >
            {format(new Date(), "EEEE, d 'de' MMMM")}
          </Text>
        </View>

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

          <View style={[styles.riskCard, { backgroundColor: colors.card, shadowColor: colors.shadow }]}>
            <Text style={[styles.riskTitle, { color: colors.textSecondary }]}>
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

        {dashboard.achievements && dashboard.achievements.length > 0 && (
          <View style={[styles.section, { backgroundColor: colors.surface }]}>
            <Text
              style={[styles.sectionTitle, { color: colors.text }]}
              accessibilityRole="header"
            >
              Logros Recientes
            </Text>
            {dashboard.achievements.slice(0, 3).map((achievement) => (
              <View
                key={achievement.id}
                style={[styles.achievementItem, { borderBottomColor: colors.border }]}
                accessibilityRole="text"
              >
                <Text style={[styles.achievementTitle, { color: colors.text }]}>
                  {achievement.title}
                </Text>
                <Text style={[styles.achievementDescription, { color: colors.textSecondary }]}>
                  {achievement.description}
                </Text>
              </View>
            ))}
          </View>
        )}

        {dashboard.upcoming_reminders && dashboard.upcoming_reminders.length > 0 && (
          <View style={[styles.section, { backgroundColor: colors.surface }]}>
            <Text
              style={[styles.sectionTitle, { color: colors.text }]}
              accessibilityRole="header"
            >
              Recordatorios
            </Text>
            {dashboard.upcoming_reminders.slice(0, 3).map((reminder) => (
              <View
                key={reminder.id}
                style={[styles.reminderItem, { borderBottomColor: colors.border }]}
                accessibilityRole="text"
              >
                <Text style={[styles.reminderTitle, { color: colors.text }]}>
                  {reminder.title}
                </Text>
                <Text style={[styles.reminderTime, { color: colors.textSecondary }]}>
                  {reminder.time}
                </Text>
              </View>
            ))}
          </View>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    paddingBottom: 16,
  },
  emptyContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  emptyText: {
    fontSize: 16,
  },
  header: {
    padding: 24,
    borderBottomWidth: 1,
  },
  greeting: {
    fontSize: 28,
    fontWeight: 'bold',
    marginBottom: 4,
  },
  date: {
    fontSize: 16,
  },
  cardsContainer: {
    padding: 16,
  },
  riskCard: {
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  riskTitle: {
    fontSize: 14,
    marginBottom: 4,
  },
  riskValue: {
    fontSize: 24,
    fontWeight: 'bold',
  },
  section: {
    padding: 16,
    marginTop: 8,
    marginHorizontal: 16,
    borderRadius: 12,
    marginBottom: 16,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 12,
  },
  achievementItem: {
    paddingVertical: 12,
    borderBottomWidth: 1,
  },
  achievementTitle: {
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 4,
  },
  achievementDescription: {
    fontSize: 14,
  },
  reminderItem: {
    paddingVertical: 12,
    borderBottomWidth: 1,
  },
  reminderTitle: {
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 4,
  },
  reminderTime: {
    fontSize: 14,
  },
});
