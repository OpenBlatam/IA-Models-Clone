import React, { useMemo } from 'react';
import {
  View,
  Text,
  ScrollView,
  StyleSheet,
  useWindowDimensions,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useTheme } from '@/contexts/theme-context';
import { DashboardCard } from '@/components/dashboard/dashboard-card';
import { SystemStatusCard } from '@/components/dashboard/system-status-card';
import { MetricsOverview } from '@/components/dashboard/metrics-overview';
import { QuickActions } from '@/components/dashboard/quick-actions';

export default function DashboardScreen(): JSX.Element {
  const { theme } = useTheme();
  const { width, height } = useWindowDimensions();

  const styles = useMemo(() => createStyles(theme, width, height), [theme, width, height]);

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView
        style={styles.scrollView}
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={false}
      >
        {/* Header */}
        <View style={styles.header}>
          <Text style={styles.title}>Blaze AI Dashboard</Text>
          <Text style={styles.subtitle}>System Overview & Monitoring</Text>
        </View>

        {/* System Status */}
        <SystemStatusCard />

        {/* Quick Actions */}
        <QuickActions />

        {/* Metrics Overview */}
        <MetricsOverview />

        {/* Module Cards */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Active Modules</Text>
          <View style={styles.cardsContainer}>
            <DashboardCard
              title="Cache Module"
              subtitle="LRU Strategy Active"
              status="healthy"
              icon="flash-outline"
              onPress={() => {}}
            />
            <DashboardCard
              title="Monitoring Module"
              subtitle="12 Metrics Active"
              status="healthy"
              icon="analytics-outline"
              onPress={() => {}}
            />
            <DashboardCard
              title="Optimization Module"
              subtitle="3 Tasks Running"
              status="warning"
              icon="trending-up-outline"
              onPress={() => {}}
            />
            <DashboardCard
              title="ML Module"
              subtitle="2 Models Training"
              status="healthy"
              icon="brain-outline"
              onPress={() => {}}
            />
          </View>
        </View>

        {/* Performance Metrics */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Performance Metrics</Text>
          <View style={styles.metricsGrid}>
            <View style={styles.metricItem}>
              <Text style={styles.metricValue}>98.5%</Text>
              <Text style={styles.metricLabel}>System Uptime</Text>
            </View>
            <View style={styles.metricItem}>
              <Text style={styles.metricValue}>2.3ms</Text>
              <Text style={styles.metricLabel}>Avg Response</Text>
            </View>
            <View style={styles.metricItem}>
              <Text style={styles.metricValue}>1.2K</Text>
              <Text style={styles.metricLabel}>Requests/min</Text>
            </View>
            <View style={styles.metricItem}>
              <Text style={styles.metricValue}>45%</Text>
              <Text style={styles.metricLabel}>CPU Usage</Text>
            </View>
          </View>
        </View>

        {/* Recent Activity */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Recent Activity</Text>
          <View style={styles.activityList}>
            <View style={styles.activityItem}>
              <View style={[styles.activityDot, { backgroundColor: theme.colors.success }]} />
              <View style={styles.activityContent}>
                <Text style={styles.activityTitle}>Cache optimization completed</Text>
                <Text style={styles.activityTime}>2 minutes ago</Text>
              </View>
            </View>
            <View style={styles.activityItem}>
              <View style={[styles.activityDot, { backgroundColor: theme.colors.info }]} />
              <View style={styles.activityContent}>
                <Text style={styles.activityTitle}>New ML model deployed</Text>
                <Text style={styles.activityTime}>15 minutes ago</Text>
              </View>
            </View>
            <View style={styles.activityItem}>
              <View style={[styles.activityDot, { backgroundColor: theme.colors.warning }]} />
              <View style={styles.activityContent}>
                <Text style={styles.activityTitle}>High memory usage detected</Text>
                <Text style={styles.activityTime}>1 hour ago</Text>
              </View>
            </View>
          </View>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

function createStyles(theme: any, width: number, height: number) {
  return StyleSheet.create({
    container: {
      flex: 1,
      backgroundColor: theme.colors.background,
    },
    scrollView: {
      flex: 1,
    },
    scrollContent: {
      paddingBottom: theme.spacing.xxl,
    },
    header: {
      padding: theme.spacing.lg,
      paddingBottom: theme.spacing.md,
    },
    title: {
      fontSize: theme.typography.h1.fontSize,
      fontWeight: theme.typography.h1.fontWeight,
      color: theme.colors.text,
      marginBottom: theme.spacing.xs,
    },
    subtitle: {
      fontSize: theme.typography.body.fontSize,
      color: theme.colors.textSecondary,
    },
    section: {
      paddingHorizontal: theme.spacing.lg,
      marginBottom: theme.spacing.xl,
    },
    sectionTitle: {
      fontSize: theme.typography.h4.fontSize,
      fontWeight: theme.typography.h4.fontWeight,
      color: theme.colors.text,
      marginBottom: theme.spacing.md,
    },
    cardsContainer: {
      gap: theme.spacing.md,
    },
    metricsGrid: {
      flexDirection: 'row',
      flexWrap: 'wrap',
      gap: theme.spacing.md,
    },
    metricItem: {
      flex: 1,
      minWidth: (width - theme.spacing.lg * 2 - theme.spacing.md) / 2,
      backgroundColor: theme.colors.surface,
      padding: theme.spacing.md,
      borderRadius: theme.borderRadius.md,
      alignItems: 'center',
    },
    metricValue: {
      fontSize: theme.typography.h3.fontSize,
      fontWeight: theme.typography.h3.fontWeight,
      color: theme.colors.primary,
      marginBottom: theme.spacing.xs,
    },
    metricLabel: {
      fontSize: theme.typography.caption.fontSize,
      color: theme.colors.textSecondary,
      textAlign: 'center',
    },
    activityList: {
      gap: theme.spacing.md,
    },
    activityItem: {
      flexDirection: 'row',
      alignItems: 'center',
      padding: theme.spacing.md,
      backgroundColor: theme.colors.surface,
      borderRadius: theme.borderRadius.md,
    },
    activityDot: {
      width: 8,
      height: 8,
      borderRadius: 4,
      marginRight: theme.spacing.md,
    },
    activityContent: {
      flex: 1,
    },
    activityTitle: {
      fontSize: theme.typography.body.fontSize,
      color: theme.colors.text,
      marginBottom: theme.spacing.xs,
    },
    activityTime: {
      fontSize: theme.typography.caption.fontSize,
      color: theme.colors.textSecondary,
    },
  });
}
import {
  View,
  Text,
  ScrollView,
  StyleSheet,
  useWindowDimensions,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useTheme } from '@/contexts/theme-context';
import { DashboardCard } from '@/components/dashboard/dashboard-card';
import { SystemStatusCard } from '@/components/dashboard/system-status-card';
import { MetricsOverview } from '@/components/dashboard/metrics-overview';
import { QuickActions } from '@/components/dashboard/quick-actions';

export default function DashboardScreen(): JSX.Element {
  const { theme } = useTheme();
  const { width, height } = useWindowDimensions();

  const styles = useMemo(() => createStyles(theme, width, height), [theme, width, height]);

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView
        style={styles.scrollView}
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={false}
      >
        {/* Header */}
        <View style={styles.header}>
          <Text style={styles.title}>Blaze AI Dashboard</Text>
          <Text style={styles.subtitle}>System Overview & Monitoring</Text>
        </View>

        {/* System Status */}
        <SystemStatusCard />

        {/* Quick Actions */}
        <QuickActions />

        {/* Metrics Overview */}
        <MetricsOverview />

        {/* Module Cards */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Active Modules</Text>
          <View style={styles.cardsContainer}>
            <DashboardCard
              title="Cache Module"
              subtitle="LRU Strategy Active"
              status="healthy"
              icon="flash-outline"
              onPress={() => {}}
            />
            <DashboardCard
              title="Monitoring Module"
              subtitle="12 Metrics Active"
              status="healthy"
              icon="analytics-outline"
              onPress={() => {}}
            />
            <DashboardCard
              title="Optimization Module"
              subtitle="3 Tasks Running"
              status="warning"
              icon="trending-up-outline"
              onPress={() => {}}
            />
            <DashboardCard
              title="ML Module"
              subtitle="2 Models Training"
              status="healthy"
              icon="brain-outline"
              onPress={() => {}}
            />
          </View>
        </View>

        {/* Performance Metrics */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Performance Metrics</Text>
          <View style={styles.metricsGrid}>
            <View style={styles.metricItem}>
              <Text style={styles.metricValue}>98.5%</Text>
              <Text style={styles.metricLabel}>System Uptime</Text>
            </View>
            <View style={styles.metricItem}>
              <Text style={styles.metricValue}>2.3ms</Text>
              <Text style={styles.metricLabel}>Avg Response</Text>
            </View>
            <View style={styles.metricItem}>
              <Text style={styles.metricValue}>1.2K</Text>
              <Text style={styles.metricLabel}>Requests/min</Text>
            </View>
            <View style={styles.metricItem}>
              <Text style={styles.metricValue}>45%</Text>
              <Text style={styles.metricLabel}>CPU Usage</Text>
            </View>
          </View>
        </View>

        {/* Recent Activity */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Recent Activity</Text>
          <View style={styles.activityList}>
            <View style={styles.activityItem}>
              <View style={[styles.activityDot, { backgroundColor: theme.colors.success }]} />
              <View style={styles.activityContent}>
                <Text style={styles.activityTitle}>Cache optimization completed</Text>
                <Text style={styles.activityTime}>2 minutes ago</Text>
              </View>
            </View>
            <View style={styles.activityItem}>
              <View style={[styles.activityDot, { backgroundColor: theme.colors.info }]} />
              <View style={styles.activityContent}>
                <Text style={styles.activityTitle}>New ML model deployed</Text>
                <Text style={styles.activityTime}>15 minutes ago</Text>
              </View>
            </View>
            <View style={styles.activityItem}>
              <View style={[styles.activityDot, { backgroundColor: theme.colors.warning }]} />
              <View style={styles.activityContent}>
                <Text style={styles.activityTitle}>High memory usage detected</Text>
                <Text style={styles.activityTime}>1 hour ago</Text>
              </View>
            </View>
          </View>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

function createStyles(theme: any, width: number, height: number) {
  return StyleSheet.create({
    container: {
      flex: 1,
      backgroundColor: theme.colors.background,
    },
    scrollView: {
      flex: 1,
    },
    scrollContent: {
      paddingBottom: theme.spacing.xxl,
    },
    header: {
      padding: theme.spacing.lg,
      paddingBottom: theme.spacing.md,
    },
    title: {
      fontSize: theme.typography.h1.fontSize,
      fontWeight: theme.typography.h1.fontWeight,
      color: theme.colors.text,
      marginBottom: theme.spacing.xs,
    },
    subtitle: {
      fontSize: theme.typography.body.fontSize,
      color: theme.colors.textSecondary,
    },
    section: {
      paddingHorizontal: theme.spacing.lg,
      marginBottom: theme.spacing.xl,
    },
    sectionTitle: {
      fontSize: theme.typography.h4.fontSize,
      fontWeight: theme.typography.h4.fontWeight,
      color: theme.colors.text,
      marginBottom: theme.spacing.md,
    },
    cardsContainer: {
      gap: theme.spacing.md,
    },
    metricsGrid: {
      flexDirection: 'row',
      flexWrap: 'wrap',
      gap: theme.spacing.md,
    },
    metricItem: {
      flex: 1,
      minWidth: (width - theme.spacing.lg * 2 - theme.spacing.md) / 2,
      backgroundColor: theme.colors.surface,
      padding: theme.spacing.md,
      borderRadius: theme.borderRadius.md,
      alignItems: 'center',
    },
    metricValue: {
      fontSize: theme.typography.h3.fontSize,
      fontWeight: theme.typography.h3.fontWeight,
      color: theme.colors.primary,
      marginBottom: theme.spacing.xs,
    },
    metricLabel: {
      fontSize: theme.typography.caption.fontSize,
      color: theme.colors.textSecondary,
      textAlign: 'center',
    },
    activityList: {
      gap: theme.spacing.md,
    },
    activityItem: {
      flexDirection: 'row',
      alignItems: 'center',
      padding: theme.spacing.md,
      backgroundColor: theme.colors.surface,
      borderRadius: theme.borderRadius.md,
    },
    activityDot: {
      width: 8,
      height: 8,
      borderRadius: 4,
      marginRight: theme.spacing.md,
    },
    activityContent: {
      flex: 1,
    },
    activityTitle: {
      fontSize: theme.typography.body.fontSize,
      color: theme.colors.text,
      marginBottom: theme.spacing.xs,
    },
    activityTime: {
      fontSize: theme.typography.caption.fontSize,
      color: theme.colors.textSecondary,
    },
  });
}


