import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useTheme } from '@/contexts/theme-context';

interface MetricItemProps {
  label: string;
  value: string;
  change: number;
  icon: string;
  color: string;
}

export function MetricsOverview(): JSX.Element {
  const { theme } = useTheme();
  const styles = createStyles(theme);

  const metrics: MetricItemProps[] = [
    {
      label: 'Response Time',
      value: '2.3ms',
      change: -12.5,
      icon: 'speedometer-outline',
      color: theme.colors.success,
    },
    {
      label: 'Throughput',
      value: '1.2K req/s',
      change: 8.3,
      icon: 'trending-up-outline',
      color: theme.colors.info,
    },
    {
      label: 'Error Rate',
      value: '0.02%',
      change: -5.1,
      icon: 'alert-circle-outline',
      color: theme.colors.warning,
    },
    {
      label: 'Cache Hit',
      value: '94.7%',
      change: 2.1,
      icon: 'flash-outline',
      color: theme.colors.primary,
    },
  ];

  return (
    <View style={styles.container}>
      <Text style={styles.sectionTitle}>Performance Metrics</Text>
      <View style={styles.metricsGrid}>
        {metrics.map((metric, index) => (
          <MetricItem
            key={index}
            label={metric.label}
            value={metric.value}
            change={metric.change}
            icon={metric.icon}
            color={metric.color}
          />
        ))}
      </View>
    </View>
  );
}

function MetricItem({ label, value, change, icon, color }: MetricItemProps): JSX.Element {
  const { theme } = useTheme();
  const styles = createMetricStyles(theme, color);

  const isPositive = change >= 0;
  const changeText = `${isPositive ? '+' : ''}${change.toFixed(1)}%`;

  return (
    <View style={styles.metricContainer}>
      <View style={styles.iconContainer}>
        <Ionicons name={icon as any} size={20} color={color} />
      </View>
      
      <View style={styles.content}>
        <Text style={styles.label}>{label}</Text>
        <Text style={styles.value}>{value}</Text>
        <View style={styles.changeContainer}>
          <Ionicons
            name={isPositive ? 'trending-up' : 'trending-down'}
            size={16}
            color={isPositive ? theme.colors.success : theme.colors.error}
          />
          <Text style={[styles.changeText, { color: isPositive ? theme.colors.success : theme.colors.error }]}>
            {changeText}
          </Text>
        </View>
      </View>
    </View>
  );
}

function createStyles(theme: any) {
  return StyleSheet.create({
    container: {
      paddingHorizontal: theme.spacing.lg,
      marginBottom: theme.spacing.xl,
    },
    sectionTitle: {
      fontSize: theme.typography.h4.fontSize,
      fontWeight: theme.typography.h4.fontWeight,
      color: theme.colors.text,
      marginBottom: theme.spacing.md,
    },
    metricsGrid: {
      gap: theme.spacing.md,
    },
  });
}

function createMetricStyles(theme: any, color: string) {
  return StyleSheet.create({
    metricContainer: {
      flexDirection: 'row',
      alignItems: 'center',
      backgroundColor: theme.colors.surface,
      padding: theme.spacing.md,
      borderRadius: theme.borderRadius.md,
      shadowColor: theme.colors.text,
      shadowOffset: {
        width: 0,
        height: 2,
      },
      shadowOpacity: 0.1,
      shadowRadius: 4,
      elevation: 3,
    },
    iconContainer: {
      width: 40,
      height: 40,
      borderRadius: 20,
      backgroundColor: `${color}15`,
      justifyContent: 'center',
      alignItems: 'center',
      marginRight: theme.spacing.md,
    },
    content: {
      flex: 1,
    },
    label: {
      fontSize: theme.typography.caption.fontSize,
      color: theme.colors.textSecondary,
      marginBottom: theme.spacing.xs,
    },
    value: {
      fontSize: theme.typography.h5.fontSize,
      fontWeight: theme.typography.h5.fontWeight,
      color: theme.colors.text,
      marginBottom: theme.spacing.xs,
    },
    changeContainer: {
      flexDirection: 'row',
      alignItems: 'center',
    },
    changeText: {
      fontSize: theme.typography.caption.fontSize,
      fontWeight: '600',
      marginLeft: theme.spacing.xs,
    },
  });
}
import { View, Text, StyleSheet } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useTheme } from '@/contexts/theme-context';

interface MetricItemProps {
  label: string;
  value: string;
  change: number;
  icon: string;
  color: string;
}

export function MetricsOverview(): JSX.Element {
  const { theme } = useTheme();
  const styles = createStyles(theme);

  const metrics: MetricItemProps[] = [
    {
      label: 'Response Time',
      value: '2.3ms',
      change: -12.5,
      icon: 'speedometer-outline',
      color: theme.colors.success,
    },
    {
      label: 'Throughput',
      value: '1.2K req/s',
      change: 8.3,
      icon: 'trending-up-outline',
      color: theme.colors.info,
    },
    {
      label: 'Error Rate',
      value: '0.02%',
      change: -5.1,
      icon: 'alert-circle-outline',
      color: theme.colors.warning,
    },
    {
      label: 'Cache Hit',
      value: '94.7%',
      change: 2.1,
      icon: 'flash-outline',
      color: theme.colors.primary,
    },
  ];

  return (
    <View style={styles.container}>
      <Text style={styles.sectionTitle}>Performance Metrics</Text>
      <View style={styles.metricsGrid}>
        {metrics.map((metric, index) => (
          <MetricItem
            key={index}
            label={metric.label}
            value={metric.value}
            change={metric.change}
            icon={metric.icon}
            color={metric.color}
          />
        ))}
      </View>
    </View>
  );
}

function MetricItem({ label, value, change, icon, color }: MetricItemProps): JSX.Element {
  const { theme } = useTheme();
  const styles = createMetricStyles(theme, color);

  const isPositive = change >= 0;
  const changeText = `${isPositive ? '+' : ''}${change.toFixed(1)}%`;

  return (
    <View style={styles.metricContainer}>
      <View style={styles.iconContainer}>
        <Ionicons name={icon as any} size={20} color={color} />
      </View>
      
      <View style={styles.content}>
        <Text style={styles.label}>{label}</Text>
        <Text style={styles.value}>{value}</Text>
        <View style={styles.changeContainer}>
          <Ionicons
            name={isPositive ? 'trending-up' : 'trending-down'}
            size={16}
            color={isPositive ? theme.colors.success : theme.colors.error}
          />
          <Text style={[styles.changeText, { color: isPositive ? theme.colors.success : theme.colors.error }]}>
            {changeText}
          </Text>
        </View>
      </View>
    </View>
  );
}

function createStyles(theme: any) {
  return StyleSheet.create({
    container: {
      paddingHorizontal: theme.spacing.lg,
      marginBottom: theme.spacing.xl,
    },
    sectionTitle: {
      fontSize: theme.typography.h4.fontSize,
      fontWeight: theme.typography.h4.fontWeight,
      color: theme.colors.text,
      marginBottom: theme.spacing.md,
    },
    metricsGrid: {
      gap: theme.spacing.md,
    },
  });
}

function createMetricStyles(theme: any, color: string) {
  return StyleSheet.create({
    metricContainer: {
      flexDirection: 'row',
      alignItems: 'center',
      backgroundColor: theme.colors.surface,
      padding: theme.spacing.md,
      borderRadius: theme.borderRadius.md,
      shadowColor: theme.colors.text,
      shadowOffset: {
        width: 0,
        height: 2,
      },
      shadowOpacity: 0.1,
      shadowRadius: 4,
      elevation: 3,
    },
    iconContainer: {
      width: 40,
      height: 40,
      borderRadius: 20,
      backgroundColor: `${color}15`,
      justifyContent: 'center',
      alignItems: 'center',
      marginRight: theme.spacing.md,
    },
    content: {
      flex: 1,
    },
    label: {
      fontSize: theme.typography.caption.fontSize,
      color: theme.colors.textSecondary,
      marginBottom: theme.spacing.xs,
    },
    value: {
      fontSize: theme.typography.h5.fontSize,
      fontWeight: theme.typography.h5.fontWeight,
      color: theme.colors.text,
      marginBottom: theme.spacing.xs,
    },
    changeContainer: {
      flexDirection: 'row',
      alignItems: 'center',
    },
    changeText: {
      fontSize: theme.typography.caption.fontSize,
      fontWeight: '600',
      marginLeft: theme.spacing.xs,
    },
  });
}


