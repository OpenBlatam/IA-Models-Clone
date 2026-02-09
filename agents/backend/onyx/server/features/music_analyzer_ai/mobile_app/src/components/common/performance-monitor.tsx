import React, { useEffect, useState } from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { usePerformance } from '../../hooks/use-performance';
import { COLORS, SPACING, TYPOGRAPHY } from '../../constants/config';

interface PerformanceMonitorProps {
  enabled?: boolean;
  showMetrics?: boolean;
  threshold?: {
    renderTime?: number;
    interactionTime?: number;
  };
  onThresholdExceeded?: (metric: string, value: number) => void;
}

export const PerformanceMonitor: React.FC<PerformanceMonitorProps> = ({
  enabled = __DEV__,
  showMetrics = false,
  threshold = {
    renderTime: 100,
    interactionTime: 300,
  },
  onThresholdExceeded,
}) => {
  const { metrics, measureAsync, measureSync } = usePerformance({
    trackRenders: enabled,
    trackInteractions: enabled,
    onMetricsUpdate: (updatedMetrics) => {
      if (threshold.renderTime && updatedMetrics.renderTime > threshold.renderTime) {
        onThresholdExceeded?.('renderTime', updatedMetrics.renderTime);
      }
      if (
        threshold.interactionTime &&
        updatedMetrics.interactionTime > threshold.interactionTime
      ) {
        onThresholdExceeded?.('interactionTime', updatedMetrics.interactionTime);
      }
    },
  });

  if (!enabled || !showMetrics) {
    return null;
  }

  return (
    <View style={styles.container}>
      <Text style={styles.label}>Performance Metrics</Text>
      <Text style={styles.metric}>
        Render: {metrics.renderTime.toFixed(2)}ms
      </Text>
      <Text style={styles.metric}>
        Interaction: {metrics.interactionTime.toFixed(2)}ms
      </Text>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    position: 'absolute',
    top: 100,
    right: SPACING.md,
    backgroundColor: COLORS.surface,
    padding: SPACING.sm,
    borderRadius: 8,
    zIndex: 9999,
  },
  label: {
    ...TYPOGRAPHY.caption,
    color: COLORS.text,
    fontWeight: '600',
    marginBottom: SPACING.xs,
  },
  metric: {
    ...TYPOGRAPHY.caption,
    color: COLORS.textSecondary,
  },
});
