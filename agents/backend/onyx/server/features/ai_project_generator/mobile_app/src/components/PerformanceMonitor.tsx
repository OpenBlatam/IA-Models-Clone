import React, { useEffect, useState } from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { useTheme } from '../contexts/ThemeContext';
import { spacing, typography } from '../theme/colors';

interface PerformanceMetrics {
  renderTime: number;
  componentName: string;
}

export const PerformanceMonitor: React.FC<{
  componentName: string;
  enabled?: boolean;
}> = ({ componentName, enabled = __DEV__ }) => {
  const { theme } = useTheme();
  const [metrics, setMetrics] = useState<PerformanceMetrics | null>(null);
  const startTime = React.useRef<number>(Date.now());

  useEffect(() => {
    if (!enabled) return;

    const renderTime = Date.now() - startTime.current;
    setMetrics({ renderTime, componentName });

    return () => {
      startTime.current = Date.now();
    };
  }, [componentName, enabled]);

  if (!enabled || !metrics) return null;

  return (
    <View style={[styles.container, { backgroundColor: theme.surfaceVariant }]}>
      <Text style={[styles.text, { color: theme.textSecondary }]}>
        {componentName}: {metrics.renderTime}ms
      </Text>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    padding: spacing.xs,
    borderRadius: 4,
    marginBottom: spacing.xs,
  },
  text: {
    ...typography.caption,
    fontFamily: 'monospace',
  },
});

export const usePerformanceMonitor = (componentName: string, enabled = __DEV__) => {
  const startTime = React.useRef<number>(Date.now());
  const [renderTime, setRenderTime] = useState<number>(0);

  useEffect(() => {
    if (!enabled) return;

    const time = Date.now() - startTime.current;
    setRenderTime(time);

    return () => {
      startTime.current = Date.now();
    };
  }, [componentName, enabled]);

  return { renderTime };
};

