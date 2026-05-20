import React, { useEffect, useState, useRef } from 'react';
import { View, Text, StyleSheet, Animated } from 'react-native';
import { useTheme } from '../contexts/ThemeContext';
import { ProgressBar } from './ProgressBar';
import { spacing, borderRadius, typography } from '../theme/colors';
import { formatDuration } from '../utils/format';

interface GenerationProgressProps {
  taskId: string;
  onComplete?: () => void;
  onError?: (error: Error) => void;
  pollInterval?: number;
}

export const GenerationProgress: React.FC<GenerationProgressProps> = ({
  taskId,
  onComplete,
  onError,
  pollInterval = 2000,
}) => {
  const { theme } = useTheme();
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState<string>('Iniciando...');
  const [elapsedTime, setElapsedTime] = useState(0);
  const fadeAnim = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    Animated.timing(fadeAnim, {
      toValue: 1,
      duration: 300,
      useNativeDriver: true,
    }).start();
  }, []);

  useEffect(() => {
    const interval = setInterval(() => {
      setElapsedTime((prev) => prev + pollInterval / 1000);
    }, pollInterval);

    return () => clearInterval(interval);
  }, [pollInterval]);

  useEffect(() => {
    if (!taskId) return;

    const pollStatus = async () => {
      try {
        const response = await fetch(`/api/v1/generate/task/${taskId}`);
        const data = await response.json();

        if (data.status === 'completed') {
          setProgress(100);
          setStatus('Completado');
          onComplete?.();
        } else if (data.status === 'failed') {
          setStatus('Fallido');
          onError?.(new Error(data.error || 'Error desconocido'));
        } else {
          setProgress(data.progress || 0);
          setStatus(data.status_message || 'Procesando...');
        }
      } catch (error) {
        console.error('Error polling status:', error);
      }
    };

    const interval = setInterval(pollStatus, pollInterval);
    pollStatus();

    return () => clearInterval(interval);
  }, [taskId, pollInterval, onComplete, onError]);

  return (
    <Animated.View
      style={[
        styles.container,
        {
          backgroundColor: theme.surface,
          borderColor: theme.border,
          opacity: fadeAnim,
        },
      ]}
    >
      <View style={styles.header}>
        <Text style={[styles.title, { color: theme.text }]}>Generando Proyecto</Text>
        <Text style={[styles.status, { color: theme.textSecondary }]}>{status}</Text>
      </View>

      <ProgressBar
        progress={progress}
        total={100}
        label={`${Math.round(progress)}%`}
        color={theme.primary}
      />

      <View style={styles.footer}>
        <Text style={[styles.time, { color: theme.textTertiary }]}>
          Tiempo transcurrido: {formatDuration(elapsedTime)}
        </Text>
      </View>
    </Animated.View>
  );
};

const styles = StyleSheet.create({
  container: {
    borderRadius: borderRadius.lg,
    padding: spacing.xl,
    borderWidth: 1,
    margin: spacing.lg,
  },
  header: {
    marginBottom: spacing.md,
  },
  title: {
    ...typography.h3,
    marginBottom: spacing.xs,
  },
  status: {
    ...typography.bodySmall,
  },
  footer: {
    marginTop: spacing.md,
    alignItems: 'center',
  },
  time: {
    ...typography.caption,
  },
});

