import React, { useState, useEffect } from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useTheme } from '../context/ThemeContext';
import { useInterval } from '../hooks/useInterval';
import { formatDuration } from '../utils/numberUtils';

interface TimerProps {
  initialSeconds?: number;
  onComplete?: () => void;
  autoStart?: boolean;
  showControls?: boolean;
}

const Timer: React.FC<TimerProps> = ({
  initialSeconds = 0,
  onComplete,
  autoStart = false,
  showControls = true,
}) => {
  const { colors } = useTheme();
  const [seconds, setSeconds] = useState(initialSeconds);
  const [isRunning, setIsRunning] = useState(autoStart);
  const [isPaused, setIsPaused] = useState(false);

  useInterval(
    () => {
      if (seconds > 0) {
        setSeconds((prev) => prev - 1);
      } else {
        setIsRunning(false);
        onComplete?.();
      }
    },
    isRunning && !isPaused ? 1000 : null
  );

  const start = () => {
    setIsRunning(true);
    setIsPaused(false);
  };

  const pause = () => {
    setIsPaused(true);
  };

  const resume = () => {
    setIsPaused(false);
  };

  const reset = () => {
    setSeconds(initialSeconds);
    setIsRunning(false);
    setIsPaused(false);
  };

  const stop = () => {
    setIsRunning(false);
    setIsPaused(false);
  };

  return (
    <View style={styles.container}>
      <Text style={[styles.time, { color: colors.text }]}>
        {formatDuration(seconds)}
      </Text>
      {showControls && (
        <View style={styles.controls}>
          {!isRunning ? (
            <TouchableOpacity
              style={[styles.button, { backgroundColor: colors.primary }]}
              onPress={start}
            >
              <Ionicons name="play" size={24} color="#fff" />
            </TouchableOpacity>
          ) : isPaused ? (
            <TouchableOpacity
              style={[styles.button, { backgroundColor: colors.success }]}
              onPress={resume}
            >
              <Ionicons name="play" size={24} color="#fff" />
            </TouchableOpacity>
          ) : (
            <TouchableOpacity
              style={[styles.button, { backgroundColor: colors.warning }]}
              onPress={pause}
            >
              <Ionicons name="pause" size={24} color="#fff" />
            </TouchableOpacity>
          )}
          <TouchableOpacity
            style={[styles.button, { backgroundColor: colors.error }]}
            onPress={stop}
          >
            <Ionicons name="stop" size={24} color="#fff" />
          </TouchableOpacity>
          <TouchableOpacity
            style={[styles.button, { backgroundColor: colors.border }]}
            onPress={reset}
          >
            <Ionicons name="refresh" size={24} color={colors.text} />
          </TouchableOpacity>
        </View>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    alignItems: 'center',
    padding: 16,
  },
  time: {
    fontSize: 48,
    fontWeight: 'bold',
    marginBottom: 16,
  },
  controls: {
    flexDirection: 'row',
    gap: 12,
  },
  button: {
    width: 56,
    height: 56,
    borderRadius: 28,
    justifyContent: 'center',
    alignItems: 'center',
  },
});

export default Timer;

