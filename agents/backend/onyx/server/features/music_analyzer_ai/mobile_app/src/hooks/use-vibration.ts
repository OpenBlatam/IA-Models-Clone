import { useCallback } from 'react';
import * as Haptics from 'expo-haptics';
import { Platform } from 'react-native';

/**
 * Hook for vibration/haptic feedback
 * Provides different vibration patterns
 */
export function useVibration() {
  const vibrate = useCallback(async (pattern: 'light' | 'medium' | 'heavy' | 'success' | 'warning' | 'error' = 'medium') => {
    if (Platform.OS === 'ios') {
      switch (pattern) {
        case 'light':
          await Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
          break;
        case 'medium':
          await Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
          break;
        case 'heavy':
          await Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Heavy);
          break;
        case 'success':
          await Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
          break;
        case 'warning':
          await Haptics.notificationAsync(Haptics.NotificationFeedbackType.Warning);
          break;
        case 'error':
          await Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error);
          break;
      }
    } else {
      // Android vibration
      const { Vibration } = await import('react-native');
      const patterns: Record<string, number[]> = {
        light: [10],
        medium: [50],
        heavy: [100],
        success: [50, 50, 50],
        warning: [100, 50, 100],
        error: [200, 100, 200],
      };

      Vibration.vibrate(patterns[pattern] || patterns.medium);
    }
  }, []);

  const vibratePattern = useCallback(async (pattern: number[]) => {
    if (Platform.OS === 'android') {
      const { Vibration } = await import('react-native');
      Vibration.vibrate(pattern);
    } else {
      // iOS doesn't support custom patterns, use medium instead
      await Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
    }
  }, []);

  return { vibrate, vibratePattern };
}

