import { useCallback } from 'react';
import * as Haptics from 'expo-haptics';
import { Platform } from 'react-native';

export type VibrationType = 'light' | 'medium' | 'heavy' | 'success' | 'warning' | 'error';

export function useVibration() {
  const vibrate = useCallback(
    (type: VibrationType = 'medium') => {
      if (Platform.OS === 'ios') {
        switch (type) {
          case 'light':
            Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
            break;
          case 'medium':
            Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
            break;
          case 'heavy':
            Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Heavy);
            break;
          case 'success':
            Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
            break;
          case 'warning':
            Haptics.notificationAsync(Haptics.NotificationFeedbackType.Warning);
            break;
          case 'error':
            Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error);
            break;
        }
      } else if (Platform.OS === 'android') {
        // Android vibration would use Vibration API
        // For now, we'll use Haptics which works on both
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
      }
    },
    []
  );

  return { vibrate };
}


