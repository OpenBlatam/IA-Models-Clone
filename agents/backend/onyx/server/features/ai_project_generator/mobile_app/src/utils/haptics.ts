import * as Haptics from 'expo-haptics';

export const hapticFeedback = {
  light: () => {
    try {
      Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    } catch (error) {
      console.warn('Haptics not available:', error);
    }
  },
  
  medium: () => {
    try {
      Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
    } catch (error) {
      console.warn('Haptics not available:', error);
    }
  },
  
  heavy: () => {
    try {
      Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Heavy);
    } catch (error) {
      console.warn('Haptics not available:', error);
    }
  },
  
  success: () => {
    try {
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
    } catch (error) {
      console.warn('Haptics not available:', error);
    }
  },
  
  warning: () => {
    try {
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Warning);
    } catch (error) {
      console.warn('Haptics not available:', error);
    }
  },
  
  error: () => {
    try {
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error);
    } catch (error) {
      console.warn('Haptics not available:', error);
    }
  },
  
  selection: () => {
    try {
      Haptics.selectionAsync();
    } catch (error) {
      console.warn('Haptics not available:', error);
    }
  },
};

