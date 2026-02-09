import React, { memo, useCallback } from 'react';
import { TouchableOpacity, ViewStyle, StyleSheet } from 'react-native';
import { useHapticFeedback } from '../../hooks/use-haptic-feedback';

interface ListItemWrapperProps {
  children: React.ReactNode;
  onPress?: () => void;
  onLongPress?: () => void;
  style?: ViewStyle;
  disabled?: boolean;
  activeOpacity?: number;
}

function ListItemWrapperComponent({
  children,
  onPress,
  onLongPress,
  style,
  disabled = false,
  activeOpacity = 0.7,
}: ListItemWrapperProps) {
  const haptics = useHapticFeedback();

  const handlePress = useCallback(() => {
    if (onPress && !disabled) {
      haptics.light();
      onPress();
    }
  }, [onPress, disabled, haptics]);

  const handleLongPress = useCallback(() => {
    if (onLongPress && !disabled) {
      haptics.medium();
      onLongPress();
    }
  }, [onLongPress, disabled, haptics]);

  if (!onPress && !onLongPress) {
    return <>{children}</>;
  }

  return (
    <TouchableOpacity
      onPress={handlePress}
      onLongPress={handleLongPress}
      disabled={disabled}
      activeOpacity={activeOpacity}
      style={[styles.container, style]}
    >
      {children}
    </TouchableOpacity>
  );
}

export const ListItemWrapper = memo(ListItemWrapperComponent);

const styles = StyleSheet.create({
  container: {
    // Container styles can be customized via style prop
  },
});

