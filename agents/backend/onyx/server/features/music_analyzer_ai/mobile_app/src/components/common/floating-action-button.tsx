import React from 'react';
import { TouchableOpacity, StyleSheet, View } from 'react-native';
import Animated, {
  useAnimatedStyle,
  useSharedValue,
  withSpring,
} from 'react-native-reanimated';
import { COLORS, SPACING, BORDER_RADIUS } from '../../constants/config';
import { useHapticFeedback } from '../../hooks/use-haptic-feedback';

interface FloatingActionButtonProps {
  onPress: () => void;
  icon: React.ReactNode;
  size?: number;
  position?: 'bottom-right' | 'bottom-left' | 'top-right' | 'top-left';
  accessibilityLabel: string;
}

const AnimatedTouchable = Animated.createAnimatedComponent(TouchableOpacity);

/**
 * Floating Action Button (FAB)
 * Follows Material Design guidelines
 */
export function FloatingActionButton({
  onPress,
  icon,
  size = 56,
  position = 'bottom-right',
  accessibilityLabel,
}: FloatingActionButtonProps) {
  const haptics = useHapticFeedback();
  const scale = useSharedValue(1);

  const handlePressIn = () => {
    haptics.selection();
    scale.value = withSpring(0.9);
  };

  const handlePressOut = () => {
    scale.value = withSpring(1);
  };

  const handlePress = () => {
    haptics.medium();
    onPress();
  };

  const animatedStyle = useAnimatedStyle(() => ({
    transform: [{ scale: scale.value }],
  }));

  const positionStyle = {
    'bottom-right': { bottom: SPACING.xl, right: SPACING.xl },
    'bottom-left': { bottom: SPACING.xl, left: SPACING.xl },
    'top-right': { top: SPACING.xl, right: SPACING.xl },
    'top-left': { top: SPACING.xl, left: SPACING.xl },
  };

  return (
    <AnimatedTouchable
      style={[
        styles.container,
        {
          width: size,
          height: size,
          borderRadius: size / 2,
        },
        positionStyle[position],
        animatedStyle,
      ]}
      onPress={handlePress}
      onPressIn={handlePressIn}
      onPressOut={handlePressOut}
      activeOpacity={0.8}
      accessibilityLabel={accessibilityLabel}
      accessibilityRole="button"
    >
      <View style={styles.iconContainer}>{icon}</View>
    </AnimatedTouchable>
  );
}

const styles = StyleSheet.create({
  container: {
    position: 'absolute',
    backgroundColor: COLORS.primary,
    justifyContent: 'center',
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 8,
    zIndex: 1000,
  },
  iconContainer: {
    justifyContent: 'center',
    alignItems: 'center',
  },
});

