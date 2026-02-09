import React from 'react';
import { View, StyleSheet, Modal } from 'react-native';
import Animated, {
  useAnimatedStyle,
  useSharedValue,
  withTiming,
} from 'react-native-reanimated';
import { COLORS, SPACING, BORDER_RADIUS } from '../../constants/config';

interface OverlayProps {
  visible: boolean;
  children: React.ReactNode;
  position?: 'top' | 'bottom' | 'center';
  onClose?: () => void;
}

/**
 * Overlay component
 * Animated overlay with content
 */
export function Overlay({
  visible,
  children,
  position = 'center',
  onClose,
}: OverlayProps) {
  const opacity = useSharedValue(0);
  const translateY = useSharedValue(position === 'bottom' ? 100 : position === 'top' ? -100 : 0);

  React.useEffect(() => {
    opacity.value = withTiming(visible ? 1 : 0, { duration: 300 });
    translateY.value = withTiming(
      visible
        ? 0
        : position === 'bottom'
        ? 100
        : position === 'top'
        ? -100
        : 0,
      { duration: 300 }
    );
  }, [visible, position, opacity, translateY]);

  const animatedStyle = useAnimatedStyle(() => ({
    opacity: opacity.value,
    transform: [{ translateY: translateY.value }],
  }));

  if (!visible) return null;

  return (
    <Modal
      visible={visible}
      transparent
      animationType="none"
      onRequestClose={onClose}
    >
      <View style={styles.container}>
        <Animated.View
          style={[
            styles.overlay,
            position === 'top' && styles.top,
            position === 'bottom' && styles.bottom,
            position === 'center' && styles.center,
            animatedStyle,
          ]}
        >
          {children}
        </Animated.View>
      </View>
    </Modal>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
  },
  overlay: {
    backgroundColor: COLORS.surface,
    borderRadius: BORDER_RADIUS.lg,
    padding: SPACING.lg,
    maxWidth: '90%',
    maxHeight: '80%',
  },
  top: {
    position: 'absolute',
    top: SPACING.xl,
    left: SPACING.lg,
    right: SPACING.lg,
  },
  bottom: {
    position: 'absolute',
    bottom: SPACING.xl,
    left: SPACING.lg,
    right: SPACING.lg,
  },
  center: {
    // Center is default
  },
});

