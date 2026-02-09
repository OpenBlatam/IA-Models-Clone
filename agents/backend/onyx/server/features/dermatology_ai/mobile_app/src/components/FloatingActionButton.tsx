import React from 'react';
import { TouchableOpacity, StyleSheet, ViewStyle } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import Animated, {
  useAnimatedStyle,
  useSharedValue,
  withSpring,
  withSequence,
} from 'react-native-reanimated';
import { useTheme } from '../context/ThemeContext';
import { useHapticFeedback } from '../hooks/useHapticFeedback';

interface FloatingActionButtonProps {
  onPress: () => void;
  icon?: keyof typeof Ionicons.glyphMap;
  size?: number;
  position?: 'bottom-right' | 'bottom-left' | 'top-right' | 'top-left';
  color?: string;
  style?: ViewStyle;
}

const FloatingActionButton: React.FC<FloatingActionButtonProps> = ({
  onPress,
  icon = 'add',
  size = 56,
  position = 'bottom-right',
  color,
  style,
}) => {
  const { colors } = useTheme();
  const { trigger } = useHapticFeedback();
  const scale = useSharedValue(1);

  const buttonColor = color || colors.primary;

  const getPositionStyles = (): ViewStyle => {
    switch (position) {
      case 'bottom-right':
        return { bottom: 24, right: 24 };
      case 'bottom-left':
        return { bottom: 24, left: 24 };
      case 'top-right':
        return { top: 24, right: 24 };
      case 'top-left':
        return { top: 24, left: 24 };
      default:
        return { bottom: 24, right: 24 };
    }
  };

  const animatedStyle = useAnimatedStyle(() => {
    return {
      transform: [{ scale: scale.value }],
    };
  });

  const handlePress = () => {
    trigger('light');
    scale.value = withSequence(
      withSpring(0.9, { damping: 10 }),
      withSpring(1, { damping: 10 })
    );
    onPress();
  };

  return (
    <Animated.View
      style={[
        styles.container,
        getPositionStyles(),
        { width: size, height: size, borderRadius: size / 2 },
        style,
        animatedStyle,
      ]}
    >
      <TouchableOpacity
        onPress={handlePress}
        style={[
          styles.button,
          {
            width: size,
            height: size,
            borderRadius: size / 2,
            backgroundColor: buttonColor,
          },
        ]}
        activeOpacity={0.8}
      >
        <Ionicons name={icon} size={size * 0.4} color="#fff" />
      </TouchableOpacity>
    </Animated.View>
  );
};

const styles = StyleSheet.create({
  container: {
    position: 'absolute',
    zIndex: 1000,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 8,
  },
  button: {
    justifyContent: 'center',
    alignItems: 'center',
  },
});

export default FloatingActionButton;
