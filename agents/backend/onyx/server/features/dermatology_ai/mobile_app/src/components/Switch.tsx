import React from 'react';
import { TouchableOpacity, StyleSheet, View } from 'react-native';
import Animated, {
  useAnimatedStyle,
  useSharedValue,
  withSpring,
} from 'react-native-reanimated';
import { useTheme } from '../context/ThemeContext';

interface SwitchProps {
  value: boolean;
  onValueChange: (value: boolean) => void;
  disabled?: boolean;
  size?: 'small' | 'medium' | 'large';
}

const Switch: React.FC<SwitchProps> = ({
  value,
  onValueChange,
  disabled = false,
  size = 'medium',
}) => {
  const { colors } = useTheme();
  const translateX = useSharedValue(value ? 1 : 0);

  React.useEffect(() => {
    translateX.value = withSpring(value ? 1 : 0, {
      damping: 15,
      stiffness: 150,
    });
  }, [value, translateX]);

  const getSizeStyles = () => {
    switch (size) {
      case 'small':
        return { width: 40, height: 24, thumbSize: 18 };
      case 'large':
        return { width: 56, height: 32, thumbSize: 26 };
      default:
        return { width: 48, height: 28, thumbSize: 22 };
    }
  };

  const sizeStyles = getSizeStyles();

  const animatedThumbStyle = useAnimatedStyle(() => {
    const maxTranslate = sizeStyles.width - sizeStyles.thumbSize - 4;
    return {
      transform: [{ translateX: translateX.value * maxTranslate }],
    };
  });

  const animatedTrackStyle = useAnimatedStyle(() => {
    return {
      backgroundColor: translateX.value === 1 ? colors.primary : colors.border,
      opacity: disabled ? 0.5 : 1,
    };
  });

  return (
    <TouchableOpacity
      onPress={() => !disabled && onValueChange(!value)}
      disabled={disabled}
      activeOpacity={0.8}
    >
      <Animated.View
        style={[
          styles.track,
          {
            width: sizeStyles.width,
            height: sizeStyles.height,
            borderRadius: sizeStyles.height / 2,
          },
          animatedTrackStyle,
        ]}
      >
        <Animated.View
          style={[
            styles.thumb,
            {
              width: sizeStyles.thumbSize,
              height: sizeStyles.thumbSize,
              borderRadius: sizeStyles.thumbSize / 2,
            },
            animatedThumbStyle,
          ]}
        />
      </Animated.View>
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  track: {
    justifyContent: 'center',
    padding: 2,
  },
  thumb: {
    backgroundColor: '#fff',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.2,
    shadowRadius: 2,
    elevation: 2,
  },
});

export default Switch;

