import React from 'react';
import { TouchableOpacity, Text, StyleSheet, View } from 'react-native';
import Animated, {
  useAnimatedStyle,
  useSharedValue,
  withSpring,
} from 'react-native-reanimated';
import { useTheme } from '../context/ThemeContext';

interface RadioButtonProps {
  label?: string;
  selected: boolean;
  onSelect: () => void;
  disabled?: boolean;
  size?: 'small' | 'medium' | 'large';
}

const RadioButton: React.FC<RadioButtonProps> = ({
  label,
  selected,
  onSelect,
  disabled = false,
  size = 'medium',
}) => {
  const { colors } = useTheme();
  const scale = useSharedValue(selected ? 1 : 0);

  React.useEffect(() => {
    scale.value = withSpring(selected ? 1 : 0, {
      damping: 10,
      stiffness: 200,
    });
  }, [selected, scale]);

  const getSizeStyles = () => {
    switch (size) {
      case 'small':
        return { outerSize: 18, innerSize: 8 };
      case 'large':
        return { outerSize: 28, innerSize: 16 };
      default:
        return { outerSize: 24, innerSize: 12 };
    }
  };

  const sizeStyles = getSizeStyles();

  const animatedStyle = useAnimatedStyle(() => {
    return {
      transform: [{ scale: scale.value }],
    };
  });

  return (
    <TouchableOpacity
      style={styles.container}
      onPress={() => !disabled && onSelect()}
      disabled={disabled}
      activeOpacity={0.7}
    >
      <View
        style={[
          styles.outer,
          {
            width: sizeStyles.outerSize,
            height: sizeStyles.outerSize,
            borderRadius: sizeStyles.outerSize / 2,
            borderColor: selected ? colors.primary : colors.border,
            opacity: disabled ? 0.5 : 1,
          },
        ]}
      >
        <Animated.View
          style={[
            styles.inner,
            {
              width: sizeStyles.innerSize,
              height: sizeStyles.innerSize,
              borderRadius: sizeStyles.innerSize / 2,
              backgroundColor: colors.primary,
            },
            animatedStyle,
          ]}
        />
      </View>
      {label && (
        <Text
          style={[
            styles.label,
            {
              color: disabled ? colors.textSecondary : colors.text,
              fontSize: size === 'small' ? 12 : size === 'large' ? 16 : 14,
            },
          ]}
        >
          {label}
        </Text>
      )}
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'center',
    marginVertical: 4,
  },
  outer: {
    borderWidth: 2,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 8,
  },
  inner: {
    position: 'absolute',
  },
  label: {
    flex: 1,
  },
});

export default RadioButton;

