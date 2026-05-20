import React from 'react';
import { TouchableOpacity, Text, StyleSheet, View } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import Animated, {
  useAnimatedStyle,
  useSharedValue,
  withSpring,
} from 'react-native-reanimated';
import { useTheme } from '../context/ThemeContext';

interface CheckboxProps {
  label?: string;
  checked: boolean;
  onToggle: (checked: boolean) => void;
  disabled?: boolean;
  size?: 'small' | 'medium' | 'large';
}

const Checkbox: React.FC<CheckboxProps> = ({
  label,
  checked,
  onToggle,
  disabled = false,
  size = 'medium',
}) => {
  const { colors } = useTheme();
  const scale = useSharedValue(checked ? 1 : 0);

  React.useEffect(() => {
    scale.value = withSpring(checked ? 1 : 0, {
      damping: 10,
      stiffness: 200,
    });
  }, [checked, scale]);

  const getSizeStyles = () => {
    switch (size) {
      case 'small':
        return { boxSize: 18, iconSize: 12 };
      case 'large':
        return { boxSize: 28, iconSize: 20 };
      default:
        return { boxSize: 24, iconSize: 16 };
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
      onPress={() => !disabled && onToggle(!checked)}
      disabled={disabled}
      activeOpacity={0.7}
    >
      <View
        style={[
          styles.box,
          {
            width: sizeStyles.boxSize,
            height: sizeStyles.boxSize,
            borderRadius: 4,
            borderColor: checked ? colors.primary : colors.border,
            backgroundColor: checked ? colors.primary : 'transparent',
            opacity: disabled ? 0.5 : 1,
          },
        ]}
      >
        <Animated.View style={animatedStyle}>
          <Ionicons
            name="checkmark"
            size={sizeStyles.iconSize}
            color="#fff"
          />
        </Animated.View>
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
  box: {
    borderWidth: 2,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 8,
  },
  label: {
    flex: 1,
  },
});

export default Checkbox;

