import React from 'react';
import { TouchableOpacity, Text, StyleSheet, ViewStyle } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useTheme } from '../context/ThemeContext';

interface ChipProps {
  label: string;
  onPress?: () => void;
  onClose?: () => void;
  selected?: boolean;
  variant?: 'default' | 'outline' | 'filled';
  size?: 'small' | 'medium' | 'large';
  icon?: keyof typeof Ionicons.glyphMap;
  style?: ViewStyle;
}

const Chip: React.FC<ChipProps> = ({
  label,
  onPress,
  onClose,
  selected = false,
  variant = 'default',
  size = 'medium',
  icon,
  style,
}) => {
  const { colors } = useTheme();

  const getSizeStyles = () => {
    switch (size) {
      case 'small':
        return { paddingVertical: 4, paddingHorizontal: 8, fontSize: 12 };
      case 'large':
        return { paddingVertical: 10, paddingHorizontal: 16, fontSize: 16 };
      default:
        return { paddingVertical: 6, paddingHorizontal: 12, fontSize: 14 };
    }
  };

  const sizeStyles = getSizeStyles();

  const getVariantStyles = () => {
    switch (variant) {
      case 'outline':
        return {
          backgroundColor: 'transparent',
          borderWidth: 1,
          borderColor: selected ? colors.primary : colors.border,
        };
      case 'filled':
        return {
          backgroundColor: selected ? colors.primary : colors.surface,
        };
      default:
        return {
          backgroundColor: selected ? `${colors.primary}20` : colors.surface,
        };
    }
  };

  const getTextColor = () => {
    if (variant === 'filled' && selected) return '#fff';
    return selected ? colors.primary : colors.text;
  };

  const Component = onPress || onClose ? TouchableOpacity : React.Fragment;
  const componentProps = onPress || onClose
    ? {
        onPress: onPress || onClose,
        activeOpacity: 0.7,
        style: [
          styles.chip,
          sizeStyles,
          getVariantStyles(),
          style,
        ],
      }
    : {
        style: [
          styles.chip,
          sizeStyles,
          getVariantStyles(),
          style,
        ],
      };

  return (
    <Component {...componentProps}>
      {icon && (
        <Ionicons
          name={icon}
          size={sizeStyles.fontSize}
          color={getTextColor()}
          style={styles.icon}
        />
      )}
      <Text
        style={[
          styles.label,
          { color: getTextColor(), fontSize: sizeStyles.fontSize },
        ]}
      >
        {label}
      </Text>
      {onClose && (
        <Ionicons
          name="close-circle"
          size={sizeStyles.fontSize}
          color={getTextColor()}
          style={styles.closeIcon}
        />
      )}
    </Component>
  );
};

const styles = StyleSheet.create({
  chip: {
    flexDirection: 'row',
    alignItems: 'center',
    borderRadius: 20,
    alignSelf: 'flex-start',
  },
  label: {
    fontWeight: '500',
  },
  icon: {
    marginRight: 4,
  },
  closeIcon: {
    marginLeft: 4,
  },
});

export default Chip;

