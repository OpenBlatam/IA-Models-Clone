import { TouchableOpacity, Text, StyleSheet, ViewStyle, TextStyle, ActivityIndicator } from 'react-native';
import { Ionicons } from '@expo/vector-icons';

interface AccessibleButtonProps {
  title: string;
  onPress: () => void;
  variant?: 'primary' | 'secondary' | 'danger' | 'outline';
  size?: 'small' | 'medium' | 'large';
  loading?: boolean;
  disabled?: boolean;
  icon?: keyof typeof Ionicons.glyphMap;
  iconPosition?: 'left' | 'right';
  accessibilityLabel?: string;
  accessibilityHint?: string;
  accessibilityRole?: 'button' | 'link' | 'none';
  style?: ViewStyle;
}

export function AccessibleButton({
  title,
  onPress,
  variant = 'primary',
  size = 'medium',
  loading = false,
  disabled = false,
  icon,
  iconPosition = 'left',
  accessibilityLabel,
  accessibilityHint,
  accessibilityRole = 'button',
  style,
}: AccessibleButtonProps) {
  const isDisabled = disabled || loading;

  const getVariantStyles = (): { container: ViewStyle; text: TextStyle } => {
    switch (variant) {
      case 'primary':
        return {
          container: { backgroundColor: isDisabled ? '#9ca3af' : '#0ea5e9' },
          text: { color: '#fff' },
        };
      case 'secondary':
        return {
          container: { backgroundColor: isDisabled ? '#e5e7eb' : '#f3f4f6' },
          text: { color: isDisabled ? '#9ca3af' : '#1f2937' },
        };
      case 'danger':
        return {
          container: { backgroundColor: isDisabled ? '#9ca3af' : '#ef4444' },
          text: { color: '#fff' },
        };
      case 'outline':
        return {
          container: {
            backgroundColor: 'transparent',
            borderWidth: 1,
            borderColor: isDisabled ? '#d1d5db' : '#0ea5e9',
          },
          text: { color: isDisabled ? '#9ca3af' : '#0ea5e9' },
        };
      default:
        return {
          container: { backgroundColor: '#0ea5e9' },
          text: { color: '#fff' },
        };
    }
  };

  const getSizeStyles = (): { container: ViewStyle; text: TextStyle } => {
    switch (size) {
      case 'small':
        return {
          container: { paddingVertical: 8, paddingHorizontal: 16, minHeight: 36 },
          text: { fontSize: 14 },
        };
      case 'large':
        return {
          container: { paddingVertical: 16, paddingHorizontal: 24, minHeight: 56 },
          text: { fontSize: 18 },
        };
      default:
        return {
          container: { paddingVertical: 12, paddingHorizontal: 20, minHeight: 44 },
          text: { fontSize: 16 },
        };
    }
  };

  const variantStyles = getVariantStyles();
  const sizeStyles = getSizeStyles();

  return (
    <TouchableOpacity
      style={[
        styles.button,
        variantStyles.container,
        sizeStyles.container,
        style,
        isDisabled && styles.disabled,
      ]}
      onPress={onPress}
      disabled={isDisabled}
      activeOpacity={0.7}
      accessibilityRole={accessibilityRole}
      accessibilityLabel={accessibilityLabel || title}
      accessibilityHint={accessibilityHint}
      accessibilityState={{ disabled: isDisabled }}
    >
      {loading ? (
        <ActivityIndicator size="small" color={variantStyles.text.color} />
      ) : (
        <>
          {icon && iconPosition === 'left' && (
            <Ionicons name={icon} size={20} color={variantStyles.text.color} style={styles.iconLeft} />
          )}
          <Text style={[styles.text, variantStyles.text, sizeStyles.text]}>{title}</Text>
          {icon && iconPosition === 'right' && (
            <Ionicons name={icon} size={20} color={variantStyles.text.color} style={styles.iconRight} />
          )}
        </>
      )}
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  button: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: 8,
  },
  text: {
    fontWeight: '600',
  },
  iconLeft: {
    marginRight: 8,
  },
  iconRight: {
    marginLeft: 8,
  },
  disabled: {
    opacity: 0.6,
  },
});


