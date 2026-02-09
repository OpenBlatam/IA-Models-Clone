import { ReactNode } from 'react';
import { View, Text, StyleSheet, ViewStyle } from 'react-native';
import { Ionicons } from '@expo/vector-icons';

interface AlertProps {
  variant?: 'info' | 'success' | 'warning' | 'error';
  title?: string;
  children: ReactNode;
  icon?: keyof typeof Ionicons.glyphMap;
  style?: ViewStyle;
}

export function Alert({ variant = 'info', title, children, icon, style }: AlertProps) {
  const getVariantStyles = () => {
    switch (variant) {
      case 'success':
        return {
          backgroundColor: '#d1fae5',
          borderColor: '#10b981',
          iconColor: '#10b981',
          textColor: '#065f46',
          defaultIcon: 'checkmark-circle' as const,
        };
      case 'warning':
        return {
          backgroundColor: '#fef3c7',
          borderColor: '#f59e0b',
          iconColor: '#f59e0b',
          textColor: '#92400e',
          defaultIcon: 'warning' as const,
        };
      case 'error':
        return {
          backgroundColor: '#fee2e2',
          borderColor: '#ef4444',
          iconColor: '#ef4444',
          textColor: '#991b1b',
          defaultIcon: 'close-circle' as const,
        };
      default:
        return {
          backgroundColor: '#dbeafe',
          borderColor: '#0ea5e9',
          iconColor: '#0ea5e9',
          textColor: '#1e40af',
          defaultIcon: 'information-circle' as const,
        };
    }
  };

  const variantStyles = getVariantStyles();
  const iconName = icon || variantStyles.defaultIcon;

  return (
    <View
      style={[
        styles.container,
        {
          backgroundColor: variantStyles.backgroundColor,
          borderColor: variantStyles.borderColor,
        },
        style,
      ]}
    >
      <View style={styles.content}>
        <Ionicons name={iconName} size={20} color={variantStyles.iconColor} style={styles.icon} />
        <View style={styles.textContainer}>
          {title && (
            <Text style={[styles.title, { color: variantStyles.textColor }]}>{title}</Text>
          )}
          <Text style={[styles.message, { color: variantStyles.textColor }]}>{children}</Text>
        </View>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    borderWidth: 1,
    borderRadius: 8,
    padding: 12,
  },
  content: {
    flexDirection: 'row',
    alignItems: 'flex-start',
  },
  icon: {
    marginRight: 12,
    marginTop: 2,
  },
  textContainer: {
    flex: 1,
  },
  title: {
    fontSize: 14,
    fontWeight: '600',
    marginBottom: 4,
  },
  message: {
    fontSize: 14,
    lineHeight: 20,
  },
});

