import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useTheme } from '../context/ThemeContext';

interface AlertProps {
  title?: string;
  message: string;
  type?: 'success' | 'error' | 'warning' | 'info';
  onClose?: () => void;
  showIcon?: boolean;
  action?: {
    label: string;
    onPress: () => void;
  };
}

const Alert: React.FC<AlertProps> = ({
  title,
  message,
  type = 'info',
  onClose,
  showIcon = true,
  action,
}) => {
  const { colors } = useTheme();

  const getTypeStyles = () => {
    switch (type) {
      case 'success':
        return {
          backgroundColor: `${colors.success}20`,
          borderColor: colors.success,
          icon: 'checkmark-circle' as const,
          iconColor: colors.success,
        };
      case 'error':
        return {
          backgroundColor: `${colors.error}20`,
          borderColor: colors.error,
          icon: 'close-circle' as const,
          iconColor: colors.error,
        };
      case 'warning':
        return {
          backgroundColor: `${colors.warning}20`,
          borderColor: colors.warning,
          icon: 'warning' as const,
          iconColor: colors.warning,
        };
      default:
        return {
          backgroundColor: `${colors.primary}20`,
          borderColor: colors.primary,
          icon: 'information-circle' as const,
          iconColor: colors.primary,
        };
    }
  };

  const typeStyles = getTypeStyles();

  return (
    <View
      style={[
        styles.container,
        {
          backgroundColor: typeStyles.backgroundColor,
          borderColor: typeStyles.borderColor,
        },
      ]}
    >
      <View style={styles.content}>
        {showIcon && (
          <Ionicons
            name={typeStyles.icon}
            size={24}
            color={typeStyles.iconColor}
            style={styles.icon}
          />
        )}
        <View style={styles.textContainer}>
          {title && (
            <Text
              style={[
                styles.title,
                {
                  color: typeStyles.iconColor,
                },
              ]}
            >
              {title}
            </Text>
          )}
          <Text style={[styles.message, { color: colors.text }]}>
            {message}
          </Text>
        </View>
        {onClose && (
          <TouchableOpacity onPress={onClose} style={styles.closeButton}>
            <Ionicons name="close" size={20} color={colors.textSecondary} />
          </TouchableOpacity>
        )}
      </View>
      {action && (
        <TouchableOpacity
          style={[
            styles.actionButton,
            {
              borderTopColor: typeStyles.borderColor,
            },
          ]}
          onPress={action.onPress}
          activeOpacity={0.7}
        >
          <Text
            style={[
              styles.actionText,
              {
                color: typeStyles.iconColor,
              },
            ]}
          >
            {action.label}
          </Text>
        </TouchableOpacity>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    borderRadius: 12,
    borderWidth: 1,
    marginVertical: 8,
    overflow: 'hidden',
  },
  content: {
    flexDirection: 'row',
    padding: 16,
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
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 4,
  },
  message: {
    fontSize: 14,
    lineHeight: 20,
  },
  closeButton: {
    padding: 4,
    marginLeft: 8,
  },
  actionButton: {
    padding: 12,
    borderTopWidth: 1,
    alignItems: 'center',
  },
  actionText: {
    fontSize: 14,
    fontWeight: '600',
  },
});

export default Alert;

