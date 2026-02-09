import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import { useTheme } from '../contexts/ThemeContext';
import { Modal } from './Modal';
import { spacing, borderRadius, typography } from '../theme/colors';

interface AlertButton {
  text: string;
  onPress: () => void;
  style?: 'default' | 'cancel' | 'destructive';
}

interface AlertProps {
  visible: boolean;
  title?: string;
  message: string;
  buttons?: AlertButton[];
  onDismiss?: () => void;
}

export const Alert: React.FC<AlertProps> = ({
  visible,
  title,
  message,
  buttons = [{ text: 'OK', onPress: () => {} }],
  onDismiss,
}) => {
  const { theme } = useTheme();

  const getButtonStyle = (style?: string) => {
    switch (style) {
      case 'destructive':
        return { backgroundColor: theme.error };
      case 'cancel':
        return { backgroundColor: theme.surfaceVariant };
      default:
        return { backgroundColor: theme.primary };
    }
  };

  const getButtonTextColor = (style?: string) => {
    switch (style) {
      case 'destructive':
      case 'cancel':
        return theme.text;
      default:
        return theme.surface;
    }
  };

  return (
    <Modal
      visible={visible}
      onClose={() => {
        onDismiss?.();
        if (buttons.length === 0) {
          buttons[0]?.onPress();
        }
      }}
      title={title}
      size="small"
      showCloseButton={false}
    >
      <Text style={[styles.message, { color: theme.text }]}>{message}</Text>
      <View style={styles.buttons}>
        {buttons.map((button, index) => (
          <TouchableOpacity
            key={index}
            style={[
              styles.button,
              getButtonStyle(button.style),
              buttons.length > 1 && styles.buttonMultiple,
            ]}
            onPress={() => {
              button.onPress();
              onDismiss?.();
            }}
            activeOpacity={0.7}
          >
            <Text
              style={[
                styles.buttonText,
                {
                  color: getButtonTextColor(button.style),
                },
              ]}
            >
              {button.text}
            </Text>
          </TouchableOpacity>
        ))}
      </View>
    </Modal>
  );
};

const styles = StyleSheet.create({
  message: {
    ...typography.body,
    marginBottom: spacing.xl,
    textAlign: 'center',
  },
  buttons: {
    flexDirection: 'row',
    gap: spacing.md,
    justifyContent: 'flex-end',
  },
  button: {
    paddingHorizontal: spacing.xl,
    paddingVertical: spacing.md,
    borderRadius: borderRadius.md,
    minWidth: 80,
    alignItems: 'center',
  },
  buttonMultiple: {
    flex: 1,
  },
  buttonText: {
    ...typography.body,
    fontWeight: '600',
  },
});

