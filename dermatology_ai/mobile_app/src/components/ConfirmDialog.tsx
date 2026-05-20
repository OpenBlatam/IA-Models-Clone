import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import { useTheme } from '../context/ThemeContext';
import Modal from './Modal';

interface ConfirmDialogProps {
  visible: boolean;
  title: string;
  message: string;
  confirmText?: string;
  cancelText?: string;
  onConfirm: () => void;
  onCancel: () => void;
  type?: 'danger' | 'warning' | 'info';
}

const ConfirmDialog: React.FC<ConfirmDialogProps> = ({
  visible,
  title,
  message,
  confirmText = 'Confirmar',
  cancelText = 'Cancelar',
  onConfirm,
  onCancel,
  type = 'info',
}) => {
  const { colors } = useTheme();

  const getTypeColor = () => {
    switch (type) {
      case 'danger':
        return colors.error;
      case 'warning':
        return colors.warning || '#f59e0b';
      default:
        return colors.primary;
    }
  };

  const typeColor = getTypeColor();

  return (
    <Modal visible={visible} onClose={onCancel} animationType="scale">
      <View style={styles.container}>
        <Text style={[styles.title, { color: colors.text }]}>{title}</Text>
        <Text style={[styles.message, { color: colors.textSecondary }]}>
          {message}
        </Text>
        <View style={styles.actions}>
          <TouchableOpacity
            style={[
              styles.button,
              styles.cancelButton,
              { borderColor: colors.border },
            ]}
            onPress={onCancel}
            activeOpacity={0.7}
          >
            <Text style={[styles.buttonText, { color: colors.text }]}>
              {cancelText}
            </Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[styles.button, styles.confirmButton, { backgroundColor: typeColor }]}
            onPress={onConfirm}
            activeOpacity={0.7}
          >
            <Text style={styles.confirmButtonText}>{confirmText}</Text>
          </TouchableOpacity>
        </View>
      </View>
    </Modal>
  );
};

const styles = StyleSheet.create({
  container: {
    minWidth: 280,
  },
  title: {
    fontSize: 20,
    fontWeight: '600',
    marginBottom: 12,
    textAlign: 'center',
  },
  message: {
    fontSize: 14,
    lineHeight: 20,
    marginBottom: 24,
    textAlign: 'center',
  },
  actions: {
    flexDirection: 'row',
    gap: 12,
  },
  button: {
    flex: 1,
    paddingVertical: 12,
    paddingHorizontal: 16,
    borderRadius: 8,
    alignItems: 'center',
  },
  cancelButton: {
    borderWidth: 1,
  },
  confirmButton: {
    // backgroundColor set dynamically
  },
  buttonText: {
    fontSize: 16,
    fontWeight: '600',
  },
  confirmButtonText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#fff',
  },
});

export default ConfirmDialog;

