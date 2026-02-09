import { ReactNode } from 'react';
import { View, Text, StyleSheet, Modal, TouchableOpacity } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { useTheme } from '@/contexts/ThemeContext';
import { AccessibleButton } from '../ui/AccessibleButton';

interface ConfirmationDialogProps {
  visible: boolean;
  title: string;
  message: string;
  confirmText?: string;
  cancelText?: string;
  onConfirm: () => void;
  onCancel: () => void;
  variant?: 'default' | 'danger';
  icon?: keyof typeof Ionicons.glyphMap;
}

export function ConfirmationDialog({
  visible,
  title,
  message,
  confirmText = 'Confirm',
  cancelText = 'Cancel',
  onConfirm,
  onCancel,
  variant = 'default',
  icon,
}: ConfirmationDialogProps) {
  const { colors } = useTheme();
  const defaultIcon = variant === 'danger' ? 'warning' : 'help-circle';

  return (
    <Modal visible={visible} transparent animationType="fade" onRequestClose={onCancel}>
      <View style={styles.overlay}>
        <SafeAreaView style={styles.container}>
          <View style={[styles.dialog, { backgroundColor: colors.surface }]}>
            <View style={styles.iconContainer}>
              <Ionicons
                name={icon || defaultIcon}
                size={48}
                color={variant === 'danger' ? colors.error : colors.primary}
              />
            </View>

            <Text style={[styles.title, { color: colors.text }]}>{title}</Text>
            <Text style={[styles.message, { color: colors.textSecondary }]}>{message}</Text>

            <View style={styles.actions}>
              <AccessibleButton
                title={cancelText}
                onPress={onCancel}
                variant="outline"
                style={styles.button}
                accessibilityLabel={cancelText}
              />
              <AccessibleButton
                title={confirmText}
                onPress={onConfirm}
                variant={variant === 'danger' ? 'danger' : 'primary'}
                style={styles.button}
                accessibilityLabel={confirmText}
              />
            </View>
          </View>
        </SafeAreaView>
      </View>
    </Modal>
  );
}

const styles = StyleSheet.create({
  overlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  container: {
    width: '100%',
    maxWidth: 400,
  },
  dialog: {
    borderRadius: 16,
    padding: 24,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 8,
  },
  iconContainer: {
    marginBottom: 16,
  },
  title: {
    fontSize: 20,
    fontWeight: '600',
    marginBottom: 8,
    textAlign: 'center',
  },
  message: {
    fontSize: 16,
    textAlign: 'center',
    marginBottom: 24,
    lineHeight: 22,
  },
  actions: {
    flexDirection: 'row',
    gap: 12,
    width: '100%',
  },
  button: {
    flex: 1,
  },
});

