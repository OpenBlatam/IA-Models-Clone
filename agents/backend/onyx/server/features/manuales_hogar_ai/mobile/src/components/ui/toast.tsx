/**
 * Toast Component
 * ===============
 * Toast notification component
 */

import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import Animated, { FadeInDown, FadeOutUp } from 'react-native-reanimated';
import { useApp } from '@/lib/context/app-context';

interface ToastProps {
  message: string;
  type?: 'success' | 'error' | 'info' | 'warning';
  visible: boolean;
  onHide: () => void;
}

export function Toast({ message, type = 'info', visible, onHide }: ToastProps) {
  const { state } = useApp();
  const colors = state.colors;

  if (!visible) return null;

  const typeConfig = {
    success: { icon: 'checkmark-circle', color: colors.success },
    error: { icon: 'close-circle', color: colors.error },
    info: { icon: 'information-circle', color: colors.info },
    warning: { icon: 'warning', color: colors.warning },
  };

  const config = typeConfig[type];

  return (
    <Animated.View
      entering={FadeInDown}
      exiting={FadeOutUp}
      style={[styles.container, { backgroundColor: config.color }]}
    >
      <Ionicons name={config.icon as any} size={20} color="#FFFFFF" />
      <Text style={styles.text}>{message}</Text>
      <TouchableOpacity onPress={onHide} style={styles.closeButton}>
        <Ionicons name="close" size={18} color="#FFFFFF" />
      </TouchableOpacity>
    </Animated.View>
  );
}

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 16,
    borderRadius: 8,
    marginHorizontal: 16,
    marginTop: 8,
    gap: 8,
  },
  text: {
    color: '#FFFFFF',
    fontSize: 14,
    fontWeight: '500',
    flex: 1,
  },
  closeButton: {
    padding: 4,
  },
});

