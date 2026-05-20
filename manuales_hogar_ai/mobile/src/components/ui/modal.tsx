/**
 * Modal Component
 * ===============
 * Custom modal component
 */

import { Modal, View, Text, StyleSheet, TouchableOpacity, Pressable } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { useApp } from '@/lib/context/app-context';
import Animated, { FadeIn, FadeOut } from 'react-native-reanimated';

interface ModalProps {
  visible: boolean;
  onClose: () => void;
  title?: string;
  children: React.ReactNode;
  showCloseButton?: boolean;
}

export function CustomModal({
  visible,
  onClose,
  title,
  children,
  showCloseButton = true,
}: ModalProps) {
  const { state } = useApp();
  const colors = state.colors;

  return (
    <Modal
      visible={visible}
      transparent
      animationType="fade"
      onRequestClose={onClose}
    >
      <Pressable style={styles.overlay} onPress={onClose}>
        <Animated.View
          entering={FadeIn}
          exiting={FadeOut}
          style={[styles.container, { backgroundColor: colors.background }]}
        >
          <Pressable onPress={(e) => e.stopPropagation()}>
            <SafeAreaView edges={['top']}>
              {(title || showCloseButton) && (
                <View style={styles.header}>
                  {title && (
                    <Text style={[styles.title, { color: colors.text }]}>{title}</Text>
                  )}
                  {showCloseButton && (
                    <TouchableOpacity onPress={onClose} style={styles.closeButton}>
                      <Ionicons name="close" size={24} color={colors.text} />
                    </TouchableOpacity>
                  )}
                </View>
              )}
              <View style={styles.content}>{children}</View>
            </SafeAreaView>
          </Pressable>
        </Animated.View>
      </Pressable>
    </Modal>
  );
}

const styles = StyleSheet.create({
  overlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  container: {
    width: '90%',
    maxWidth: 500,
    borderRadius: 16,
    maxHeight: '80%',
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#E5E5E5',
  },
  title: {
    fontSize: 20,
    fontWeight: '600',
    flex: 1,
  },
  closeButton: {
    padding: 4,
  },
  content: {
    padding: 16,
  },
});



