import React from 'react';
import {
  Modal as RNModal,
  View,
  StyleSheet,
  TouchableOpacity,
  SafeAreaView,
} from 'react-native';
import { BlurView } from 'expo-blur';
import { COLORS, SPACING, BORDER_RADIUS } from '../../constants/config';

interface ModalProps {
  visible: boolean;
  onClose: () => void;
  children: React.ReactNode;
  title?: string;
  showCloseButton?: boolean;
  transparent?: boolean;
  animationType?: 'none' | 'slide' | 'fade';
}

/**
 * Modal component with backdrop
 * Follows mobile UI best practices
 */
export function Modal({
  visible,
  onClose,
  children,
  title,
  showCloseButton = true,
  transparent = true,
  animationType = 'fade',
}: ModalProps) {
  return (
    <RNModal
      visible={visible}
      transparent={transparent}
      animationType={animationType}
      onRequestClose={onClose}
    >
      <View style={styles.overlay}>
        {transparent ? (
          <BlurView intensity={20} style={StyleSheet.absoluteFill} />
        ) : (
          <View style={styles.backdrop} />
        )}

        <SafeAreaView style={styles.container} edges={['top', 'bottom']}>
          <View style={styles.content}>
            {title && (
              <View style={styles.header}>
                <View style={styles.titleContainer}>
                  {/* Title can be added here */}
                </View>
                {showCloseButton && (
                  <TouchableOpacity
                    style={styles.closeButton}
                    onPress={onClose}
                    accessibilityLabel="Close modal"
                    accessibilityRole="button"
                  >
                    <View style={styles.closeIcon}>✕</View>
                  </TouchableOpacity>
                )}
              </View>
            )}
            {children}
          </View>
        </SafeAreaView>
      </View>
    </RNModal>
  );
}

const styles = StyleSheet.create({
  overlay: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  backdrop: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
  },
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    width: '100%',
    padding: SPACING.lg,
  },
  content: {
    backgroundColor: COLORS.surface,
    borderRadius: BORDER_RADIUS.xl,
    width: '100%',
    maxWidth: 500,
    maxHeight: '80%',
    overflow: 'hidden',
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: SPACING.md,
    borderBottomWidth: 1,
    borderBottomColor: COLORS.surfaceLight,
  },
  titleContainer: {
    flex: 1,
  },
  closeButton: {
    padding: SPACING.xs,
  },
  closeIcon: {
    fontSize: 24,
    color: COLORS.text,
  },
});

