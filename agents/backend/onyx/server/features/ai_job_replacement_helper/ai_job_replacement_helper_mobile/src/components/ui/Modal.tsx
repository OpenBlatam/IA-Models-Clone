import React, { memo, ReactNode } from 'react';
import { Modal as RNModal, View, Text, StyleSheet, TouchableOpacity, KeyboardAvoidingView, Platform } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { useTheme } from '@/theme/theme';
import { useKeyboard } from '@/hooks/useKeyboard';

export interface ModalProps {
  visible: boolean;
  onClose: () => void;
  children: ReactNode;
  title?: string;
  showCloseButton?: boolean;
  animationType?: 'none' | 'slide' | 'fade';
  presentationStyle?: 'fullScreen' | 'pageSheet' | 'formSheet' | 'overFullScreen';
  transparent?: boolean;
  accessibilityLabel?: string;
}

function ModalComponent({
  visible,
  onClose,
  children,
  title,
  showCloseButton = true,
  animationType = 'slide',
  presentationStyle = 'pageSheet',
  transparent = false,
  accessibilityLabel,
}: ModalProps) {
  const theme = useTheme();
  const { height: keyboardHeight } = useKeyboard();

  return (
    <RNModal
      visible={visible}
      animationType={animationType}
      presentationStyle={presentationStyle}
      transparent={transparent}
      onRequestClose={onClose}
      accessibilityLabel={accessibilityLabel}
    >
      <SafeAreaView
        style={[styles.container, { backgroundColor: transparent ? 'transparent' : theme.colors.background }]}
        edges={['top', 'bottom']}
      >
        <KeyboardAvoidingView
          behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
          style={styles.keyboardView}
          keyboardVerticalOffset={keyboardHeight}
        >
          {(title || showCloseButton) && (
            <View style={[styles.header, { borderBottomColor: theme.colors.border }]}>
              {title && (
                <View style={styles.titleContainer}>
                  <Text style={[styles.title, { color: theme.colors.text }]}>{title}</Text>
                </View>
              )}
              {showCloseButton && (
                <TouchableOpacity
                  onPress={onClose}
                  style={styles.closeButton}
                  accessibilityRole="button"
                  accessibilityLabel="Close modal"
                >
                  <Ionicons name="close" size={24} color={theme.colors.text} />
                </TouchableOpacity>
              )}
            </View>
          )}
          <View style={styles.content}>{children}</View>
        </KeyboardAvoidingView>
      </SafeAreaView>
    </RNModal>
  );
}

export const Modal = memo(ModalComponent);

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  keyboardView: {
    flex: 1,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderBottomWidth: 1,
  },
  titleContainer: {
    flex: 1,
  },
  title: {
    fontSize: 20,
    fontWeight: '600',
  },
  closeButton: {
    padding: 8,
  },
  content: {
    flex: 1,
  },
});

