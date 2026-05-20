import React, { ReactNode } from 'react';
import Modal from 'react-native-modal';
import { View, StyleSheet, TouchableOpacity, ViewStyle } from 'react-native';
import { useColors } from '@/theme/colors';
import { SPACING } from '@/theme/spacing';

interface CustomModalProps {
  isVisible: boolean;
  onClose: () => void;
  children: ReactNode;
  animationIn?: 'fadeIn' | 'slideInUp' | 'slideInDown' | 'zoomIn';
  animationOut?: 'fadeOut' | 'slideOutDown' | 'slideOutUp' | 'zoomOut';
  style?: ViewStyle;
  avoidKeyboard?: boolean;
  backdropOpacity?: number;
}

export function CustomModal({
  isVisible,
  onClose,
  children,
  animationIn = 'slideInUp',
  animationOut = 'slideOutDown',
  style,
  avoidKeyboard = true,
  backdropOpacity = 0.5,
}: CustomModalProps): JSX.Element {
  const colors = useColors();

  return (
    <Modal
      isVisible={isVisible}
      onBackdropPress={onClose}
      onBackButtonPress={onClose}
      animationIn={animationIn}
      animationOut={animationOut}
      backdropOpacity={backdropOpacity}
      avoidKeyboard={avoidKeyboard}
      style={[styles.modal, { backgroundColor: colors.surface }, style]}
    >
      <View style={[styles.content, { backgroundColor: colors.surface }]}>
        {children}
      </View>
    </Modal>
  );
}

interface ModalHeaderProps {
  title: string;
  onClose: () => void;
  showCloseButton?: boolean;
}

export function ModalHeader({
  title,
  onClose,
  showCloseButton = true,
}: ModalHeaderProps): JSX.Element {
  const colors = useColors();

  return (
    <View style={[styles.header, { borderBottomColor: colors.border }]}>
      <View style={styles.headerContent}>
        {showCloseButton && (
          <TouchableOpacity
            onPress={onClose}
            style={styles.closeButton}
            accessibilityRole="button"
            accessibilityLabel="Cerrar"
          >
            {/* Close icon would go here */}
          </TouchableOpacity>
        )}
        <View style={styles.titleContainer}>
          {/* Title would go here */}
        </View>
        {!showCloseButton && <View style={styles.closeButtonPlaceholder} />}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  modal: {
    justifyContent: 'flex-end',
    margin: 0,
  },
  content: {
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
    padding: SPACING.lg,
    maxHeight: '90%',
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingBottom: SPACING.md,
    borderBottomWidth: 1,
    marginBottom: SPACING.md,
  },
  headerContent: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
  },
  closeButtonPlaceholder: {
    width: 24,
  },
  titleContainer: {
    flex: 1,
    alignItems: 'center',
  },
  closeButton: {
    width: 24,
    height: 24,
    alignItems: 'center',
    justifyContent: 'center',
  },
});

