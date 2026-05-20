import React, { useState, useRef } from 'react';
import {
  View,
  TouchableOpacity,
  StyleSheet,
  Modal,
  TouchableWithoutFeedback,
} from 'react-native';
import { COLORS, SPACING, BORDER_RADIUS } from '../../constants/config';

interface PopoverProps {
  trigger: React.ReactNode;
  content: React.ReactNode;
  placement?: 'top' | 'bottom' | 'left' | 'right';
  onOpen?: () => void;
  onClose?: () => void;
}

/**
 * Popover component
 * Contextual popup menu
 */
export function Popover({
  trigger,
  content,
  placement = 'bottom',
  onOpen,
  onClose,
}: PopoverProps) {
  const [visible, setVisible] = useState(false);
  const triggerRef = useRef<View>(null);

  const handleOpen = () => {
    setVisible(true);
    onOpen?.();
  };

  const handleClose = () => {
    setVisible(false);
    onClose?.();
  };

  return (
    <>
      <TouchableOpacity onPress={handleOpen} ref={triggerRef}>
        {trigger}
      </TouchableOpacity>

      <Modal
        visible={visible}
        transparent
        animationType="fade"
        onRequestClose={handleClose}
      >
        <TouchableWithoutFeedback onPress={handleClose}>
          <View style={styles.overlay}>
            <TouchableWithoutFeedback>
              <View
                style={[
                  styles.popover,
                  placement === 'top' && styles.popoverTop,
                  placement === 'bottom' && styles.popoverBottom,
                  placement === 'left' && styles.popoverLeft,
                  placement === 'right' && styles.popoverRight,
                ]}
              >
                {content}
              </View>
            </TouchableWithoutFeedback>
          </View>
        </TouchableWithoutFeedback>
      </Modal>
    </>
  );
}

const styles = StyleSheet.create({
  overlay: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
  },
  popover: {
    backgroundColor: COLORS.surface,
    borderRadius: BORDER_RADIUS.md,
    padding: SPACING.md,
    minWidth: 200,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 3.84,
    elevation: 5,
  },
  popoverTop: {
    marginBottom: SPACING.md,
  },
  popoverBottom: {
    marginTop: SPACING.md,
  },
  popoverLeft: {
    marginRight: SPACING.md,
  },
  popoverRight: {
    marginLeft: SPACING.md,
  },
});

