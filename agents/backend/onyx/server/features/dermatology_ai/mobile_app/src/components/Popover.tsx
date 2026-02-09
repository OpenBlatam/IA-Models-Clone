import React, { useState } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  Modal,
  TouchableWithoutFeedback,
} from 'react-native';
import { useTheme } from '../context/ThemeContext';

interface PopoverProps {
  trigger: React.ReactNode;
  content: React.ReactNode;
  placement?: 'top' | 'bottom' | 'left' | 'right';
  onOpen?: () => void;
  onClose?: () => void;
}

const Popover: React.FC<PopoverProps> = ({
  trigger,
  content,
  placement = 'bottom',
  onOpen,
  onClose,
}) => {
  const { colors } = useTheme();
  const [visible, setVisible] = useState(false);

  const handleOpen = () => {
    setVisible(true);
    onOpen?.();
  };

  const handleClose = () => {
    setVisible(false);
    onClose?.();
  };

  const getPlacementStyles = () => {
    switch (placement) {
      case 'top':
        return { bottom: '100%', marginBottom: 8 };
      case 'bottom':
        return { top: '100%', marginTop: 8 };
      case 'left':
        return { right: '100%', marginRight: 8 };
      case 'right':
        return { left: '100%', marginLeft: 8 };
      default:
        return { top: '100%', marginTop: 8 };
    }
  };

  return (
    <View style={styles.container}>
      <TouchableOpacity onPress={handleOpen} activeOpacity={0.8}>
        {trigger}
      </TouchableOpacity>

      <Modal
        visible={visible}
        transparent={true}
        animationType="fade"
        onRequestClose={handleClose}
      >
        <TouchableWithoutFeedback onPress={handleClose}>
          <View style={styles.overlay}>
            <TouchableWithoutFeedback>
              <View
                style={[
                  styles.popover,
                  {
                    backgroundColor: colors.card,
                    ...getPlacementStyles(),
                  },
                ]}
              >
                {content}
              </View>
            </TouchableWithoutFeedback>
          </View>
        </TouchableWithoutFeedback>
      </Modal>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    position: 'relative',
  },
  overlay: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  popover: {
    position: 'absolute',
    borderRadius: 12,
    padding: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.25,
    shadowRadius: 8,
    elevation: 10,
    minWidth: 200,
    maxWidth: '80%',
  },
});

export default Popover;

