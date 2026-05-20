import React, { useEffect } from 'react';
import {
  View,
  StyleSheet,
  Modal,
  TouchableWithoutFeedback,
  Dimensions,
} from 'react-native';
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  withSpring,
  withTiming,
} from 'react-native-reanimated';
import { useTheme } from '../context/ThemeContext';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

const { height: SCREEN_HEIGHT } = Dimensions.get('window');

interface ModalProps {
  visible: boolean;
  onClose: () => void;
  children: React.ReactNode;
  animationType?: 'fade' | 'slide' | 'scale';
  transparent?: boolean;
  dismissible?: boolean;
}

const CustomModal: React.FC<ModalProps> = ({
  visible,
  onClose,
  children,
  animationType = 'fade',
  transparent = true,
  dismissible = true,
}) => {
  const { colors } = useTheme();
  const insets = useSafeAreaInsets();
  const opacity = useSharedValue(0);
  const scale = useSharedValue(0.9);
  const translateY = useSharedValue(SCREEN_HEIGHT);

  useEffect(() => {
    if (visible) {
      opacity.value = withTiming(1, { duration: 300 });
      if (animationType === 'scale') {
        scale.value = withSpring(1, { damping: 15 });
      } else if (animationType === 'slide') {
        translateY.value = withSpring(0, { damping: 20 });
      }
    } else {
      opacity.value = withTiming(0, { duration: 300 });
      if (animationType === 'scale') {
        scale.value = withTiming(0.9, { duration: 300 });
      } else if (animationType === 'slide') {
        translateY.value = withTiming(SCREEN_HEIGHT, { duration: 300 });
      }
    }
  }, [visible, animationType]);

  const backdropStyle = useAnimatedStyle(() => {
    return {
      opacity: opacity.value,
    };
  });

  const contentStyle = useAnimatedStyle(() => {
    if (animationType === 'scale') {
      return {
        opacity: opacity.value,
        transform: [{ scale: scale.value }],
      };
    } else if (animationType === 'slide') {
      return {
        transform: [{ translateY: translateY.value }],
      };
    }
    return {
      opacity: opacity.value,
    };
  });

  return (
    <Modal
      visible={visible}
      transparent={transparent}
      animationType="none"
      onRequestClose={onClose}
    >
      <View style={styles.container}>
        {transparent && (
          <Animated.View
            style={[
              styles.backdrop,
              { backgroundColor: 'rgba(0, 0, 0, 0.5)' },
              backdropStyle,
            ]}
          >
            {dismissible && (
              <TouchableWithoutFeedback onPress={onClose}>
                <View style={StyleSheet.absoluteFill} />
              </TouchableWithoutFeedback>
            )}
          </Animated.View>
        )}

        <Animated.View
          style={[
            styles.content,
            {
              backgroundColor: colors.card,
              paddingBottom: insets.bottom,
            },
            contentStyle,
          ]}
        >
          {children}
        </Animated.View>
      </View>
    </Modal>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  backdrop: {
    ...StyleSheet.absoluteFillObject,
  },
  content: {
    borderRadius: 16,
    padding: 20,
    maxWidth: '90%',
    maxHeight: '80%',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.25,
    shadowRadius: 8,
    elevation: 10,
  },
});

export default CustomModal;

