import React from 'react';
import {
  View,
  StyleSheet,
  TouchableWithoutFeedback,
  Modal,
} from 'react-native';
import Animated, {
  useAnimatedStyle,
  useSharedValue,
  withTiming,
} from 'react-native-reanimated';
import { useTheme } from '../context/ThemeContext';

interface BackdropProps {
  visible: boolean;
  onClose: () => void;
  children: React.ReactNode;
  opacity?: number;
  animated?: boolean;
}

const Backdrop: React.FC<BackdropProps> = ({
  visible,
  onClose,
  children,
  opacity = 0.5,
  animated = true,
}) => {
  const { colors } = useTheme();
  const backdropOpacity = useSharedValue(0);

  React.useEffect(() => {
    if (animated) {
      backdropOpacity.value = withTiming(visible ? opacity : 0, {
        duration: 300,
      });
    } else {
      backdropOpacity.value = visible ? opacity : 0;
    }
  }, [visible, opacity, animated, backdropOpacity]);

  const animatedStyle = useAnimatedStyle(() => {
    return {
      opacity: backdropOpacity.value,
    };
  });

  if (!visible) return null;

  return (
    <Modal
      visible={visible}
      transparent={true}
      animationType="none"
      onRequestClose={onClose}
    >
      <View style={styles.container}>
        <TouchableWithoutFeedback onPress={onClose}>
          <Animated.View
            style={[
              styles.backdrop,
              {
                backgroundColor: colors.text,
              },
              animatedStyle,
            ]}
          />
        </TouchableWithoutFeedback>
        <View style={styles.content}>{children}</View>
      </View>
    </Modal>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    position: 'relative',
  },
  backdrop: {
    ...StyleSheet.absoluteFillObject,
  },
  content: {
    flex: 1,
    zIndex: 1,
  },
});

export default Backdrop;

