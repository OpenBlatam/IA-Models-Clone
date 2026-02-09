import React from 'react';
import {
  View,
  TouchableOpacity,
  StyleSheet,
  Modal,
  Animated,
} from 'react-native';
import { BlurView } from 'expo-blur';
import { COLORS } from '../../constants/config';

interface BackdropProps {
  visible: boolean;
  onClose: () => void;
  children?: React.ReactNode;
  blur?: boolean;
  opacity?: number;
}

/**
 * Backdrop component
 * Modal backdrop with blur
 */
export function Backdrop({
  visible,
  onClose,
  children,
  blur = true,
  opacity = 0.5,
}: BackdropProps) {
  const fadeAnim = React.useRef(new Animated.Value(0)).current;

  React.useEffect(() => {
    Animated.timing(fadeAnim, {
      toValue: visible ? 1 : 0,
      duration: 300,
      useNativeDriver: true,
    }).start();
  }, [visible, fadeAnim]);

  if (!visible) return null;

  return (
    <Modal
      visible={visible}
      transparent
      animationType="none"
      onRequestClose={onClose}
    >
      <Animated.View
        style={[
          styles.container,
          {
            opacity: fadeAnim,
          },
        ]}
      >
        {blur ? (
          <BlurView intensity={20} style={StyleSheet.absoluteFill} />
        ) : (
          <View
            style={[
              StyleSheet.absoluteFill,
              { backgroundColor: `rgba(0, 0, 0, ${opacity})` },
            ]}
          />
        )}
        <TouchableOpacity
          style={StyleSheet.absoluteFill}
          activeOpacity={1}
          onPress={onClose}
        />
        {children && <View style={styles.content}>{children}</View>}
      </Animated.View>
    </Modal>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  content: {
    zIndex: 1,
  },
});

