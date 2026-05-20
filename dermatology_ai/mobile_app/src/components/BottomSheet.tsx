import React, { useEffect } from 'react';
import {
  View,
  StyleSheet,
  Modal,
  TouchableWithoutFeedback,
  PanResponder,
  Dimensions,
} from 'react-native';
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  withSpring,
  withTiming,
  runOnJS,
} from 'react-native-reanimated';
import { useTheme } from '../context/ThemeContext';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

const { height: SCREEN_HEIGHT } = Dimensions.get('window');

interface BottomSheetProps {
  visible: boolean;
  onClose: () => void;
  children: React.ReactNode;
  height?: number | string;
  snapPoints?: number[];
}

const BottomSheet: React.FC<BottomSheetProps> = ({
  visible,
  onClose,
  children,
  height = '80%',
  snapPoints = [],
}) => {
  const { colors } = useTheme();
  const insets = useSafeAreaInsets();
  const translateY = useSharedValue(SCREEN_HEIGHT);
  const panY = useSharedValue(0);

  const sheetHeight =
    typeof height === 'string'
      ? (parseFloat(height) / 100) * SCREEN_HEIGHT
      : height;

  useEffect(() => {
    if (visible) {
      translateY.value = withSpring(0, { damping: 20 });
    } else {
      translateY.value = withTiming(SCREEN_HEIGHT, { duration: 300 });
    }
  }, [visible]);

  const panResponder = PanResponder.create({
    onStartShouldSetPanResponder: () => true,
    onMoveShouldSetPanResponder: (_, gestureState) => {
      return Math.abs(gestureState.dy) > 5;
    },
    onPanResponderGrant: () => {
      panY.value = translateY.value;
    },
    onPanResponderMove: (_, gestureState) => {
      if (gestureState.dy > 0) {
        translateY.value = panY.value + gestureState.dy;
      }
    },
    onPanResponderRelease: (_, gestureState) => {
      const shouldClose = gestureState.dy > sheetHeight * 0.3 || gestureState.vy > 0.5;

      if (shouldClose) {
        translateY.value = withTiming(SCREEN_HEIGHT, { duration: 300 }, () => {
          runOnJS(onClose)();
        });
      } else {
        translateY.value = withSpring(0, { damping: 20 });
      }
      panY.value = 0;
    },
  });

  const animatedStyle = useAnimatedStyle(() => {
    return {
      transform: [{ translateY: translateY.value }],
    };
  });

  const backdropOpacity = useSharedValue(0);

  useEffect(() => {
    backdropOpacity.value = visible
      ? withTiming(0.5, { duration: 300 })
      : withTiming(0, { duration: 300 });
  }, [visible]);

  const backdropStyle = useAnimatedStyle(() => {
    return {
      opacity: backdropOpacity.value,
    };
  });

  return (
    <Modal
      visible={visible}
      transparent={true}
      animationType="none"
      onRequestClose={onClose}
    >
      <View style={styles.container}>
        <Animated.View
          style={[styles.backdrop, backdropStyle]}
          pointerEvents={visible ? 'auto' : 'none'}
        >
          <TouchableWithoutFeedback onPress={onClose}>
            <View style={StyleSheet.absoluteFill} />
          </TouchableWithoutFeedback>
        </Animated.View>

        <Animated.View
          style={[
            styles.sheet,
            {
              height: sheetHeight,
              backgroundColor: colors.card,
              paddingBottom: insets.bottom,
            },
            animatedStyle,
          ]}
          {...panResponder.panHandlers}
        >
          <View style={styles.handleContainer}>
            <View style={[styles.handle, { backgroundColor: colors.border }]} />
          </View>
          {children}
        </Animated.View>
      </View>
    </Modal>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'flex-end',
  },
  backdrop: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: '#000',
  },
  sheet: {
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: -4 },
    shadowOpacity: 0.25,
    shadowRadius: 8,
    elevation: 10,
  },
  handleContainer: {
    alignItems: 'center',
    paddingTop: 12,
    paddingBottom: 8,
  },
  handle: {
    width: 40,
    height: 4,
    borderRadius: 2,
  },
});

export default BottomSheet;
