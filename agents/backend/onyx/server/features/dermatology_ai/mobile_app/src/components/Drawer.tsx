import React, { useEffect } from 'react';
import {
  View,
  StyleSheet,
  TouchableWithoutFeedback,
  Dimensions,
} from 'react-native';
import Animated, {
  useAnimatedStyle,
  useSharedValue,
  withSpring,
  withTiming,
} from 'react-native-reanimated';
import { useTheme } from '../context/ThemeContext';

interface DrawerProps {
  visible: boolean;
  onClose: () => void;
  children: React.ReactNode;
  position?: 'left' | 'right' | 'top' | 'bottom';
  width?: number | string;
  height?: number | string;
}

const Drawer: React.FC<DrawerProps> = ({
  visible,
  onClose,
  children,
  position = 'left',
  width = '80%',
  height = '100%',
}) => {
  const { colors } = useTheme();
  const screenWidth = Dimensions.get('window').width;
  const screenHeight = Dimensions.get('window').height;
  const translateX = useSharedValue(0);
  const translateY = useSharedValue(0);
  const opacity = useSharedValue(0);

  useEffect(() => {
    if (visible) {
      opacity.value = withTiming(1, { duration: 300 });
      translateX.value = withSpring(0, { damping: 20, stiffness: 90 });
      translateY.value = withSpring(0, { damping: 20, stiffness: 90 });
    } else {
      opacity.value = withTiming(0, { duration: 300 });
      if (position === 'left') translateX.value = withSpring(-screenWidth);
      if (position === 'right') translateX.value = withSpring(screenWidth);
      if (position === 'top') translateY.value = withSpring(-screenHeight);
      if (position === 'bottom') translateY.value = withSpring(screenHeight);
    }
  }, [visible, position, screenWidth, screenHeight, translateX, translateY, opacity]);

  const getInitialPosition = () => {
    switch (position) {
      case 'left':
        return { left: 0, translateX: -screenWidth };
      case 'right':
        return { right: 0, translateX: screenWidth };
      case 'top':
        return { top: 0, translateY: -screenHeight };
      case 'bottom':
        return { bottom: 0, translateY: screenHeight };
      default:
        return { left: 0, translateX: -screenWidth };
    }
  };

  const initialPos = getInitialPosition();

  const drawerStyle = useAnimatedStyle(() => {
    const baseStyle: any = {
      opacity: opacity.value,
    };

    if (position === 'left' || position === 'right') {
      baseStyle.transform = [{ translateX: translateX.value }];
      baseStyle.width = typeof width === 'number' ? width : screenWidth * (parseFloat(width) / 100);
      baseStyle.height = '100%';
    } else {
      baseStyle.transform = [{ translateY: translateY.value }];
      baseStyle.height = typeof height === 'number' ? height : screenHeight * (parseFloat(height) / 100);
      baseStyle.width = '100%';
    }

    return baseStyle;
  });

  const backdropStyle = useAnimatedStyle(() => {
    return {
      opacity: opacity.value * 0.5,
    };
  });

  if (!visible) return null;

  return (
    <View style={styles.container}>
      <TouchableWithoutFeedback onPress={onClose}>
        <Animated.View
          style={[
            styles.backdrop,
            {
              backgroundColor: colors.text,
            },
            backdropStyle,
          ]}
        />
      </TouchableWithoutFeedback>
      <Animated.View
        style={[
          styles.drawer,
          {
            backgroundColor: colors.card,
            ...initialPos,
          },
          drawerStyle,
        ]}
      >
        {children}
      </Animated.View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    ...StyleSheet.absoluteFillObject,
    zIndex: 1000,
  },
  backdrop: {
    ...StyleSheet.absoluteFillObject,
  },
  drawer: {
    position: 'absolute',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 8,
    elevation: 10,
  },
});

export default Drawer;

