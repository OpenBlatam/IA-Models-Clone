import React, { useEffect } from 'react';
import { View, StyleSheet, ViewStyle } from 'react-native';
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  withRepeat,
  withTiming,
  Easing,
} from 'react-native-reanimated';
// Note: Requires react-native-svg
// import Svg, { Path } from 'react-native-svg';
import { useTheme } from '../context/ThemeContext';

interface WaveAnimationProps {
  height?: number;
  amplitude?: number;
  frequency?: number;
  speed?: number;
  color?: string;
  style?: ViewStyle;
}

const WaveAnimation: React.FC<WaveAnimationProps> = ({
  height = 100,
  amplitude = 20,
  frequency = 2,
  speed = 1,
  color,
  style,
}) => {
  const { colors } = useTheme();
  const waveOffset = useSharedValue(0);
  const waveColor = color || colors.primary;

  useEffect(() => {
    waveOffset.value = withRepeat(
      withTiming(360, {
        duration: 2000 / speed,
        easing: Easing.linear,
      }),
      -1,
      false
    );
  }, [speed, waveOffset]);

  const animatedStyle = useAnimatedStyle(() => {
    return {
      transform: [{ translateX: waveOffset.value }],
    };
  });

  const generateWavePath = (offset: number) => {
    const width = 200;
    let path = `M 0 ${height / 2}`;
    
    for (let x = 0; x <= width; x += 1) {
      const y =
        height / 2 +
        amplitude *
          Math.sin((x / width) * Math.PI * frequency * 2 + (offset * Math.PI) / 180);
      path += ` L ${x} ${y}`;
    }
    
    path += ` L ${width} ${height} L 0 ${height} Z`;
    return path;
  };

  // Simplified version without SVG for now
  return (
    <View style={[styles.container, { height, backgroundColor: waveColor + '20' }, style]}>
      <Animated.View style={[styles.wave, animatedStyle]}>
        <View style={[styles.waveShape, { backgroundColor: waveColor, opacity: 0.6 }]} />
      </Animated.View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    overflow: 'hidden',
    width: '100%',
  },
  wave: {
    position: 'absolute',
    width: '200%',
    height: '100%',
  },
  waveShape: {
    width: '100%',
    height: '100%',
    borderRadius: 20,
  },
});

export default WaveAnimation;

