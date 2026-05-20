import React, { useEffect, useRef } from 'react';
import { View, StyleSheet, Animated } from 'react-native';
import { useTheme } from '../contexts/ThemeContext';

interface WaveAnimationProps {
  color?: string;
  size?: number;
  duration?: number;
}

export const WaveAnimation: React.FC<WaveAnimationProps> = ({
  color,
  size = 50,
  duration = 1000,
}) => {
  const { theme } = useTheme();
  const scale1 = useRef(new Animated.Value(0)).current;
  const scale2 = useRef(new Animated.Value(0)).current;
  const scale3 = useRef(new Animated.Value(0)).current;
  const opacity1 = useRef(new Animated.Value(1)).current;
  const opacity2 = useRef(new Animated.Value(1)).current;
  const opacity3 = useRef(new Animated.Value(1)).current;

  const waveColor = color || theme.primary;

  useEffect(() => {
    const animate = (scale: Animated.Value, opacity: Animated.Value, delay: number) => {
      return Animated.loop(
        Animated.parallel([
          Animated.sequence([
            Animated.delay(delay),
            Animated.parallel([
              Animated.timing(scale, {
                toValue: 1,
                duration,
                useNativeDriver: true,
              }),
              Animated.timing(opacity, {
                toValue: 0,
                duration,
                useNativeDriver: true,
              }),
            ]),
          ]),
        ])
      );
    };

    animate(scale1, opacity1, 0).start();
    animate(scale2, opacity2, duration / 3).start();
    animate(scale3, opacity3, (duration * 2) / 3).start();
  }, [scale1, scale2, scale3, opacity1, opacity2, opacity3, duration]);

  const Wave = ({ scale, opacity, delay }: any) => (
    <Animated.View
      style={[
        styles.wave,
        {
          width: size,
          height: size,
          borderRadius: size / 2,
          borderColor: waveColor,
          transform: [{ scale }],
          opacity,
        },
      ]}
    />
  );

  return (
    <View style={[styles.container, { width: size, height: size }]}>
      <Wave scale={scale1} opacity={opacity1} />
      <Wave scale={scale2} opacity={opacity2} />
      <Wave scale={scale3} opacity={opacity3} />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    justifyContent: 'center',
    alignItems: 'center',
    position: 'relative',
  },
  wave: {
    position: 'absolute',
    borderWidth: 2,
    backgroundColor: 'transparent',
  },
});

