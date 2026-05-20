import React, { useEffect } from 'react';
import { View, StyleSheet, ViewStyle } from 'react-native';
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  withRepeat,
  withTiming,
  withSequence,
  Easing,
} from 'react-native-reanimated';
import { useTheme } from '../context/ThemeContext';

interface Particle {
  id: number;
  x: number;
  y: number;
  size: number;
  duration: number;
}

interface ParticleEffectProps {
  particleCount?: number;
  colors?: string[];
  style?: ViewStyle;
}

const ParticleEffect: React.FC<ParticleEffectProps> = ({
  particleCount = 20,
  colors,
  style,
}) => {
  const { colors: themeColors } = useTheme();
  const particleColors = colors || [themeColors.primary, themeColors.secondary];
  const [particles, setParticles] = React.useState<Particle[]>([]);

  useEffect(() => {
    const newParticles: Particle[] = Array.from({ length: particleCount }, (_, i) => ({
      id: i,
      x: Math.random() * 100,
      y: Math.random() * 100,
      size: Math.random() * 4 + 2,
      duration: Math.random() * 2000 + 1000,
    }));
    setParticles(newParticles);
  }, [particleCount]);

  return (
    <View style={[styles.container, style]}>
      {particles.map((particle) => (
        <ParticleItem
          key={particle.id}
          particle={particle}
          color={particleColors[particle.id % particleColors.length]}
        />
      ))}
    </View>
  );
};

const ParticleItem: React.FC<{ particle: Particle; color: string }> = ({
  particle,
  color,
}) => {
  const opacity = useSharedValue(1);
  const translateY = useSharedValue(0);
  const translateX = useSharedValue(0);

  useEffect(() => {
    opacity.value = withRepeat(
      withSequence(
        withTiming(0, { duration: particle.duration / 2 }),
        withTiming(1, { duration: particle.duration / 2 })
      ),
      -1,
      true
    );
    translateY.value = withRepeat(
      withTiming(-100, {
        duration: particle.duration,
        easing: Easing.out(Easing.ease),
      }),
      -1,
      false
    );
    translateX.value = withRepeat(
      withTiming((Math.random() - 0.5) * 50, {
        duration: particle.duration,
        easing: Easing.inOut(Easing.ease),
      }),
      -1,
      true
    );
  }, [particle.duration, opacity, translateY, translateX]);

  const animatedStyle = useAnimatedStyle(() => {
    return {
      opacity: opacity.value,
      transform: [
        { translateY: translateY.value },
        { translateX: translateX.value },
      ],
    };
  });

  return (
    <Animated.View
      style={[
        styles.particle,
        {
          left: `${particle.x}%`,
          top: `${particle.y}%`,
          width: particle.size,
          height: particle.size,
          backgroundColor: color,
          borderRadius: particle.size / 2,
        },
        animatedStyle,
      ]}
    />
  );
};

const styles = StyleSheet.create({
  container: {
    ...StyleSheet.absoluteFillObject,
    overflow: 'hidden',
  },
  particle: {
    position: 'absolute',
  },
});

export default ParticleEffect;

