import React, { ReactNode, memo } from 'react';
import { View, StyleSheet, ViewStyle } from 'react-native';
import Animated, { FadeIn, FadeOut } from 'react-native-reanimated';
import { COLORS, SPACING, BORDER_RADIUS } from '../../constants/config';

const AnimatedView = Animated.createAnimatedComponent(View);

interface CardProps {
  children: ReactNode;
  style?: ViewStyle;
  animated?: boolean;
  delay?: number;
}

export const Card = memo(function Card({
  children,
  style,
  animated = true,
  delay = 0,
}: CardProps) {
  const content = <View style={[styles.card, style]}>{children}</View>;

  if (!animated) {
    return content;
  }

  return (
    <AnimatedView
      entering={FadeIn.delay(delay).duration(300)}
      exiting={FadeOut.duration(200)}
    >
      {content}
    </AnimatedView>
  );
});

const styles = StyleSheet.create({
  card: {
    backgroundColor: COLORS.surface,
    borderRadius: BORDER_RADIUS.md,
    padding: SPACING.md,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
});

