import React, { memo } from 'react';
import LottieView, { LottieViewProps } from 'lottie-react-native';

interface LottieAnimationProps extends Omit<LottieViewProps, 'source'> {
  source: string | { uri: string } | any;
  autoPlay?: boolean;
  loop?: boolean;
  speed?: number;
}

function LottieAnimationComponent({
  source,
  autoPlay = true,
  loop = true,
  speed = 1,
  ...props
}: LottieAnimationProps): JSX.Element {
  return (
    <LottieView
      source={source}
      autoPlay={autoPlay}
      loop={loop}
      speed={speed}
      {...props}
    />
  );
}

export const LottieAnimation = memo(LottieAnimationComponent);

