'use client';

import { ReactNode } from 'react';
import { useSpring, animated, SpringValue } from '@react-spring/web';
import { cn } from '@/lib/utils/cn';

interface SpringAnimationProps {
  children: ReactNode;
  from?: { opacity?: number; transform?: string };
  to?: { opacity?: number; transform?: string };
  delay?: number;
  className?: string;
  config?: {
    tension?: number;
    friction?: number;
    mass?: number;
  };
}

export default function SpringAnimation({
  children,
  from = { opacity: 0, transform: 'translateY(20px)' },
  to = { opacity: 1, transform: 'translateY(0px)' },
  delay = 0,
  className,
  config = { tension: 280, friction: 60 },
}: SpringAnimationProps) {
  const [springs, api] = useSpring(() => ({
    from,
    to,
    delay,
    config,
  }));

  return (
    <animated.div
      style={springs}
      className={cn(className)}
    >
      {children}
    </animated.div>
  );
}



