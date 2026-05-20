'use client';

import React from 'react';
import { clsx } from 'clsx';

interface MarqueeProps {
  children: React.ReactNode;
  direction?: 'left' | 'right' | 'up' | 'down';
  speed?: number;
  pauseOnHover?: boolean;
  className?: string;
}

export const Marquee: React.FC<MarqueeProps> = ({
  children,
  direction = 'left',
  speed = 50,
  pauseOnHover = false,
  className,
}) => {
  const directionClasses = {
    left: 'animate-marquee-left',
    right: 'animate-marquee-right',
    up: 'animate-marquee-up',
    down: 'animate-marquee-down',
  };

  return (
    <div
      className={clsx(
        'overflow-hidden whitespace-nowrap',
        pauseOnHover && 'hover:[animation-play-state:paused]',
        className
      )}
      style={{
        animationDuration: `${speed}s`,
      }}
    >
      <div className={clsx('inline-block', directionClasses[direction])}>
        {children}
      </div>
    </div>
  );
};
