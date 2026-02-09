'use client';

import { ReactNode } from 'react';
import { cn } from '@/lib/utils';

interface MarqueeProps {
  children: ReactNode;
  direction?: 'left' | 'right' | 'up' | 'down';
  speed?: 'slow' | 'normal' | 'fast';
  pauseOnHover?: boolean;
  className?: string;
}

const speedClasses = {
  slow: 'animate-marquee-slow',
  normal: 'animate-marquee',
  fast: 'animate-marquee-fast',
};

export const Marquee = ({
  children,
  direction = 'left',
  speed = 'normal',
  pauseOnHover = false,
  className,
}: MarqueeProps) => {
  const isHorizontal = direction === 'left' || direction === 'right';
  const isReverse = direction === 'right' || direction === 'down';

  return (
    <div
      className={cn(
        'overflow-hidden',
        isHorizontal ? 'flex' : 'flex flex-col',
        className
      )}
    >
      <div
        className={cn(
          'flex',
          isHorizontal ? 'flex-row' : 'flex-col',
          speedClasses[speed],
          pauseOnHover && 'hover:[animation-play-state:paused]',
          'whitespace-nowrap'
        )}
        style={{
          animationDirection: isReverse ? 'reverse' : 'normal',
        }}
      >
        <div className={cn('flex', isHorizontal ? 'flex-row' : 'flex-col')}>
          {children}
        </div>
        <div className={cn('flex', isHorizontal ? 'flex-row' : 'flex-col')} aria-hidden="true">
          {children}
        </div>
      </div>
    </div>
  );
};

