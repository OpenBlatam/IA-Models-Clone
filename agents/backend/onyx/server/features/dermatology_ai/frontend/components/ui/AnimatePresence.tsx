'use client';

import React, { useEffect, useState } from 'react';
import { clsx } from 'clsx';

interface AnimatePresenceProps {
  children: React.ReactNode;
  show: boolean;
  animation?: 'fade' | 'slide' | 'scale' | 'slideUp' | 'slideDown';
  duration?: number;
  className?: string;
}

export const AnimatePresence: React.FC<AnimatePresenceProps> = ({
  children,
  show,
  animation = 'fade',
  duration = 300,
  className,
}) => {
  const [shouldRender, setShouldRender] = useState(show);
  const [isAnimating, setIsAnimating] = useState(false);

  useEffect(() => {
    if (show) {
      setShouldRender(true);
      setTimeout(() => setIsAnimating(true), 10);
    } else {
      setIsAnimating(false);
      setTimeout(() => setShouldRender(false), duration);
    }
  }, [show, duration]);

  if (!shouldRender) return null;

  const animationClasses = {
    fade: isAnimating ? 'opacity-100' : 'opacity-0',
    slide: isAnimating
      ? 'translate-x-0 opacity-100'
      : '-translate-x-full opacity-0',
    scale: isAnimating
      ? 'scale-100 opacity-100'
      : 'scale-95 opacity-0',
    slideUp: isAnimating
      ? 'translate-y-0 opacity-100'
      : 'translate-y-4 opacity-0',
    slideDown: isAnimating
      ? 'translate-y-0 opacity-100'
      : '-translate-y-4 opacity-0',
  };

  return (
    <div
      className={clsx(
        'transition-all',
        animationClasses[animation],
        className
      )}
      style={{ transitionDuration: `${duration}ms` }}
    >
      {children}
    </div>
  );
};


