'use client';

import React from 'react';
import { clsx } from 'clsx';

interface GlassmorphismProps {
  children: React.ReactNode;
  blur?: number;
  opacity?: number;
  className?: string;
}

export const Glassmorphism: React.FC<GlassmorphismProps> = ({
  children,
  blur = 10,
  opacity = 0.1,
  className,
}) => {
  return (
    <div
      className={clsx(
        'backdrop-blur-md bg-white/10 dark:bg-gray-900/10 border border-white/20 dark:border-gray-700/20 rounded-lg',
        className
      )}
      style={{
        backdropFilter: `blur(${blur}px)`,
        backgroundColor: `rgba(255, 255, 255, ${opacity})`,
      }}
    >
      {children}
    </div>
  );
};


