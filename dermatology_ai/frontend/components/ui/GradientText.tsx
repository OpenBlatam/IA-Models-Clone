'use client';

import React from 'react';
import { clsx } from 'clsx';

interface GradientTextProps {
  children: React.ReactNode;
  gradient?: string;
  className?: string;
}

export const GradientText: React.FC<GradientTextProps> = ({
  children,
  gradient = 'from-primary-600 via-purple-600 to-pink-600',
  className,
}) => {
  return (
    <span
      className={clsx(
        'bg-gradient-to-r bg-clip-text text-transparent',
        gradient,
        className
      )}
    >
      {children}
    </span>
  );
};


