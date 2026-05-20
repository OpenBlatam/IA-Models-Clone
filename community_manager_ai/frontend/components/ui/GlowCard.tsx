'use client';

import { ReactNode } from 'react';
import { cn } from '@/lib/utils';

interface GlowCardProps {
  children: ReactNode;
  className?: string;
  glowColor?: 'primary' | 'success' | 'warning' | 'error';
  intensity?: 'low' | 'medium' | 'high';
}

const glowColors = {
  primary: 'shadow-primary-500/50',
  success: 'shadow-green-500/50',
  warning: 'shadow-yellow-500/50',
  error: 'shadow-red-500/50',
};

const intensityClasses = {
  low: 'shadow-lg',
  medium: 'shadow-xl',
  high: 'shadow-2xl',
};

export const GlowCard = ({
  children,
  className,
  glowColor = 'primary',
  intensity = 'medium',
}: GlowCardProps) => {
  return (
    <div
      className={cn(
        'rounded-lg border border-gray-200 dark:border-gray-700',
        'bg-white dark:bg-gray-800',
        glowColors[glowColor],
        intensityClasses[intensity],
        'transition-shadow duration-300',
        className
      )}
    >
      {children}
    </div>
  );
};



