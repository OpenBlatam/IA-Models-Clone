'use client';

import { ReactNode } from 'react';
import { cn } from '@/lib/utils';

interface GlassCardProps {
  children: ReactNode;
  className?: string;
  blur?: 'sm' | 'md' | 'lg';
}

const blurClasses = {
  sm: 'backdrop-blur-sm',
  md: 'backdrop-blur-md',
  lg: 'backdrop-blur-lg',
};

export const GlassCard = ({
  children,
  className,
  blur = 'md',
}: GlassCardProps) => {
  return (
    <div
      className={cn(
        'rounded-lg border border-white/20 bg-white/10 dark:bg-gray-900/10',
        blurClasses[blur],
        'shadow-lg',
        className
      )}
    >
      {children}
    </div>
  );
};



