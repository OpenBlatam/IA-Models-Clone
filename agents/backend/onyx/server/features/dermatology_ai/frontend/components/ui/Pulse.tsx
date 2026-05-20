'use client';

import React from 'react';
import { clsx } from 'clsx';

interface PulseProps {
  children: React.ReactNode;
  className?: string;
  size?: 'sm' | 'md' | 'lg';
}

export const Pulse: React.FC<PulseProps> = ({
  children,
  className,
  size = 'md',
}) => {
  const sizeClasses = {
    sm: 'h-2 w-2',
    md: 'h-3 w-3',
    lg: 'h-4 w-4',
  };

  return (
    <div className={clsx('relative inline-flex items-center', className)}>
      {children}
      <span
        className={clsx(
          'absolute inline-flex rounded-full bg-primary-400 opacity-75 animate-ping',
          sizeClasses[size]
        )}
        style={{
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
        }}
      />
    </div>
  );
};


