'use client';

import React from 'react';
import { clsx } from 'clsx';

interface BlurBackgroundProps {
  children: React.ReactNode;
  blur?: number;
  className?: string;
}

export const BlurBackground: React.FC<BlurBackgroundProps> = ({
  children,
  blur = 8,
  className,
}) => {
  return (
    <div
      className={clsx('backdrop-blur-sm', className)}
      style={{
        backdropFilter: `blur(${blur}px)`,
      }}
    >
      {children}
    </div>
  );
};


