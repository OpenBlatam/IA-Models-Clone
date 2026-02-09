'use client';

import React from 'react';
import { clsx } from 'clsx';

interface WatermarkProps {
  text: string;
  opacity?: number;
  fontSize?: number;
  rotation?: number;
  className?: string;
}

export const Watermark: React.FC<WatermarkProps> = ({
  text,
  opacity = 0.1,
  fontSize = 48,
  rotation = -45,
  className,
}) => {
  return (
    <div
      className={clsx('absolute inset-0 pointer-events-none overflow-hidden', className)}
      style={{
        opacity,
      }}
    >
      <div
        className="absolute inset-0 flex items-center justify-center"
        style={{
          transform: `rotate(${rotation}deg)`,
          fontSize: `${fontSize}px`,
          fontWeight: 'bold',
          color: 'currentColor',
          whiteSpace: 'nowrap',
        }}
      >
        {text}
      </div>
    </div>
  );
};


