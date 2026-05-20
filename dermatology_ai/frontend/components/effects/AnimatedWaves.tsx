'use client';

import React, { memo } from 'react';

interface AnimatedWavesProps {
  height?: number;
  className?: string;
}

export const AnimatedWaves: React.FC<AnimatedWavesProps> = memo(({
  height = 128,
  className = '',
}) => {
  return (
    <div
      className={`absolute bottom-0 left-0 right-0 overflow-hidden pointer-events-none ${className}`}
      style={{ height: `${height}px` }}
    >
      <svg
        className="absolute bottom-0 w-full h-full"
        viewBox="0 0 1200 120"
        preserveAspectRatio="none"
      >
        <path
          d="M0,60 Q300,20 600,60 T1200,60 L1200,120 L0,120 Z"
          fill="url(#waveGradient)"
          className="animate-wave"
        />
        <defs>
          <linearGradient id="waveGradient" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="rgba(59, 130, 246, 0.1)" />
            <stop offset="50%" stopColor="rgba(147, 51, 234, 0.1)" />
            <stop offset="100%" stopColor="rgba(236, 72, 153, 0.1)" />
          </linearGradient>
        </defs>
      </svg>
    </div>
  );
});

AnimatedWaves.displayName = 'AnimatedWaves';

