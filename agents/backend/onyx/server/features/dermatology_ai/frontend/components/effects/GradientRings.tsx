'use client';

import React, { memo } from 'react';

interface GradientRingsProps {
  count?: number;
  sizes?: number[];
  className?: string;
}

export const GradientRings: React.FC<GradientRingsProps> = memo(({
  count = 2,
  sizes = [800, 600],
  className = '',
}) => {
  const gradients = [
    'radial-gradient(circle, rgba(59, 130, 246, 0.05), rgba(147, 51, 234, 0.05), transparent)',
    'radial-gradient(circle, rgba(147, 51, 234, 0.05), rgba(236, 72, 153, 0.05), transparent)',
  ];

  return (
    <>
      {Array.from({ length: count }, (_, i) => (
        <div
          key={i}
          className={`absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 rounded-full pointer-events-none animate-rotate-slow ${className}`}
          style={{
            width: `${sizes[i] || sizes[0]}px`,
            height: `${sizes[i] || sizes[0]}px`,
            background: gradients[i % gradients.length],
            animationDirection: i % 2 === 1 ? 'reverse' : 'normal',
            animationDuration: i % 2 === 1 ? '25s' : '20s',
          }}
        />
      ))}
    </>
  );
});

GradientRings.displayName = 'GradientRings';

