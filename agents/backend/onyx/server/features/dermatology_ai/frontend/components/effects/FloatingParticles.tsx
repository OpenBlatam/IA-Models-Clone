'use client';

import React, { useMemo, memo } from 'react';

interface FloatingParticlesProps {
  count?: number;
  colors?: string[];
  className?: string;
}

export const FloatingParticles: React.FC<FloatingParticlesProps> = memo(({
  count = 30,
  colors = [
    'rgba(59, 130, 246, 0.3)',
    'rgba(147, 51, 234, 0.3)',
    'rgba(236, 72, 153, 0.3)',
  ],
  className = '',
}) => {
  const particles = useMemo(
    () =>
      Array.from({ length: count }, (_, i) => ({
        id: i,
        left: Math.random() * 100,
        top: Math.random() * 100,
        size: 2 + Math.random() * 4,
        color: colors[i % colors.length],
        delay: Math.random() * 5,
        duration: 5 + Math.random() * 5,
        glow: 4 + Math.random() * 4,
      })),
    [count, colors]
  );

  return (
    <div className={`absolute inset-0 overflow-hidden pointer-events-none ${className}`}>
      {particles.map((particle) => (
        <div
          key={particle.id}
          className="absolute rounded-full animate-float"
          style={{
            left: `${particle.left}%`,
            top: `${particle.top}%`,
            width: `${particle.size}px`,
            height: `${particle.size}px`,
            background: particle.color,
            animationDelay: `${particle.delay}s`,
            animationDuration: `${particle.duration}s`,
            boxShadow: `0 0 ${particle.glow}px currentColor`,
          }}
        />
      ))}
    </div>
  );
});

FloatingParticles.displayName = 'FloatingParticles';

