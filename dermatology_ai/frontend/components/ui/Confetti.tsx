'use client';

import React, { useEffect, useState } from 'react';
import { clsx } from 'clsx';

interface ConfettiProps {
  active: boolean;
  duration?: number;
  colors?: string[];
  particleCount?: number;
}

export const Confetti: React.FC<ConfettiProps> = ({
  active,
  duration = 3000,
  colors = ['#0ea5e9', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981'],
  particleCount = 50,
}) => {
  const [particles, setParticles] = useState<Array<{
    id: number;
    left: number;
    delay: number;
    duration: number;
    color: string;
  }>>([]);

  useEffect(() => {
    if (active) {
      const newParticles = Array.from({ length: particleCount }, (_, i) => ({
        id: i,
        left: Math.random() * 100,
        delay: Math.random() * 500,
        duration: duration + Math.random() * 1000,
        color: colors[Math.floor(Math.random() * colors.length)],
      }));
      setParticles(newParticles);

      const timer = setTimeout(() => {
        setParticles([]);
      }, duration + 2000);

      return () => clearTimeout(timer);
    }
  }, [active, duration, colors, particleCount]);

  if (!active || particles.length === 0) return null;

  return (
    <div className="fixed inset-0 pointer-events-none z-50 overflow-hidden">
      {particles.map((particle) => (
        <div
          key={particle.id}
          className="absolute w-2 h-2 rounded-full"
          style={{
            left: `${particle.left}%`,
            backgroundColor: particle.color,
            animation: `confetti-fall ${particle.duration}ms ease-out ${particle.delay}ms forwards`,
          }}
        />
      ))}
      <style jsx>{`
        @keyframes confetti-fall {
          0% {
            transform: translateY(-100vh) rotate(0deg);
            opacity: 1;
          }
          100% {
            transform: translateY(100vh) rotate(720deg);
            opacity: 0;
          }
        }
      `}</style>
    </div>
  );
};


