'use client';

import { useEffect, useRef } from 'react';
import { motion } from 'framer-motion';
import { cn } from '@/lib/utils/cn';

interface ParticleBackgroundProps {
  count?: number;
  className?: string;
  color?: string;
  speed?: number;
  size?: number;
}

export default function ParticleBackground({
  count = 50,
  className,
  color = '#0062cc',
  speed = 1,
  size = 2,
}: ParticleBackgroundProps) {
  const particles = Array.from({ length: count }, (_, i) => ({
    id: i,
    x: Math.random() * 100,
    y: Math.random() * 100,
    duration: 10 + Math.random() * 20,
    delay: Math.random() * 5,
  }));

  return (
    <div className={cn('absolute inset-0 overflow-hidden pointer-events-none', className)}>
      {particles.map((particle) => (
        <motion.div
          key={particle.id}
          className="absolute rounded-full"
          style={{
            width: size,
            height: size,
            backgroundColor: color,
            opacity: 0.3,
            left: `${particle.x}%`,
            top: `${particle.y}%`,
          }}
          animate={{
            y: [0, -100, 0],
            x: [0, Math.random() * 50 - 25, 0],
            opacity: [0.3, 0.6, 0.3],
          }}
          transition={{
            duration: particle.duration / speed,
            delay: particle.delay,
            repeat: Infinity,
            ease: 'easeInOut',
          }}
        />
      ))}
    </div>
  );
}

interface FloatingElementsProps {
  children: React.ReactNode;
  count?: number;
  className?: string;
}

export function FloatingElements({ children, count = 3, className }: FloatingElementsProps) {
  const elements = Array.from({ length: count }, (_, i) => i);

  return (
    <div className={cn('relative', className)}>
      {elements.map((i) => (
        <motion.div
          key={i}
          className="absolute"
          animate={{
            y: [0, -20, 0],
            rotate: [0, 5, 0],
            opacity: [0.5, 1, 0.5],
          }}
          transition={{
            duration: 3 + i,
            delay: i * 0.5,
            repeat: Infinity,
            ease: 'easeInOut',
          }}
          style={{
            left: `${20 + i * 30}%`,
            top: `${10 + i * 20}%`,
          }}
        >
          {children}
        </motion.div>
      ))}
    </div>
  );
}



