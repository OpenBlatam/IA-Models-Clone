'use client';

import { ReactNode } from 'react';
import { motion } from 'framer-motion';
import { cn } from '@/lib/utils/cn';

interface ShimmerEffectProps {
  children: ReactNode;
  className?: string;
  duration?: number;
  gradient?: string;
}

export default function ShimmerEffect({
  children,
  className,
  duration = 2,
  gradient = 'linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent)',
}: ShimmerEffectProps) {
  return (
    <div className={cn('relative overflow-hidden', className)}>
      {children}
      <motion.div
        className="absolute inset-0 pointer-events-none"
        style={{
          background: gradient,
          backgroundSize: '200% 100%',
        }}
        animate={{
          x: ['-100%', '200%'],
        }}
        transition={{
          duration,
          repeat: Infinity,
          ease: 'linear',
        }}
      />
    </div>
  );
}

interface ShimmerTextProps {
  children: ReactNode;
  className?: string;
  duration?: number;
}

export function ShimmerText({ children, className, duration = 2 }: ShimmerTextProps) {
  return (
    <div className={cn('relative inline-block overflow-hidden', className)}>
      <span className="relative z-10">{children}</span>
      <motion.span
        className="absolute inset-0 bg-gradient-to-r from-transparent via-white/30 to-transparent"
        animate={{
          x: ['-100%', '200%'],
        }}
        transition={{
          duration,
          repeat: Infinity,
          ease: 'linear',
        }}
      />
    </div>
  );
}

interface GlowEffectProps {
  children: ReactNode;
  className?: string;
  color?: string;
  intensity?: number;
  size?: number;
}

export function GlowEffect({
  children,
  className,
  color = '#0062cc',
  intensity = 0.5,
  size = 100,
}: GlowEffectProps) {
  return (
    <div
      className={cn('relative', className)}
      style={{
        filter: `drop-shadow(0 0 ${size}px ${color}${Math.round(intensity * 255).toString(16)})`,
      }}
    >
      {children}
    </div>
  );
}



