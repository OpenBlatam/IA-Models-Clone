'use client';

import { ReactNode, useEffect, useState } from 'react';
import { cn } from '@/lib/utils';

interface SpotlightProps {
  children: ReactNode;
  className?: string;
  size?: 'sm' | 'md' | 'lg';
  intensity?: 'low' | 'medium' | 'high';
}

const sizeClasses = {
  sm: 'w-32 h-32',
  md: 'w-64 h-64',
  lg: 'w-96 h-96',
};

const intensityClasses = {
  low: 'opacity-20',
  medium: 'opacity-40',
  high: 'opacity-60',
};

export const Spotlight = ({
  children,
  className,
  size = 'md',
  intensity = 'medium',
}: SpotlightProps) => {
  const [position, setPosition] = useState({ x: 50, y: 50 });

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      const x = (e.clientX / window.innerWidth) * 100;
      const y = (e.clientY / window.innerHeight) * 100;
      setPosition({ x, y });
    };

    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, []);

  return (
    <div className={cn('relative overflow-hidden', className)}>
      <div
        className={cn(
          'absolute rounded-full blur-3xl pointer-events-none transition-all duration-300',
          sizeClasses[size],
          intensityClasses[intensity],
          'bg-primary-500'
        )}
        style={{
          left: `${position.x}%`,
          top: `${position.y}%`,
          transform: 'translate(-50%, -50%)',
        }}
      />
      {children}
    </div>
  );
};



