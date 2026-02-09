'use client';

import React, { useEffect, useState } from 'react';
import { clsx } from 'clsx';

interface SpotlightProps {
  children: React.ReactNode;
  className?: string;
  intensity?: number;
}

export const Spotlight: React.FC<SpotlightProps> = ({
  children,
  className,
  intensity = 0.3,
}) => {
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      setMousePosition({ x: e.clientX, y: e.clientY });
    };

    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, []);

  return (
    <div
      className={clsx('relative', className)}
      style={{
        background: `radial-gradient(circle at ${mousePosition.x}px ${mousePosition.y}px, rgba(14, 165, 233, ${intensity}) 0%, transparent 50%)`,
      }}
    >
      {children}
    </div>
  );
};


