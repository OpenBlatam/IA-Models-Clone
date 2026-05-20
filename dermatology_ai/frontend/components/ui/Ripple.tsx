'use client';

import React, { useState, useRef } from 'react';
import { clsx } from 'clsx';

interface RippleProps {
  children: React.ReactNode;
  className?: string;
  color?: string;
}

export const Ripple: React.FC<RippleProps> = ({
  children,
  className,
  color = 'rgba(255, 255, 255, 0.5)',
}) => {
  const [ripples, setRipples] = useState<Array<{ id: number; x: number; y: number }>>([]);
  const containerRef = useRef<HTMLDivElement>(null);

  const handleClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!containerRef.current) return;

    const rect = containerRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const newRipple = {
      id: Date.now(),
      x,
      y,
    };

    setRipples((prev) => [...prev, newRipple]);

    setTimeout(() => {
      setRipples((prev) => prev.filter((ripple) => ripple.id !== newRipple.id));
    }, 600);
  };

  return (
    <div
      ref={containerRef}
      onClick={handleClick}
      className={clsx('relative overflow-hidden', className)}
    >
      {children}
      {ripples.map((ripple) => (
        <span
          key={ripple.id}
          className="absolute rounded-full bg-white/50 pointer-events-none animate-ripple"
          style={{
            left: `${ripple.x}px`,
            top: `${ripple.y}px`,
            transform: 'translate(-50%, -50%)',
            backgroundColor: color,
          }}
        />
      ))}
      <style jsx>{`
        @keyframes ripple {
          0% {
            width: 0;
            height: 0;
            opacity: 1;
          }
          100% {
            width: 200px;
            height: 200px;
            opacity: 0;
          }
        }
        .animate-ripple {
          animation: ripple 0.6s ease-out;
        }
      `}</style>
    </div>
  );
};


