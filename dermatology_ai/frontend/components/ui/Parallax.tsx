'use client';

import React, { useEffect, useRef, useState } from 'react';
import { useScrollPosition } from '@/hooks';

interface ParallaxProps {
  children: React.ReactNode;
  speed?: number;
  className?: string;
}

export const Parallax: React.FC<ParallaxProps> = ({
  children,
  speed = 0.5,
  className,
}) => {
  const { y } = useScrollPosition();
  const [offset, setOffset] = useState(0);
  const elementRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (elementRef.current) {
      const rect = elementRef.current.getBoundingClientRect();
      const elementTop = rect.top + window.scrollY;
      const scrollProgress = (y - elementTop + window.innerHeight) / (window.innerHeight + rect.height);
      setOffset(scrollProgress * 100 * speed);
    }
  }, [y, speed]);

  return (
    <div
      ref={elementRef}
      className={className}
      style={{
        transform: `translateY(${offset}px)`,
        transition: 'transform 0.1s ease-out',
      }}
    >
      {children}
    </div>
  );
};


