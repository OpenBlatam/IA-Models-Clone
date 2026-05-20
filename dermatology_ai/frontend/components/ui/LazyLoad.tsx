'use client';

import React from 'react';
import { useIntersectionObserver } from '@/hooks';

interface LazyLoadProps {
  children: React.ReactNode;
  fallback?: React.ReactNode;
  threshold?: number;
  rootMargin?: string;
  triggerOnce?: boolean;
}

export const LazyLoad: React.FC<LazyLoadProps> = ({
  children,
  fallback,
  threshold = 0.1,
  rootMargin = '50px',
  triggerOnce = true,
}) => {
  const [elementRef, isIntersecting] = useIntersectionObserver({
    threshold,
    rootMargin,
    triggerOnce,
  });

  return (
    <div ref={elementRef as React.RefObject<HTMLDivElement>}>
      {isIntersecting ? children : fallback}
    </div>
  );
};


