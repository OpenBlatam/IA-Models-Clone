'use client';

import React, { useEffect, useState, useRef } from 'react';
import { useIntersectionObserver } from '@/hooks';

interface CountUpProps {
  end: number;
  start?: number;
  duration?: number;
  decimals?: number;
  prefix?: string;
  suffix?: string;
  className?: string;
}

export const CountUp: React.FC<CountUpProps> = ({
  end,
  start = 0,
  duration = 2000,
  decimals = 0,
  prefix = '',
  suffix = '',
  className,
}) => {
  const [count, setCount] = useState(start);
  const [elementRef, isIntersecting] = useIntersectionObserver({
    threshold: 0.5,
    triggerOnce: true,
  });
  const animationFrameRef = useRef<number>();
  const startTimeRef = useRef<number>();

  useEffect(() => {
    if (!isIntersecting) return;

    const startTime = performance.now();
    startTimeRef.current = startTime;

    const animate = (currentTime: number) => {
      if (!startTimeRef.current) return;

      const elapsed = currentTime - startTimeRef.current;
      const progress = Math.min(elapsed / duration, 1);

      const easeOutQuart = 1 - Math.pow(1 - progress, 4);
      const currentCount = start + (end - start) * easeOutQuart;

      setCount(currentCount);

      if (progress < 1) {
        animationFrameRef.current = requestAnimationFrame(animate);
      } else {
        setCount(end);
      }
    };

    animationFrameRef.current = requestAnimationFrame(animate);

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [isIntersecting, start, end, duration]);

  const formattedCount = count.toFixed(decimals);

  return (
    <span ref={elementRef as React.RefObject<HTMLSpanElement>} className={className}>
      {prefix}
      {formattedCount}
      {suffix}
    </span>
  );
};


