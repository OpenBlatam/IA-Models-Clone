'use client';

import { ReactNode, useRef, useEffect, useState } from 'react';
import { motion, useScroll, useTransform, MotionValue } from 'framer-motion';
import { cn } from '@/lib/utils/cn';

interface ParallaxSectionProps {
  children: ReactNode;
  speed?: number;
  direction?: 'up' | 'down' | 'left' | 'right';
  className?: string;
  offset?: [string, string];
}

function useParallax(value: MotionValue<number>, speed: number, direction: 'up' | 'down' | 'left' | 'right') {
  const multiplier = direction === 'up' || direction === 'left' ? -1 : 1;
  return useTransform(value, [0, 1], [0, speed * multiplier]);
}

export default function ParallaxSection({
  children,
  speed = 0.5,
  direction = 'up',
  className,
  offset = ['start end', 'end start'],
}: ParallaxSectionProps) {
  const ref = useRef<HTMLDivElement>(null);
  const { scrollYProgress } = useScroll({
    target: ref,
    offset,
  });

  const y = useParallax(scrollYProgress, 100 * speed, direction);
  const x = useParallax(scrollYProgress, 100 * speed, direction === 'left' || direction === 'right' ? direction : 'right');

  return (
    <div ref={ref} className={cn('relative', className)}>
      <motion.div
        style={{
          y: direction === 'up' || direction === 'down' ? y : 0,
          x: direction === 'left' || direction === 'right' ? x : 0,
        }}
      >
        {children}
      </motion.div>
    </div>
  );
}

interface ParallaxImageProps {
  src: string;
  alt: string;
  speed?: number;
  className?: string;
}

export function ParallaxImage({ src, alt, speed = 0.5, className }: ParallaxImageProps) {
  const ref = useRef<HTMLDivElement>(null);
  const { scrollYProgress } = useScroll({
    target: ref,
    offset: ['start end', 'end start'],
  });

  const y = useTransform(scrollYProgress, [0, 1], ['0%', `${50 * speed}%`]);
  const opacity = useTransform(scrollYProgress, [0, 0.5, 1], [0.3, 1, 0.3]);
  const scale = useTransform(scrollYProgress, [0, 0.5, 1], [1, 1.1, 1]);

  return (
    <div ref={ref} className={cn('relative overflow-hidden', className)}>
      <motion.img
        src={src}
        alt={alt}
        style={{ y, opacity, scale }}
        className="w-full h-full object-cover"
      />
    </div>
  );
}



