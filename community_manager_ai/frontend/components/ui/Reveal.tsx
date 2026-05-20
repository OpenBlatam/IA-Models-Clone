'use client';

import { ReactNode } from 'react';
import { motion } from 'framer-motion';
import { useIntersectionObserver } from '@/hooks/useIntersectionObserver';
import { cn } from '@/lib/utils';

interface RevealProps {
  children: ReactNode;
  direction?: 'up' | 'down' | 'left' | 'right' | 'fade';
  delay?: number;
  duration?: number;
  className?: string;
}

const directionVariants = {
  up: { initial: { y: 50, opacity: 0 }, animate: { y: 0, opacity: 1 } },
  down: { initial: { y: -50, opacity: 0 }, animate: { y: 0, opacity: 1 } },
  left: { initial: { x: 50, opacity: 0 }, animate: { x: 0, opacity: 1 } },
  right: { initial: { x: -50, opacity: 0 }, animate: { x: 0, opacity: 1 } },
  fade: { initial: { opacity: 0 }, animate: { opacity: 1 } },
};

export const Reveal = ({
  children,
  direction = 'up',
  delay = 0,
  duration = 0.5,
  className,
}: RevealProps) => {
  const [ref, isIntersecting] = useIntersectionObserver<HTMLDivElement>({ threshold: 0.1 });
  const variant = directionVariants[direction];

  return (
    <motion.div
      ref={ref}
      initial={variant.initial}
      animate={isIntersecting ? variant.animate : variant.initial}
      transition={{ duration, delay }}
      className={cn(className)}
    >
      {children}
    </motion.div>
  );
};

