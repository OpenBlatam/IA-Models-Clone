'use client';

import { ReactNode } from 'react';
import { motion } from 'framer-motion';
import { useIntersectionObserver } from '@/hooks/useIntersectionObserver';
import { cn } from '@/lib/utils';

interface BlurFadeProps {
  children: ReactNode;
  delay?: number;
  duration?: number;
  className?: string;
}

export const BlurFade = ({
  children,
  delay = 0,
  duration = 0.6,
  className,
}: BlurFadeProps) => {
  const [ref, isIntersecting] = useIntersectionObserver<HTMLDivElement>({ threshold: 0.1 });

  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, filter: 'blur(10px)' }}
      animate={
        isIntersecting
          ? { opacity: 1, filter: 'blur(0px)' }
          : { opacity: 0, filter: 'blur(10px)' }
      }
      transition={{ duration, delay }}
      className={cn(className)}
    >
      {children}
    </motion.div>
  );
};

