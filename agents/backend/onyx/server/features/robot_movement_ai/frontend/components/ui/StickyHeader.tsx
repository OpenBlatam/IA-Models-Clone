'use client';

import { ReactNode, useEffect, useState } from 'react';
import { motion, useScroll, useTransform } from 'framer-motion';
import { cn } from '@/lib/utils/cn';

interface StickyHeaderProps {
  children: ReactNode;
  className?: string;
  threshold?: number;
}

export default function StickyHeader({ children, className, threshold = 100 }: StickyHeaderProps) {
  const [isScrolled, setIsScrolled] = useState(false);
  const { scrollY } = useScroll();
  const opacity = useTransform(scrollY, [0, threshold], [1, 0.95]);
  const backgroundColor = useTransform(
    scrollY,
    [0, threshold],
    ['rgba(255, 255, 255, 0)', 'rgba(255, 255, 255, 0.95)']
  );
  const backdropBlur = useTransform(scrollY, [0, threshold], ['blur(0px)', 'blur(20px)']);

  useEffect(() => {
    const unsubscribe = scrollY.on('change', (latest) => {
      setIsScrolled(latest > threshold);
    });
    return () => unsubscribe();
  }, [scrollY, threshold]);

  return (
    <motion.header
      style={{ opacity, backgroundColor, backdropFilter: backdropBlur }}
      className={cn(
        'sticky top-0 z-50 w-full border-b border-gray-200/50 transition-all duration-300',
        isScrolled && 'shadow-sm',
        className
      )}
    >
      {children}
    </motion.header>
  );
}



