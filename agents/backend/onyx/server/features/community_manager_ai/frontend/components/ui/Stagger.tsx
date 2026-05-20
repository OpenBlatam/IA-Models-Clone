'use client';

import { ReactNode } from 'react';
import { motion } from 'framer-motion';
import { cn } from '@/lib/utils';

interface StaggerProps {
  children: ReactNode[];
  delay?: number;
  duration?: number;
  className?: string;
}

export const Stagger = ({
  children,
  delay = 0.1,
  duration = 0.3,
  className,
}: StaggerProps) => {
  return (
    <div className={cn(className)}>
      {children.map((child, index) => (
        <motion.div
          key={index}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration, delay: index * delay }}
        >
          {child}
        </motion.div>
      ))}
    </div>
  );
};



