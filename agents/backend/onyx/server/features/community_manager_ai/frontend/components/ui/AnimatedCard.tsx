'use client';

import { motion } from 'framer-motion';
import { Card, CardContent, CardHeader, CardTitle } from './Card';
import { HTMLAttributes } from 'react';

interface AnimatedCardProps extends HTMLAttributes<HTMLDivElement> {
  children: React.ReactNode;
  delay?: number;
}

export const AnimatedCard = ({ children, delay = 0, className, ...props }: AnimatedCardProps) => {
  const { onAnimationStart, onAnimationEnd, ...motionProps } = props as any;
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, delay }}
      className={className}
      {...motionProps}
    >
      {children}
    </motion.div>
  );
};

