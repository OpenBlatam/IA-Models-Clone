'use client';

import { ReactNode } from 'react';
import { useInView } from 'react-intersection-observer';
import { motion } from 'framer-motion';

interface LazyLoadProps {
  children: ReactNode;
  className?: string;
  fallback?: ReactNode;
  threshold?: number;
}

export default function LazyLoad({ 
  children, 
  className = '', 
  fallback = null,
  threshold = 0.1 
}: LazyLoadProps) {
  const { ref, inView } = useInView({
    threshold,
    triggerOnce: true,
  });

  return (
    <div ref={ref} className={className}>
      {inView ? (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
        >
          {children}
        </motion.div>
      ) : (
        fallback
      )}
    </div>
  );
}



