'use client';

import { HTMLAttributes } from 'react';
import { motion } from 'framer-motion';

interface CardProps extends HTMLAttributes<HTMLDivElement> {
  hover?: boolean;
  padding?: 'none' | 'sm' | 'md' | 'lg';
}

export function Card({ 
  children, 
  className = '', 
  hover = false,
  padding = 'md',
  ...props 
}: CardProps) {
  const paddingClasses = {
    none: '',
    sm: 'p-3',
    md: 'p-6',
    lg: 'p-8',
  };

  const baseClasses = `
    bg-white dark:bg-gray-800
    rounded-xl shadow-sm
    border border-gray-200 dark:border-gray-700
    ${paddingClasses[padding]}
    ${hover ? 'hover:shadow-md transition-shadow cursor-pointer' : ''}
    ${className}
  `;

  if (hover) {
    return (
      <motion.div
        whileHover={{ y: -2 }}
        className={baseClasses}
        {...props}
      >
        {children}
      </motion.div>
    );
  }

  return (
    <div className={baseClasses} {...props}>
      {children}
    </div>
  );
}

