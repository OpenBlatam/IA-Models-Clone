'use client';

import { motion } from 'framer-motion';
import { LucideIcon } from 'lucide-react';
import { cn } from '@/lib/utils/cn';
import { Button } from './Button';

interface EmptyStateProps {
  icon?: LucideIcon;
  title: string;
  description?: string;
  action?: {
    label: string;
    onClick: () => void;
  };
  className?: string;
  size?: 'sm' | 'md' | 'lg';
}

export default function EmptyState({
  icon: Icon,
  title,
  description,
  action,
  className,
  size = 'md',
}: EmptyStateProps) {
  const iconSizes = {
    sm: 'w-12 h-12', // 48px
    md: 'w-16 h-16', // 64px
    lg: 'w-24 h-24', // 96px
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className={cn('flex flex-col items-center justify-center text-center', 'py-tesla-2xl md:py-tesla-3xl px-tesla-md', className)}
      style={{ paddingTop: 'var(--tesla-spacing-2xl)', paddingBottom: 'var(--tesla-spacing-2xl)', paddingLeft: 'var(--tesla-spacing-md)', paddingRight: 'var(--tesla-spacing-md)' }}
    >
      {Icon && (
        <motion.div
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          transition={{ delay: 0.1, type: 'spring' }}
          className={cn('mb-tesla-lg text-tesla-gray-light', iconSizes[size])}
        >
          <Icon className="w-full h-full" />
        </motion.div>
      )}
      
      <motion.h3
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.2 }}
        className="text-xl md:text-2xl font-semibold text-tesla-black mb-tesla-sm"
      >
        {title}
      </motion.h3>
      
      {description && (
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.3 }}
          className="text-tesla-gray-dark max-w-md mb-tesla-lg"
        >
          {description}
        </motion.p>
      )}
      
      {action && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
        >
          <Button onClick={action.onClick} variant="primary">
            {action.label}
          </Button>
        </motion.div>
      )}
    </motion.div>
  );
}

