'use client';

import { ReactNode } from 'react';
import { Button } from './Button';
import { cn } from '@/lib/utils';
import { motion } from 'framer-motion';

interface QuickAction {
  id: string;
  label: string;
  icon: ReactNode;
  onClick: () => void;
  variant?: 'primary' | 'secondary' | 'ghost';
  disabled?: boolean;
}

interface QuickActionsProps {
  actions: QuickAction[];
  className?: string;
}

export const QuickActions = ({ actions, className }: QuickActionsProps) => {
  if (actions.length === 0) return null;

  return (
    <div className={cn('flex flex-wrap items-center gap-2', className)}>
      {actions.map((action, index) => (
        <motion.div
          key={action.id}
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: index * 0.05 }}
        >
          <Button
            variant={action.variant || 'secondary'}
            size="sm"
            onClick={action.onClick}
            disabled={action.disabled}
            className="flex items-center gap-2"
            aria-label={action.label}
          >
            {action.icon}
            {action.label}
          </Button>
        </motion.div>
      ))}
    </div>
  );
};



