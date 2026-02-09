'use client';

import { ReactNode } from 'react';
import { cn } from '@/lib/utils';

interface ListProps {
  children: ReactNode;
  className?: string;
  variant?: 'default' | 'bordered' | 'divided';
}

interface ListItemProps {
  children: ReactNode;
  className?: string;
  onClick?: () => void;
  selected?: boolean;
}

const variantClasses = {
  default: 'space-y-1',
  bordered: 'space-y-2 border border-gray-200 dark:border-gray-700 rounded-lg p-2',
  divided: 'divide-y divide-gray-200 dark:divide-gray-700',
};

export const List = ({ children, className, variant = 'default' }: ListProps) => {
  return (
    <ul className={cn(variantClasses[variant], className)} role="list">
      {children}
    </ul>
  );
};

export const ListItem = ({ children, className, onClick, selected }: ListItemProps) => {
  return (
    <li
      onClick={onClick}
      className={cn(
        'px-3 py-2 rounded-md transition-colors',
        onClick && 'cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-800',
        selected && 'bg-primary-50 dark:bg-primary-900/20',
        className
      )}
      role={onClick ? 'button' : 'listitem'}
      tabIndex={onClick ? 0 : undefined}
      onKeyDown={onClick ? (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          onClick();
        }
      } : undefined}
    >
      {children}
    </li>
  );
};



