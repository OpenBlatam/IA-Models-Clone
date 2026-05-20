'use client';

import React, { memo } from 'react';
import { LucideIcon } from 'lucide-react';
import { clsx } from 'clsx';

interface PageHeaderProps {
  title: string;
  description?: string;
  icon?: LucideIcon;
  badge?: React.ReactNode;
  actions?: React.ReactNode;
  className?: string;
}

export const PageHeader: React.FC<PageHeaderProps> = memo(({
  title,
  description,
  icon: Icon,
  badge,
  actions,
  className = '',
}) => {
  return (
    <div className={clsx('mb-8', className)}>
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center space-x-3">
          {Icon && (
            <Icon className="h-8 w-8 text-primary-600 dark:text-primary-400" />
          )}
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
            {title}
          </h1>
          {badge && <div className="ml-2">{badge}</div>}
        </div>
        {actions && <div className="flex items-center gap-2">{actions}</div>}
      </div>
      {description && (
        <p className="text-gray-600 dark:text-gray-400">{description}</p>
      )}
    </div>
  );
});

PageHeader.displayName = 'PageHeader';

