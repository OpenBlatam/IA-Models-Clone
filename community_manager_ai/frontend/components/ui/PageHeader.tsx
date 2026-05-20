'use client';

import { ReactNode } from 'react';
import { Breadcrumbs } from './Breadcrumbs';
import { cn } from '@/lib/utils';

interface PageHeaderProps {
  title: string;
  description?: string;
  breadcrumbs?: boolean;
  actions?: ReactNode;
  className?: string;
}

export const PageHeader = ({
  title,
  description,
  breadcrumbs = true,
  actions,
  className,
}: PageHeaderProps) => {
  return (
    <div className={cn('mb-6', className)}>
      {breadcrumbs && <Breadcrumbs className="mb-4" />}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-gray-100">{title}</h1>
          {description && (
            <p className="mt-2 text-gray-600 dark:text-gray-400">{description}</p>
          )}
        </div>
        {actions && <div className="flex items-center gap-2">{actions}</div>}
      </div>
    </div>
  );
};



