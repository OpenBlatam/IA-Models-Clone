'use client';

import React, { memo } from 'react';
import { clsx } from 'clsx';

interface PageLayoutProps {
  children: React.ReactNode;
  className?: string;
  maxWidth?: 'sm' | 'md' | 'lg' | 'xl' | '2xl' | '7xl' | 'full';
}

const maxWidthClasses = {
  sm: 'max-w-sm',
  md: 'max-w-md',
  lg: 'max-w-lg',
  xl: 'max-w-xl',
  '2xl': 'max-w-2xl',
  '7xl': 'max-w-7xl',
  full: 'max-w-full',
};

export const PageLayout: React.FC<PageLayoutProps> = memo(({
  children,
  className = '',
  maxWidth = '7xl',
}) => {
  return (
    <div className={clsx('min-h-screen bg-gray-50 dark:bg-gray-900', className)}>
      <div className={clsx(maxWidthClasses[maxWidth], 'mx-auto px-4 sm:px-6 lg:px-8 py-8')}>
        {children}
      </div>
    </div>
  );
});

PageLayout.displayName = 'PageLayout';

