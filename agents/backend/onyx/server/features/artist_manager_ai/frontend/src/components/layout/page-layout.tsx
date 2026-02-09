'use client';

import { ReactNode } from 'react';
import { cn } from '@/lib/utils';

interface PageLayoutProps {
  children: ReactNode;
  className?: string;
}

const PageLayout = ({ children, className }: PageLayoutProps) => {
  return (
    <div className={cn('min-h-screen bg-gray-50 p-6', className)}>
      <div className="max-w-7xl mx-auto">{children}</div>
    </div>
  );
};

export { PageLayout };

