'use client';

import React, { memo } from 'react';
import { clsx } from 'clsx';

interface ResultCardProps {
  children: React.ReactNode;
  variant?: 'analysis' | 'recommendations';
  className?: string;
}

export const ResultCard: React.FC<ResultCardProps> = memo(({
  children,
  variant = 'analysis',
  className,
}) => {
  const gradientColor = variant === 'analysis' 
    ? 'bg-blue-500/5 dark:bg-blue-500/10' 
    : 'bg-purple-500/5 dark:bg-purple-500/10';

  return (
    <div className={clsx(
      'bg-white/90 dark:bg-gray-900/90 backdrop-blur-md rounded-xl shadow-2xl border border-gray-200/50 dark:border-gray-800/50 p-6 lg:p-8 relative overflow-hidden group',
      className
    )}>
      <div className={clsx('absolute top-0 right-0 w-64 h-64 rounded-full blur-3xl', gradientColor)} />
      <div className="relative z-10">
        {children}
      </div>
    </div>
  );
});

ResultCard.displayName = 'ResultCard';



