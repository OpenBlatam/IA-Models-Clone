'use client';

import React, { memo } from 'react';
import { clsx } from 'clsx';

interface StepCardProps {
  step: number | string;
  title: string;
  description: string;
  icon?: React.ReactNode;
  className?: string;
  iconColor?: string;
}

export const StepCard: React.FC<StepCardProps> = memo(({
  step,
  title,
  description,
  icon,
  className,
  iconColor = 'text-primary-600 dark:text-primary-400',
}) => {
  return (
    <div className={clsx('text-center group', className)}>
      {icon ? (
        <div className={clsx('inline-flex items-center justify-center w-16 h-16 rounded-full bg-primary-100 dark:bg-primary-900/50 mb-4 group-hover:scale-110 transition-transform duration-300', iconColor)}>
          {icon}
        </div>
      ) : (
        <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-primary-100 dark:bg-primary-900/50 mb-4 group-hover:scale-110 transition-transform duration-300">
          <span className="text-2xl font-bold text-primary-600 dark:text-primary-400">{step}</span>
        </div>
      )}
      <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">{title}</h3>
      <p className="text-sm text-gray-600 dark:text-gray-400">{description}</p>
    </div>
  );
});

StepCard.displayName = 'StepCard';



