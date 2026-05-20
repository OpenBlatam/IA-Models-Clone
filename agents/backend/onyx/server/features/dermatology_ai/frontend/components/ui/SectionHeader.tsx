'use client';

import React, { memo } from 'react';
import { clsx } from 'clsx';

interface SectionHeaderProps {
  title: string;
  description?: string;
  className?: string;
  align?: 'left' | 'center' | 'right';
  size?: 'sm' | 'md' | 'lg';
}

const sizeClasses = {
  sm: {
    title: 'text-2xl md:text-3xl',
    description: 'text-base',
  },
  md: {
    title: 'text-3xl md:text-4xl',
    description: 'text-lg',
  },
  lg: {
    title: 'text-4xl md:text-5xl',
    description: 'text-xl',
  },
};

const alignClasses = {
  left: 'text-left',
  center: 'text-center',
  right: 'text-right',
};

export const SectionHeader: React.FC<SectionHeaderProps> = memo(({
  title,
  description,
  className,
  align = 'center',
  size = 'md',
}) => {
  return (
    <div className={clsx('mb-12', alignClasses[align], className)}>
      <h2
        className={clsx(
          'font-bold text-gray-900 dark:text-white mb-4',
          sizeClasses[size].title
        )}
      >
        {title}
      </h2>
      {description && (
        <p
          className={clsx(
            'text-gray-600 dark:text-gray-400 max-w-2xl',
            align === 'center' && 'mx-auto',
            align === 'right' && 'ml-auto',
            sizeClasses[size].description
          )}
        >
          {description}
        </p>
      )}
    </div>
  );
});

SectionHeader.displayName = 'SectionHeader';



