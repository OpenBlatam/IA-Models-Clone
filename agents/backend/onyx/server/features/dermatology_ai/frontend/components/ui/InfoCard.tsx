'use client';

import React, { memo } from 'react';
import { clsx } from 'clsx';

interface InfoCardProps {
  title: string;
  description: string;
  hoverColor?: 'blue' | 'purple' | 'pink' | 'green' | 'yellow' | 'indigo';
  className?: string;
}

const hoverColorClasses = {
  blue: 'group-hover:text-blue-600 dark:group-hover:text-blue-400',
  purple: 'group-hover:text-purple-600 dark:group-hover:text-purple-400',
  pink: 'group-hover:text-pink-600 dark:group-hover:text-pink-400',
  green: 'group-hover:text-green-600 dark:group-hover:text-green-400',
  yellow: 'group-hover:text-yellow-600 dark:group-hover:text-yellow-400',
  indigo: 'group-hover:text-indigo-600 dark:group-hover:text-indigo-400',
};

export const InfoCard: React.FC<InfoCardProps> = memo(({
  title,
  description,
  hoverColor = 'blue',
  className,
}) => {
  return (
    <div
      className={clsx(
        'p-6 rounded-xl bg-white/50 dark:bg-gray-900/50 backdrop-blur-sm',
        'border border-gray-200/50 dark:border-gray-800/50',
        'hover:bg-white/70 dark:hover:bg-gray-900/70',
        'transition-all duration-300 group',
        className
      )}
    >
      <h3
        className={clsx(
          'text-lg font-semibold text-gray-900 dark:text-white mb-2',
          'transition-colors',
          hoverColorClasses[hoverColor]
        )}
      >
        {title}
      </h3>
      <p className="text-sm text-gray-600 dark:text-gray-400">{description}</p>
    </div>
  );
});

InfoCard.displayName = 'InfoCard';



