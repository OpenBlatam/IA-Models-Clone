'use client';

import { HTMLAttributes } from 'react';

interface SectionProps extends HTMLAttributes<HTMLElement> {
  title?: string;
  description?: string;
  spacing?: 'none' | 'sm' | 'md' | 'lg';
}

const spacingClasses = {
  none: '',
  sm: 'py-4',
  md: 'py-8',
  lg: 'py-12',
};

export function Section({
  children,
  title,
  description,
  spacing = 'md',
  className = '',
  ...props
}: SectionProps) {
  return (
    <section className={`${spacingClasses[spacing]} ${className}`} {...props}>
      {(title || description) && (
        <div className="mb-6">
          {title && (
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
              {title}
            </h2>
          )}
          {description && (
            <p className="text-gray-600 dark:text-gray-400">{description}</p>
          )}
        </div>
      )}
      {children}
    </section>
  );
}

