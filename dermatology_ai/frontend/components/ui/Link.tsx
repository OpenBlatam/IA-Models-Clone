'use client';

import React from 'react';
import NextLink from 'next/link';
import { clsx } from 'clsx';

interface LinkProps {
  href: string;
  children: React.ReactNode;
  className?: string;
  external?: boolean;
  variant?: 'default' | 'primary' | 'secondary';
}

export const Link: React.FC<LinkProps> = ({
  href,
  children,
  className,
  external = false,
  variant = 'default',
}) => {
  const variantClasses = {
    default: 'text-gray-900 dark:text-gray-100 hover:text-primary-600 dark:hover:text-primary-400',
    primary: 'text-primary-600 dark:text-primary-400 hover:text-primary-700 dark:hover:text-primary-300',
    secondary: 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-100',
  };

  const baseClasses = 'underline-offset-4 hover:underline transition-colors';

  if (external || href.startsWith('http')) {
    return (
      <a
        href={href}
        target="_blank"
        rel="noopener noreferrer"
        className={clsx(baseClasses, variantClasses[variant], className)}
      >
        {children}
      </a>
    );
  }

  return (
    <NextLink
      href={href}
      className={clsx(baseClasses, variantClasses[variant], className)}
    >
      {children}
    </NextLink>
  );
};
