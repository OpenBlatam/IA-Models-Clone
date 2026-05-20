'use client';

import React from 'react';
import { ChevronRight, Home } from 'lucide-react';
import Link from 'next/link';
import { clsx } from 'clsx';

interface BreadcrumbItem {
  label: string;
  href?: string;
  icon?: React.ReactNode;
}

interface BreadcrumbNavProps {
  items: BreadcrumbItem[];
  className?: string;
  separator?: React.ReactNode;
  showHome?: boolean;
}

export const BreadcrumbNav: React.FC<BreadcrumbNavProps> = ({
  items,
  className,
  separator = <ChevronRight className="h-4 w-4" />,
  showHome = true,
}) => {
  const allItems = showHome
    ? [{ label: 'Inicio', href: '/', icon: <Home className="h-4 w-4" /> }, ...items]
    : items;

  return (
    <nav
      className={clsx('flex items-center space-x-2 text-sm', className)}
      aria-label="Breadcrumb"
    >
      {allItems.map((item, index) => {
        const isLast = index === allItems.length - 1;
        const isLink = item.href && !isLast;

        return (
          <React.Fragment key={index}>
            {isLink ? (
              <Link
                href={item.href!}
                className="flex items-center space-x-1 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors"
              >
                {item.icon && <span>{item.icon}</span>}
                <span>{item.label}</span>
              </Link>
            ) : (
              <span
                className={clsx(
                  'flex items-center space-x-1',
                  isLast
                    ? 'text-gray-900 dark:text-white font-medium'
                    : 'text-gray-600 dark:text-gray-400'
                )}
                aria-current={isLast ? 'page' : undefined}
              >
                {item.icon && <span>{item.icon}</span>}
                <span>{item.label}</span>
              </span>
            )}
            {!isLast && (
              <span className="text-gray-400 dark:text-gray-600" aria-hidden="true">
                {separator}
              </span>
            )}
          </React.Fragment>
        );
      })}
    </nav>
  );
};


