'use client';

import Link from 'next/link';
import { ChevronRight, Home } from 'lucide-react';
import { cn } from '@/lib/utils';

interface BreadcrumbItem {
  label: string;
  href?: string;
}

interface BreadcrumbsProps {
  items: BreadcrumbItem[];
  className?: string;
  showHome?: boolean;
}

const Breadcrumbs = ({ items, className, showHome = true }: BreadcrumbsProps) => {
  const allItems = showHome
    ? [{ label: 'Inicio', href: '/' }, ...items]
    : items;

  return (
    <nav aria-label="Breadcrumb" className={cn('flex items-center space-x-2 text-sm', className)}>
      <ol className="flex items-center space-x-2">
        {allItems.map((item, index) => {
          const isLast = index === allItems.length - 1;

          return (
            <li key={index} className="flex items-center">
              {index === 0 && showHome ? (
                <Link
                  href={item.href || '#'}
                  className="text-gray-500 hover:text-gray-700 transition-colors"
                  aria-label="Inicio"
                >
                  <Home className="w-4 h-4" />
                </Link>
              ) : (
                <>
                  {item.href && !isLast ? (
                    <Link
                      href={item.href}
                      className="text-gray-500 hover:text-gray-700 transition-colors"
                    >
                      {item.label}
                    </Link>
                  ) : (
                    <span className={cn(isLast ? 'text-gray-900 font-medium' : 'text-gray-500')}>
                      {item.label}
                    </span>
                  )}
                </>
              )}
              {!isLast && (
                <ChevronRight className="w-4 h-4 text-gray-400 mx-2" aria-hidden="true" />
              )}
            </li>
          );
        })}
      </ol>
    </nav>
  );
};

export { Breadcrumbs };

