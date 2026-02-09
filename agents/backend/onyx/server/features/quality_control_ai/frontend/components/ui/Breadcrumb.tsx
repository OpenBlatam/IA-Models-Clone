'use client';

import { memo, useCallback } from 'react';
import { ChevronRight, Home } from 'lucide-react';
import { cn } from '@/lib/utils';
import Link from 'next/link';

interface BreadcrumbItem {
  label: string;
  href?: string;
  icon?: React.ReactNode;
}

interface BreadcrumbProps {
  items: BreadcrumbItem[];
  className?: string;
  separator?: React.ReactNode;
  showHome?: boolean;
  homeHref?: string;
  maxItems?: number;
}

const Breadcrumb = memo(
  ({
    items,
    className,
    separator,
    showHome = false,
    homeHref = '/',
    maxItems,
  }: BreadcrumbProps): JSX.Element => {
    const defaultSeparator = <ChevronRight className="w-4 h-4 text-gray-400 mx-2" aria-hidden="true" />;
    const sep = separator || defaultSeparator;

    const displayItems = useCallback(() => {
      if (!maxItems || items.length <= maxItems) return items;
      
      const first = items[0];
      const last = items[items.length - 1];
      const middle = { label: '...', href: undefined };
      
      return [first, middle, last];
    }, [items, maxItems]);

    const renderedItems = displayItems();

    return (
      <nav className={cn('flex items-center space-x-2', className)} aria-label="Breadcrumb">
        <ol className="flex items-center space-x-2 list-none">
          {showHome && (
            <li className="flex items-center">
              <Link
                href={homeHref}
                className="text-sm text-gray-600 hover:text-gray-900 transition-colors"
                aria-label="Home"
              >
                <Home className="w-4 h-4" aria-hidden="true" />
              </Link>
              {renderedItems.length > 0 && sep}
            </li>
          )}
          {renderedItems.map((item, index) => {
            const isLast = index === renderedItems.length - 1;
            const isEllipsis = item.label === '...';

            return (
              <li key={index} className="flex items-center">
                {index > 0 && !showHome && sep}
                {index > 0 && showHome && sep}
                {isEllipsis ? (
                  <span className="text-sm text-gray-400" aria-hidden="true">
                    {item.label}
                  </span>
                ) : item.href && !isLast ? (
                  <Link
                    href={item.href}
                    className="text-sm text-gray-600 hover:text-gray-900 transition-colors flex items-center gap-1"
                    aria-current={isLast ? 'page' : undefined}
                  >
                    {item.icon && <span className="flex-shrink-0">{item.icon}</span>}
                    {item.label}
                  </Link>
                ) : (
                  <span
                    className="text-sm text-gray-900 font-medium flex items-center gap-1"
                    aria-current={isLast ? 'page' : undefined}
                  >
                    {item.icon && <span className="flex-shrink-0">{item.icon}</span>}
                    {item.label}
                  </span>
                )}
              </li>
            );
          })}
        </ol>
      </nav>
    );
  }
);

Breadcrumb.displayName = 'Breadcrumb';

export default Breadcrumb;

