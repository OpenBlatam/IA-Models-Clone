'use client';

import { ChevronRight, Home } from 'lucide-react';
import { cn } from '@/lib/utils/cn';
// Note: Using regular anchor tag for flexibility

interface BreadcrumbItem {
  label: string;
  href?: string;
  onClick?: () => void;
}

interface BreadcrumbsProps {
  items: BreadcrumbItem[];
  className?: string;
  showHome?: boolean;
}

export default function Breadcrumbs({
  items,
  className,
  showHome = true,
}: BreadcrumbsProps) {
  const allItems = showHome
    ? [{ label: 'Inicio', href: '/' }, ...items]
    : items;

  return (
    <nav
      aria-label="Breadcrumb"
      className={cn('flex items-center gap-2 text-sm', className)}
    >
      <ol className="flex items-center gap-2">
        {allItems.map((item, index) => {
          const isLast = index === allItems.length - 1;

          return (
            <li key={index} className="flex items-center gap-2">
              {index === 0 && showHome ? (
                <a
                  href={item.href || '#'}
                  onClick={item.onClick}
                  className="text-tesla-gray-dark hover:text-tesla-blue transition-colors min-h-[44px] min-w-[44px] flex items-center justify-center"
                  aria-label="Inicio"
                >
                  <Home className="w-4 h-4" />
                </a>
              ) : (
                <>
                  {item.href ? (
                    <a
                      href={item.href}
                      onClick={item.onClick}
                      className={cn(
                        'transition-colors min-h-[44px] flex items-center',
                        isLast
                          ? 'text-tesla-black font-medium'
                          : 'text-tesla-gray-dark hover:text-tesla-blue'
                      )}
                    >
                      {item.label}
                    </a>
                  ) : (
                    <span
                      onClick={item.onClick}
                      className={cn(
                        'cursor-pointer transition-colors min-h-[44px] flex items-center',
                        isLast
                          ? 'text-tesla-black font-medium'
                          : 'text-tesla-gray-dark hover:text-tesla-blue'
                      )}
                    >
                      {item.label}
                    </span>
                  )}
                </>
              )}
              {!isLast && (
                <ChevronRight className="w-4 h-4 text-tesla-gray-light" />
              )}
            </li>
          );
        })}
      </ol>
    </nav>
  );
}

