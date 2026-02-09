'use client';

import { ReactNode } from 'react';
import { FiChevronRight, FiHome } from 'react-icons/fi';
import { cn } from '@/utils/classNames';

interface BreadcrumbItem {
  label: ReactNode;
  href?: string;
  onClick?: () => void;
}

interface BreadcrumbsProps {
  items: BreadcrumbItem[];
  showHome?: boolean;
  separator?: ReactNode;
  className?: string;
}

export function Breadcrumbs({
  items,
  showHome = true,
  separator,
  className,
}: BreadcrumbsProps) {
  const allItems = showHome
    ? [{ label: <FiHome size={16} />, href: '/' }, ...items]
    : items;

  return (
    <nav className={cn('flex items-center space-x-2 text-sm', className)} aria-label="Breadcrumb">
      <ol className="flex items-center space-x-2">
        {allItems.map((item, index) => {
          const isLast = index === allItems.length - 1;
          const Separator = separator || <FiChevronRight size={16} className="text-gray-400" />;

          return (
            <li key={index} className="flex items-center">
              {index > 0 && <span className="mx-2">{Separator}</span>}
              {isLast ? (
                <span className="text-gray-900 dark:text-white font-medium">
                  {item.label}
                </span>
              ) : item.href ? (
                <a
                  href={item.href}
                  className="text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 transition-colors"
                >
                  {item.label}
                </a>
              ) : item.onClick ? (
                <button
                  onClick={item.onClick}
                  className="text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 transition-colors"
                >
                  {item.label}
                </button>
              ) : (
                <span className="text-gray-500 dark:text-gray-400">{item.label}</span>
              )}
            </li>
          );
        })}
      </ol>
    </nav>
  );
}

