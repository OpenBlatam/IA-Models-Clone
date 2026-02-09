'use client';

import { memo, useState, type ReactNode } from 'react';
import { ChevronDown } from 'lucide-react';
import { cn } from '@/lib/utils';

interface CollapsibleProps {
  title: string;
  children: ReactNode;
  defaultOpen?: boolean;
  className?: string;
  headerClassName?: string;
  contentClassName?: string;
}

const Collapsible = memo(
  ({
    title,
    children,
    defaultOpen = false,
    className,
    headerClassName,
    contentClassName,
  }: CollapsibleProps): JSX.Element => {
    const [isOpen, setIsOpen] = useState(defaultOpen);

    return (
      <div className={cn('border border-gray-200 rounded-lg overflow-hidden', className)}>
        <button
          type="button"
          onClick={() => setIsOpen(!isOpen)}
          className={cn(
            'w-full px-4 py-3 flex items-center justify-between text-left bg-gray-50 hover:bg-gray-100 transition-colors',
            headerClassName
          )}
          aria-expanded={isOpen}
        >
          <span className="font-medium text-gray-900">{title}</span>
          <ChevronDown
            className={cn(
              'w-5 h-5 text-gray-500 transition-transform',
              isOpen && 'transform rotate-180'
            )}
            aria-hidden="true"
          />
        </button>
        {isOpen && (
          <div className={cn('px-4 py-3 bg-white', contentClassName)}>{children}</div>
        )}
      </div>
    );
  }
);

Collapsible.displayName = 'Collapsible';

export default Collapsible;

