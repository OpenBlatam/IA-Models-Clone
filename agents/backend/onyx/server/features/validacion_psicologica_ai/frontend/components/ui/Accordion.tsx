/**
 * Accordion component (enhanced version)
 */

'use client';

import React, { useState } from 'react';
import { cn } from '@/lib/utils/cn';
import { ChevronDown } from 'lucide-react';

export interface AccordionItem {
  id: string;
  title: string;
  content: React.ReactNode;
  icon?: React.ReactNode;
  disabled?: boolean;
}

export interface AccordionProps {
  items: AccordionItem[];
  allowMultiple?: boolean;
  defaultOpen?: string[];
  className?: string;
}

export const Accordion: React.FC<AccordionProps> = ({
  items,
  allowMultiple = false,
  defaultOpen = [],
  className,
}) => {
  const [openItems, setOpenItems] = useState<string[]>(defaultOpen);

  const handleToggle = (itemId: string) => {
    setOpenItems((prev) => {
      if (allowMultiple) {
        return prev.includes(itemId)
          ? prev.filter((id) => id !== itemId)
          : [...prev, itemId];
      } else {
        return prev.includes(itemId) ? [] : [itemId];
      }
    });
  };

  const handleKeyDown = (event: React.KeyboardEvent, itemId: string) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      handleToggle(itemId);
    }
  };

  return (
    <div className={cn('space-y-2', className)}>
      {items.map((item) => {
        const isOpen = openItems.includes(item.id);
        const isDisabled = item.disabled;

        return (
          <div key={item.id} className="border rounded-lg">
            <button
              type="button"
              onClick={() => !isDisabled && handleToggle(item.id)}
              onKeyDown={(e) => !isDisabled && handleKeyDown(e, item.id)}
              disabled={isDisabled}
              className={cn(
                'w-full flex items-center justify-between p-4 hover:bg-accent transition-colors focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2 rounded-t-lg',
                isDisabled && 'opacity-50 cursor-not-allowed'
              )}
              aria-expanded={isOpen}
              aria-controls={`accordion-content-${item.id}`}
              aria-disabled={isDisabled}
              tabIndex={isDisabled ? -1 : 0}
            >
              <div className="flex items-center gap-2">
                {item.icon && <span aria-hidden="true">{item.icon}</span>}
                <span className="font-medium text-left">{item.title}</span>
              </div>
              <ChevronDown
                className={cn(
                  'h-4 w-4 transition-transform',
                  isOpen && 'transform rotate-180'
                )}
                aria-hidden="true"
              />
            </button>
            {isOpen && (
              <div
                id={`accordion-content-${item.id}`}
                className="p-4 border-t"
                role="region"
                aria-labelledby={`accordion-header-${item.id}`}
              >
                {item.content}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
};
