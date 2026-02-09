'use client';

import { memo, useState, type ReactNode } from 'react';
import { ChevronDown } from 'lucide-react';
import { cn } from '@/lib/utils';

interface AccordionItem {
  title: string;
  content: ReactNode;
  defaultOpen?: boolean;
}

interface AccordionProps {
  items: AccordionItem[];
  allowMultiple?: boolean;
  className?: string;
}

const Accordion = memo(
  ({ items, allowMultiple = false, className }: AccordionProps): JSX.Element => {
    const [openItems, setOpenItems] = useState<Set<number>>(
      new Set(items.map((item, index) => (item.defaultOpen ? index : -1)).filter((i) => i >= 0))
    );

    const toggleItem = (index: number): void => {
      setOpenItems((prev) => {
        const newSet = new Set(prev);
        if (newSet.has(index)) {
          if (allowMultiple) {
            newSet.delete(index);
          } else {
            newSet.clear();
          }
        } else {
          if (!allowMultiple) {
            newSet.clear();
          }
          newSet.add(index);
        }
        return newSet;
      });
    };

    return (
      <div className={cn('space-y-2', className)} role="region" aria-label="Accordion">
        {items.map((item, index) => {
          const isOpen = openItems.has(index);

          return (
            <div
              key={index}
              className="border border-gray-200 rounded-lg overflow-hidden"
              role="group"
            >
              <button
                type="button"
                onClick={() => toggleItem(index)}
                className="w-full px-4 py-3 flex items-center justify-between text-left bg-gray-50 hover:bg-gray-100 transition-colors"
                aria-expanded={isOpen}
                aria-controls={`accordion-content-${index}`}
              >
                <span className="font-medium text-gray-900">{item.title}</span>
                <ChevronDown
                  className={cn(
                    'w-5 h-5 text-gray-500 transition-transform',
                    isOpen && 'transform rotate-180'
                  )}
                  aria-hidden="true"
                />
              </button>
              {isOpen && (
                <div
                  id={`accordion-content-${index}`}
                  className="px-4 py-3 bg-white"
                  role="region"
                  aria-labelledby={`accordion-header-${index}`}
                >
                  {item.content}
                </div>
              )}
            </div>
          );
        })}
      </div>
    );
  }
);

Accordion.displayName = 'Accordion';

export default Accordion;

