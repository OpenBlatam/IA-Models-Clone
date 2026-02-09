import { useState, useCallback } from 'react';
import { cn } from '@/lib/utils';

interface AccordionItem {
  id: string;
  title: string;
  content: React.ReactNode;
  defaultOpen?: boolean;
}

interface AccordionProps {
  items: AccordionItem[];
  allowMultiple?: boolean;
  className?: string;
}

const Accordion = ({ items, allowMultiple = false, className = '' }: AccordionProps): JSX.Element => {
  const [openItems, setOpenItems] = useState<Set<string>>(
    new Set(items.filter((item) => item.defaultOpen).map((item) => item.id))
  );

  const handleToggle = useCallback(
    (itemId: string): void => {
      setOpenItems((prev) => {
        const newSet = new Set(prev);
        if (newSet.has(itemId)) {
          newSet.delete(itemId);
        } else {
          if (!allowMultiple) {
            newSet.clear();
          }
          newSet.add(itemId);
        }
        return newSet;
      });
    },
    [allowMultiple]
  );

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLButtonElement>, itemId: string): void => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        handleToggle(itemId);
      }
    },
    [handleToggle]
  );

  return (
    <div className={cn('space-y-2', className)}>
      {items.map((item) => {
        const isOpen = openItems.has(item.id);

        return (
          <div key={item.id} className="border rounded-lg overflow-hidden">
            <button
              onClick={() => handleToggle(item.id)}
              onKeyDown={(e) => handleKeyDown(e, item.id)}
              className="w-full px-4 py-3 flex items-center justify-between hover:bg-gray-50 transition-colors"
              aria-expanded={isOpen}
              aria-controls={`accordion-content-${item.id}`}
              tabIndex={0}
            >
              <span className="font-semibold text-left">{item.title}</span>
              <span className="text-xl transition-transform duration-200" aria-hidden="true">
                {isOpen ? '−' : '+'}
              </span>
            </button>
            {isOpen && (
              <div
                id={`accordion-content-${item.id}`}
                className="px-4 py-3 border-t"
                role="region"
                aria-labelledby={`accordion-title-${item.id}`}
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

export default Accordion;



