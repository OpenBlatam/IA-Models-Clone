'use client';

import { ReactNode, useState, createContext, useContext } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FiChevronDown } from 'react-icons/fi';
import { cn } from '@/utils/classNames';

interface AccordionContextType {
  openItems: Set<string>;
  toggleItem: (id: string) => void;
  allowMultiple?: boolean;
}

const AccordionContext = createContext<AccordionContextType | undefined>(undefined);

function useAccordionContext() {
  const context = useContext(AccordionContext);
  if (!context) {
    throw new Error('Accordion components must be used within Accordion');
  }
  return context;
}

interface AccordionProps {
  children: ReactNode;
  allowMultiple?: boolean;
  defaultOpen?: string[];
}

export function Accordion({
  children,
  allowMultiple = false,
  defaultOpen = [],
}: AccordionProps) {
  const [openItems, setOpenItems] = useState<Set<string>>(new Set(defaultOpen));

  const toggleItem = (id: string) => {
    setOpenItems((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        if (!allowMultiple) {
          next.clear();
        }
        next.add(id);
      }
      return next;
    });
  };

  return (
    <AccordionContext.Provider value={{ openItems, toggleItem, allowMultiple }}>
      <div className="w-full">{children}</div>
    </AccordionContext.Provider>
  );
}

interface AccordionItemProps {
  id: string;
  children: ReactNode;
  className?: string;
}

export function AccordionItem({ id, children, className }: AccordionItemProps) {
  return (
    <div className={cn('border-b border-gray-200 dark:border-gray-700', className)}>
      {children}
    </div>
  );
}

interface AccordionTriggerProps {
  children: ReactNode;
  itemId: string;
  className?: string;
}

export function AccordionTrigger({ children, itemId, className }: AccordionTriggerProps) {
  const { openItems, toggleItem } = useAccordionContext();
  const isOpen = openItems.has(itemId);

  return (
    <button
      onClick={() => toggleItem(itemId)}
      className={cn(
        'w-full flex items-center justify-between p-4',
        'hover:bg-gray-50 dark:hover:bg-gray-800',
        'transition-colors',
        className
      )}
      aria-expanded={isOpen}
    >
      <span className="flex-1 text-left">{children}</span>
      <FiChevronDown
        size={20}
        className={cn(
          'text-gray-400 transition-transform',
          isOpen && 'transform rotate-180'
        )}
      />
    </button>
  );
}

interface AccordionContentProps {
  children: ReactNode;
  itemId: string;
  className?: string;
}

export function AccordionContent({ children, itemId, className }: AccordionContentProps) {
  const { openItems } = useAccordionContext();
  const isOpen = openItems.has(itemId);

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial={{ height: 0, opacity: 0 }}
          animate={{ height: 'auto', opacity: 1 }}
          exit={{ height: 0, opacity: 0 }}
          transition={{ duration: 0.2 }}
          className="overflow-hidden"
        >
          <div className={cn('p-4', className)}>{children}</div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

