'use client';

import { ReactNode, useState, createContext, useContext } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FiChevronDown } from 'react-icons/fi';
import { cn } from '@/utils/classNames';

interface CollapsibleContextType {
  isOpen: boolean;
  toggle: () => void;
}

const CollapsibleContext = createContext<CollapsibleContextType | undefined>(undefined);

function useCollapsibleContext() {
  const context = useContext(CollapsibleContext);
  if (!context) {
    throw new Error('Collapsible components must be used within Collapsible');
  }
  return context;
}

interface CollapsibleProps {
  children: ReactNode;
  defaultOpen?: boolean;
  onOpenChange?: (open: boolean) => void;
}

export function Collapsible({
  children,
  defaultOpen = false,
  onOpenChange,
}: CollapsibleProps) {
  const [isOpen, setIsOpen] = useState(defaultOpen);

  const toggle = () => {
    const newState = !isOpen;
    setIsOpen(newState);
    onOpenChange?.(newState);
  };

  return (
    <CollapsibleContext.Provider value={{ isOpen, toggle }}>
      <div className="w-full">{children}</div>
    </CollapsibleContext.Provider>
  );
}

interface CollapsibleTriggerProps {
  children: ReactNode;
  className?: string;
  asChild?: boolean;
}

export function CollapsibleTrigger({
  children,
  className,
  asChild = false,
}: CollapsibleTriggerProps) {
  const { isOpen, toggle } = useCollapsibleContext();

  if (asChild && typeof children === 'object' && 'props' in children) {
    return (
      <div onClick={toggle} className={className}>
        {children}
      </div>
    );
  }

  return (
    <button
      onClick={toggle}
      className={cn(
        'w-full flex items-center justify-between',
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

interface CollapsibleContentProps {
  children: ReactNode;
  className?: string;
}

export function CollapsibleContent({
  children,
  className,
}: CollapsibleContentProps) {
  const { isOpen } = useCollapsibleContext();

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
          <div className={cn('pt-2', className)}>{children}</div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

