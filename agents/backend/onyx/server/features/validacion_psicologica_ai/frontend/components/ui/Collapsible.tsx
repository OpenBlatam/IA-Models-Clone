/**
 * Collapsible component
 */

'use client';

import React, { useState, useRef, useEffect } from 'react';
import { cn } from '@/lib/utils/cn';
import { ChevronDown } from 'lucide-react';

export interface CollapsibleProps {
  title: string;
  children: React.ReactNode;
  defaultOpen?: boolean;
  className?: string;
  icon?: React.ReactNode;
}

export const Collapsible: React.FC<CollapsibleProps> = ({
  title,
  children,
  defaultOpen = false,
  className,
  icon,
}) => {
  const [isOpen, setIsOpen] = useState(defaultOpen);
  const contentRef = useRef<HTMLDivElement>(null);
  const [height, setHeight] = useState<number | string>(isOpen ? 'auto' : 0);

  useEffect(() => {
    if (contentRef.current) {
      if (isOpen) {
        setHeight(contentRef.current.scrollHeight);
      } else {
        setHeight(0);
      }
    }
  }, [isOpen, children]);

  const handleToggle = () => {
    setIsOpen(!isOpen);
  };

  const handleKeyDown = (event: React.KeyboardEvent) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      handleToggle();
    }
  };

  return (
    <div className={cn('border rounded-lg', className)}>
      <button
        type="button"
        onClick={handleToggle}
        onKeyDown={handleKeyDown}
        className="w-full flex items-center justify-between p-4 hover:bg-accent transition-colors focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2 rounded-t-lg"
        aria-expanded={isOpen}
        aria-controls="collapsible-content"
        tabIndex={0}
      >
        <div className="flex items-center gap-2">
          {icon && <span aria-hidden="true">{icon}</span>}
          <span className="font-medium">{title}</span>
        </div>
        <ChevronDown
          className={cn(
            'h-4 w-4 transition-transform',
            isOpen && 'transform rotate-180'
          )}
          aria-hidden="true"
        />
      </button>
      <div
        id="collapsible-content"
        ref={contentRef}
        className="overflow-hidden transition-all duration-300 ease-in-out"
        style={{ height }}
        aria-hidden={!isOpen}
      >
        <div className="p-4">{children}</div>
      </div>
    </div>
  );
};



