'use client';

import React, { useState, useRef, useEffect } from 'react';
import { clsx } from 'clsx';

interface HoverCardProps {
  trigger: React.ReactNode;
  content: React.ReactNode;
  delay?: number;
  className?: string;
}

export const HoverCard: React.FC<HoverCardProps> = ({
  trigger,
  content,
  delay = 200,
  className,
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [position, setPosition] = useState({ top: 0, left: 0 });
  const timeoutRef = useRef<NodeJS.Timeout>();
  const cardRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (isOpen && triggerRef.current && cardRef.current) {
      const triggerRect = triggerRef.current.getBoundingClientRect();
      const cardRect = cardRef.current.getBoundingClientRect();

      setPosition({
        top: triggerRect.bottom + 8,
        left: triggerRect.left + triggerRect.width / 2 - cardRect.width / 2,
      });
    }
  }, [isOpen]);

  const handleMouseEnter = () => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }
    timeoutRef.current = setTimeout(() => {
      setIsOpen(true);
    }, delay);
  };

  const handleMouseLeave = () => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }
    setIsOpen(false);
  };

  return (
    <div className={clsx('relative inline-block', className)}>
      <div
        ref={triggerRef}
        onMouseEnter={handleMouseEnter}
        onMouseLeave={handleMouseLeave}
      >
        {trigger}
      </div>
      {isOpen && (
        <div
          ref={cardRef}
          onMouseEnter={handleMouseEnter}
          onMouseLeave={handleMouseLeave}
          className="fixed z-50 bg-white dark:bg-gray-800 rounded-lg shadow-xl border border-gray-200 dark:border-gray-700 p-4 min-w-[200px] max-w-[300px] animate-fade-in"
          style={{ top: `${position.top}px`, left: `${position.left}px` }}
        >
          {content}
        </div>
      )}
    </div>
  );
};


