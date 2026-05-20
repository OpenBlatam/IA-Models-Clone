'use client';

import { ReactNode, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useClickOutside } from '@/hooks';

interface TooltipProps {
  content: ReactNode;
  children: ReactNode;
  position?: 'top' | 'bottom' | 'left' | 'right';
  delay?: number;
  disabled?: boolean;
}

const positionClasses = {
  top: 'bottom-full left-1/2 transform -translate-x-1/2 mb-2',
  bottom: 'top-full left-1/2 transform -translate-x-1/2 mt-2',
  left: 'right-full top-1/2 transform -translate-y-1/2 mr-2',
  right: 'left-full top-1/2 transform -translate-y-1/2 ml-2',
};

export function Tooltip({
  content,
  children,
  position = 'top',
  delay = 200,
  disabled = false,
}: TooltipProps) {
  const [isVisible, setIsVisible] = useState(false);
  const [timeoutId, setTimeoutId] = useState<NodeJS.Timeout | null>(null);

  const handleMouseEnter = () => {
    if (disabled) return;
    const id = setTimeout(() => setIsVisible(true), delay);
    setTimeoutId(id);
  };

  const handleMouseLeave = () => {
    if (timeoutId) {
      clearTimeout(timeoutId);
      setTimeoutId(null);
    }
    setIsVisible(false);
  };

  return (
    <div
      className="relative inline-block"
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
    >
      {children}
      <AnimatePresence>
        {isVisible && !disabled && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            className={`
              absolute z-50
              ${positionClasses[position]}
              px-3 py-1.5
              bg-gray-900 dark:bg-gray-700
              text-white text-sm
              rounded-lg shadow-lg
              whitespace-nowrap
              pointer-events-none
            `}
          >
            {content}
            {position === 'top' && (
              <div className="absolute top-full left-1/2 transform -translate-x-1/2 -mt-1">
                <div className="border-4 border-transparent border-t-gray-900 dark:border-t-gray-700"></div>
              </div>
            )}
            {position === 'bottom' && (
              <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 -mb-1">
                <div className="border-4 border-transparent border-b-gray-900 dark:border-b-gray-700"></div>
              </div>
            )}
            {position === 'left' && (
              <div className="absolute left-full top-1/2 transform -translate-y-1/2 -ml-1">
                <div className="border-4 border-transparent border-l-gray-900 dark:border-l-gray-700"></div>
              </div>
            )}
            {position === 'right' && (
              <div className="absolute right-full top-1/2 transform -translate-y-1/2 -mr-1">
                <div className="border-4 border-transparent border-r-gray-900 dark:border-r-gray-700"></div>
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

