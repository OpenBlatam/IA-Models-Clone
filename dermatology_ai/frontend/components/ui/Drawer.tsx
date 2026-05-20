'use client';

import React, { useEffect, memo, useCallback, useMemo } from 'react';
import { X } from 'lucide-react';
import { clsx } from 'clsx';
import { Portal } from './Portal';
import { FocusTrap } from './FocusTrap';

interface DrawerProps {
  isOpen: boolean;
  onClose: () => void;
  title?: string;
  children: React.ReactNode;
  side?: 'left' | 'right' | 'top' | 'bottom';
  size?: 'sm' | 'md' | 'lg' | 'xl' | 'full';
  showCloseButton?: boolean;
}

const getSizes = (side: 'left' | 'right' | 'top' | 'bottom') => ({
  sm: side === 'left' || side === 'right' ? 'w-64' : 'h-64',
  md: side === 'left' || side === 'right' ? 'w-96' : 'h-96',
  lg: side === 'left' || side === 'right' ? 'w-[32rem]' : 'h-[32rem]',
  xl: side === 'left' || side === 'right' ? 'w-[42rem]' : 'h-[42rem]',
  full: side === 'left' || side === 'right' ? 'w-full' : 'h-full',
});

const sides = {
  left: 'left-0 top-0 bottom-0',
  right: 'right-0 top-0 bottom-0',
  top: 'top-0 left-0 right-0',
  bottom: 'bottom-0 left-0 right-0',
};

export const Drawer: React.FC<DrawerProps> = memo(({
  isOpen,
  onClose,
  title,
  children,
  side = 'right',
  size = 'md',
  showCloseButton = true,
}) => {
  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = 'unset';
    }
    return () => {
      document.body.style.overflow = 'unset';
    };
  }, [isOpen]);

  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen) {
        onClose();
      }
    };

    window.addEventListener('keydown', handleEscape);
    return () => window.removeEventListener('keydown', handleEscape);
  }, [isOpen, onClose]);

  const sizes = useMemo(() => getSizes(side), [side]);

  const handleBackdropClick = useCallback(() => {
    onClose();
  }, [onClose]);

  if (!isOpen) return null;

  return (
    <Portal>
      <div className="fixed inset-0 z-50 overflow-hidden">
        <div
          className="fixed inset-0 bg-black bg-opacity-50 transition-opacity"
          onClick={handleBackdropClick}
        />
        <FocusTrap active={isOpen} onEscape={onClose}>
          <div
            className={clsx(
              'fixed bg-white dark:bg-gray-900 shadow-xl',
              sides[side],
              sizes[size],
              'transform transition-transform duration-300 ease-in-out',
              'flex flex-col'
            )}
          >
            {(title || showCloseButton) && (
              <div className="flex items-center justify-between p-6 border-b border-gray-200 dark:border-gray-700">
                {title && (
                  <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                    {title}
                  </h3>
                )}
                {showCloseButton && (
                  <button
                    onClick={onClose}
                    className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
                    aria-label="Close"
                  >
                    <X className="h-6 w-6" />
                  </button>
                )}
              </div>
            )}
            <div className="flex-1 overflow-y-auto p-6">{children}</div>
          </div>
        </FocusTrap>
      </div>
    </Portal>
  );
});

Drawer.displayName = 'Drawer';
