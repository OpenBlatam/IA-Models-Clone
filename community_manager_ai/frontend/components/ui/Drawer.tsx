'use client';

import { ReactNode } from 'react';
import * as Dialog from '@radix-ui/react-dialog';
import { X } from 'lucide-react';
import { cn } from '@/lib/utils';
import { motion, AnimatePresence } from 'framer-motion';

interface DrawerProps {
  isOpen: boolean;
  onClose: () => void;
  children: ReactNode;
  title?: string;
  side?: 'left' | 'right' | 'top' | 'bottom';
  size?: 'sm' | 'md' | 'lg' | 'xl' | 'full';
  className?: string;
}

const sizeClasses = {
  sm: 'max-w-sm',
  md: 'max-w-md',
  lg: 'max-w-lg',
  xl: 'max-w-xl',
  full: 'max-w-full',
};

const sideClasses = {
  left: 'left-0 top-0 h-full',
  right: 'right-0 top-0 h-full',
  top: 'top-0 left-0 w-full',
  bottom: 'bottom-0 left-0 w-full',
};

export const Drawer = ({
  isOpen,
  onClose,
  children,
  title,
  side = 'right',
  size = 'md',
  className,
}: DrawerProps) => {
  const isVertical = side === 'top' || side === 'bottom';
  const widthClass = isVertical ? 'w-full' : sizeClasses[size];
  const heightClass = isVertical ? sizeClasses[size] : 'h-full';

  return (
    <Dialog.Root open={isOpen} onOpenChange={onClose}>
      <AnimatePresence>
        {isOpen && (
          <>
            <Dialog.Overlay asChild>
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="fixed inset-0 z-50 bg-black/50"
                onClick={onClose}
              />
            </Dialog.Overlay>
            <Dialog.Content
              asChild
              className={cn(
                'fixed z-50 bg-white dark:bg-gray-800 shadow-xl',
                sideClasses[side],
                widthClass,
                heightClass,
                className
              )}
            >
              <motion.div
                initial={
                  side === 'left'
                    ? { x: '-100%' }
                    : side === 'right'
                    ? { x: '100%' }
                    : side === 'top'
                    ? { y: '-100%' }
                    : { y: '100%' }
                }
                animate={{ x: 0, y: 0 }}
                exit={
                  side === 'left'
                    ? { x: '-100%' }
                    : side === 'right'
                    ? { x: '100%' }
                    : side === 'top'
                    ? { y: '-100%' }
                    : { y: '100%' }
                }
                transition={{ type: 'spring', damping: 25, stiffness: 200 }}
                className="flex flex-col"
              >
                {title && (
                  <div className="flex items-center justify-between border-b border-gray-200 dark:border-gray-700 px-6 py-4">
                    <Dialog.Title className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                      {title}
                    </Dialog.Title>
                    <button
                      onClick={onClose}
                      className="rounded-lg p-1 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 transition-colors"
                      aria-label="Cerrar"
                    >
                      <X className="h-5 w-5" />
                    </button>
                  </div>
                )}
                <div className="flex-1 overflow-y-auto p-6">{children}</div>
              </motion.div>
            </Dialog.Content>
          </>
        )}
      </AnimatePresence>
    </Dialog.Root>
  );
};



