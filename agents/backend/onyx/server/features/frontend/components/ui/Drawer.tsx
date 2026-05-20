'use client';

import { ReactNode, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FiX } from 'react-icons/fi';
import { useClickOutside } from '@/hooks';
import { cn } from '@/utils/classNames';

interface DrawerProps {
  isOpen: boolean;
  onClose: () => void;
  children: ReactNode;
  title?: string;
  placement?: 'left' | 'right' | 'top' | 'bottom';
  size?: 'sm' | 'md' | 'lg' | 'xl' | 'full';
  showCloseButton?: boolean;
  closeOnOverlayClick?: boolean;
}

const sizeClasses = {
  sm: 'max-w-sm',
  md: 'max-w-md',
  lg: 'max-w-lg',
  xl: 'max-w-xl',
  full: 'w-full',
};

const placementClasses = {
  left: 'left-0 top-0 bottom-0',
  right: 'right-0 top-0 bottom-0',
  top: 'top-0 left-0 right-0',
  bottom: 'bottom-0 left-0 right-0',
};

const placementAnimations = {
  left: { x: '-100%' },
  right: { x: '100%' },
  top: { y: '-100%' },
  bottom: { y: '100%' },
};

export function Drawer({
  isOpen,
  onClose,
  children,
  title,
  placement = 'right',
  size = 'md',
  showCloseButton = true,
  closeOnOverlayClick = true,
}: DrawerProps) {
  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = '';
    }
    return () => {
      document.body.style.overflow = '';
    };
  }, [isOpen]);

  const drawerRef = useClickOutside<HTMLDivElement>(() => {
    if (closeOnOverlayClick) onClose();
  });

  const isVertical = placement === 'top' || placement === 'bottom';
  const widthClass = isVertical ? 'w-full' : sizeClasses[size];

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black bg-opacity-50 z-50"
            onClick={closeOnOverlayClick ? onClose : undefined}
          />
          <motion.div
            ref={drawerRef}
            initial={placementAnimations[placement]}
            animate={{ x: 0, y: 0 }}
            exit={placementAnimations[placement]}
            className={cn(
              'fixed z-50 bg-white dark:bg-gray-800 shadow-xl',
              placementClasses[placement],
              widthClass,
              isVertical ? 'h-auto' : 'h-full',
              'flex flex-col'
            )}
            onClick={(e) => e.stopPropagation()}
          >
            {(title || showCloseButton) && (
              <div className="flex items-center justify-between p-6 border-b border-gray-200 dark:border-gray-700">
                {title && (
                  <h3 className="text-xl font-bold text-gray-900 dark:text-white">
                    {title}
                  </h3>
                )}
                {showCloseButton && (
                  <button
                    onClick={onClose}
                    className="btn-icon"
                    aria-label="Cerrar"
                  >
                    <FiX size={20} />
                  </button>
                )}
              </div>
            )}
            <div className="flex-1 overflow-y-auto p-6">{children}</div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}

