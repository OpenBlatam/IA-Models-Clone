'use client';

import React, { useEffect } from 'react';
import { clsx } from 'clsx';

interface BackdropProps {
  isOpen: boolean;
  onClose?: () => void;
  className?: string;
  blur?: boolean;
}

export const Backdrop: React.FC<BackdropProps> = ({
  isOpen,
  onClose,
  className,
  blur = false,
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

  if (!isOpen) return null;

  return (
    <div
      className={clsx(
        'fixed inset-0 z-40 bg-black transition-opacity',
        blur && 'backdrop-blur-sm',
        className
      )}
      style={{ opacity: 0.5 }}
      onClick={onClose}
      aria-hidden="true"
    />
  );
};
