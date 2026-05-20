'use client';

import React from 'react';
import { Toast } from 'react-hot-toast';
import { clsx } from 'clsx';

interface ToastContainerProps {
  toasts: Toast[];
  position?: 'top-left' | 'top-center' | 'top-right' | 'bottom-left' | 'bottom-center' | 'bottom-right';
}

export const ToastContainer: React.FC<ToastContainerProps> = ({
  toasts,
  position = 'top-right',
}) => {
  const positions = {
    'top-left': 'top-4 left-4',
    'top-center': 'top-4 left-1/2 -translate-x-1/2',
    'top-right': 'top-4 right-4',
    'bottom-left': 'bottom-4 left-4',
    'bottom-center': 'bottom-4 left-1/2 -translate-x-1/2',
    'bottom-right': 'bottom-4 right-4',
  };

  return (
    <div
      className={clsx(
        'fixed z-50 flex flex-col space-y-2',
        positions[position]
      )}
    >
      {toasts.map((toast) => (
        <div
          key={toast.id}
          className={clsx(
            'bg-white dark:bg-gray-800 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700 p-4 min-w-[300px] max-w-[500px]',
            'animate-fade-in'
          )}
        >
          {toast.message}
        </div>
      ))}
    </div>
  );
};


