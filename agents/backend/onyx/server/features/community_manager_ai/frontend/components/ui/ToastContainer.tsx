'use client';

import { Toast } from './Toast';
import { AnimatePresence } from 'framer-motion';
import { useToast } from '@/hooks/useToast';

export const ToastContainer = () => {
  const { toasts, removeToast } = useToast();

  return (
    <div className="fixed top-4 right-4 z-50 flex flex-col gap-2 max-w-md w-full">
      <AnimatePresence>
        {toasts.map((toast) => (
          <Toast
            key={toast.id}
            id={toast.id}
            message={toast.message}
            type={toast.type}
            duration={toast.duration}
            onClose={removeToast}
          />
        ))}
      </AnimatePresence>
    </div>
  );
};

export const useToastContext = () => {
  const { success, error, warning, info } = useToast();
  return { success, error, warning, info };
};
