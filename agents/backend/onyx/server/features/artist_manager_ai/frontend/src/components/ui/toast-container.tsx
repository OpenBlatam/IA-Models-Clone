'use client';

import { Toaster } from 'sonner';

export const ToastContainer = () => {
  return (
    <Toaster
      position="top-right"
      richColors
      expand={true}
      duration={4000}
      closeButton
      toastOptions={{
        classNames: {
          success: 'bg-green-50 border-green-200 text-green-800',
          error: 'bg-red-50 border-red-200 text-red-800',
          warning: 'bg-yellow-50 border-yellow-200 text-yellow-800',
          info: 'bg-blue-50 border-blue-200 text-blue-800',
        },
      }}
    />
  );
};

