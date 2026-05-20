'use client';

import { QueryClientProvider } from '@tanstack/react-query';
import { Toaster } from 'react-hot-toast';
import { useState } from 'react';
import { createQueryClient } from '@/lib/utils/query-client';
import { TOAST_CONFIG } from '@/lib/constants';

export const Providers = ({ children }: { children: React.ReactNode }): JSX.Element => {
  const [queryClient] = useState(() => createQueryClient());

  return (
    <QueryClientProvider client={queryClient}>
      {children}
      <Toaster
        position="top-right"
        toastOptions={{
          duration: TOAST_CONFIG.DURATION,
          style: {
            background: '#363636',
            color: '#fff',
          },
          success: {
            duration: TOAST_CONFIG.SUCCESS_DURATION,
            iconTheme: {
              primary: '#10b981',
              secondary: '#fff',
            },
          },
          error: {
            duration: TOAST_CONFIG.ERROR_DURATION,
            iconTheme: {
              primary: '#ef4444',
              secondary: '#fff',
            },
          },
        }}
      />
    </QueryClientProvider>
  );
};

