/**
 * Providers Component
 * Centralized providers for the application
 * Optimized with code splitting and better organization
 */

'use client';

import { Toaster } from 'sonner';
import { SessionProvider } from '@/components/auth/SessionProvider';
import { QueryProvider } from '@/lib/providers/query-provider';
import { StripeProvider } from '@/lib/providers/stripe-provider';
import type { ReactNode } from 'react';

interface ProvidersProps {
  children: ReactNode;
}

/**
 * Root providers component
 * Combines all necessary providers for the application
 */
export const Providers = ({ children }: ProvidersProps) => {
  return (
    <SessionProvider>
      <QueryProvider>
        <StripeProvider>
          {children}
          <Toaster
            position="top-right"
            richColors
            closeButton
            toastOptions={{
              duration: 4000,
              style: {
                background: 'var(--background)',
                color: 'var(--foreground)',
              },
            }}
          />
        </StripeProvider>
      </QueryProvider>
    </SessionProvider>
  );
};
