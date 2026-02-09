'use client';

import { ErrorBoundary } from 'react-error-boundary';
import { QueryProvider } from '@/components/providers/query-provider';
import { ErrorBoundaryFallback } from '@/components/ui/error-boundary-fallback';
import { ToastContainer } from '@/components/ui/toast-container';

interface ProvidersProps {
  children: React.ReactNode;
}

export const Providers = ({ children }: ProvidersProps) => {
  return (
    <ErrorBoundary FallbackComponent={ErrorBoundaryFallback}>
      <QueryProvider>
        {children}
        <ToastContainer />
      </QueryProvider>
    </ErrorBoundary>
  );
};

