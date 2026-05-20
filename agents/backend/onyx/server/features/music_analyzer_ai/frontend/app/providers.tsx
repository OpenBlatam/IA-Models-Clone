/**
 * Root providers component that wraps the application with necessary context providers.
 * Includes React Query for data fetching and state management.
 * Enhanced with DevTools and better error handling.
 */

'use client';

import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ReactQueryDevtools } from '@tanstack/react-query-devtools';
import { useState, type ReactNode } from 'react';
import { ErrorBoundary } from '@/components/error-boundary';
import { env } from '@/lib/config/env';
import { performanceConfig } from '@/lib/config/app';

/**
 * Creates a QueryClient with optimized default options.
 * Configures retry logic, caching, and network behavior.
 * Uses performance configuration for cache settings.
 *
 * @returns Configured QueryClient instance
 */
function createQueryClient(): QueryClient {
  return new QueryClient({
    defaultOptions: {
      queries: {
        staleTime: performanceConfig.cache.staleTime,
        gcTime: performanceConfig.cache.gcTime,
        refetchOnWindowFocus: false,
        refetchOnMount: true,
        refetchOnReconnect: true,
        // Optimize network requests
        refetchInterval: false,
        refetchIntervalInBackground: false,
        retry: (failureCount, error) => {
          // Don't retry on 4xx errors (client errors)
          if (
            error &&
            typeof error === 'object' &&
            'statusCode' in error &&
            typeof error.statusCode === 'number' &&
            error.statusCode >= 400 &&
            error.statusCode < 500
          ) {
            return false;
          }
          // Retry up to 2 times for other errors
          return failureCount < 2;
        },
        retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),
        // Network mode for better offline handling
        networkMode: 'online',
        // Structural sharing for better performance
        structuralSharing: true,
      },
      mutations: {
        retry: false, // Don't retry mutations by default
        networkMode: 'online',
        // Optimize mutation behavior
        onError: (error) => {
          // Log mutation errors in development
          if (env.IS_DEVELOPMENT) {
            console.error('Mutation error:', error);
          }
        },
      },
    },
  });
}

interface ProvidersProps {
  children: ReactNode;
}

/**
 * Root providers component.
 * Wraps the application with React Query, DevTools, and error boundary.
 *
 * @param props - Component props
 * @returns Providers component
 */
export function Providers({ children }: ProvidersProps) {
  // Use useState with lazy initialization to create QueryClient only once
  const [queryClient] = useState(createQueryClient);

  return (
    <ErrorBoundary level="page">
      <QueryClientProvider client={queryClient}>
        {children}
        {/* React Query DevTools - only in development */}
        {env.IS_DEVELOPMENT && (
          <ReactQueryDevtools
            initialIsOpen={false}
            position="bottom-right"
            buttonPosition="bottom-right"
          />
        )}
      </QueryClientProvider>
    </ErrorBoundary>
  );
}
