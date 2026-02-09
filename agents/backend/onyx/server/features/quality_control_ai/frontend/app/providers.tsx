'use client';

import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useState } from 'react';
import { APP_CONFIG } from '@/config/app.config';

const Providers = ({ children }: { children: React.ReactNode }): JSX.Element => {
  const [queryClient] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            staleTime: APP_CONFIG.QUERY_CONFIG.STALE_TIME,
            refetchOnWindowFocus: APP_CONFIG.QUERY_CONFIG.REFETCH_ON_WINDOW_FOCUS,
            refetchOnMount: APP_CONFIG.QUERY_CONFIG.REFETCH_ON_MOUNT,
            retry: APP_CONFIG.QUERY_CONFIG.RETRY,
          },
        },
      })
  );

  return (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
};

export default Providers;
