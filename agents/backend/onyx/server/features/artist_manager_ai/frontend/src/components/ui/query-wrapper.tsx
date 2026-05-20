'use client';

import { ReactNode } from 'react';
import { UseQueryResult } from '@tanstack/react-query';
import { LoadingSpinner } from '@/components/ui/loading-spinner';
import { EmptyState } from '@/components/ui/empty-state';
import { ErrorDisplay } from '@/components/ui/error-display';
import { AlertCircle } from 'lucide-react';

interface QueryWrapperProps<TData> {
  query: UseQueryResult<TData>;
  children: (data: TData) => ReactNode;
  loadingComponent?: ReactNode;
  errorComponent?: ReactNode;
  emptyComponent?: ReactNode;
  isEmpty?: (data: TData) => boolean;
  emptyMessage?: string;
  emptyTitle?: string;
}

const QueryWrapper = <TData,>({
  query,
  children,
  loadingComponent,
  errorComponent,
  emptyComponent,
  isEmpty,
  emptyMessage = 'No se encontraron datos',
  emptyTitle = 'Sin datos',
}: QueryWrapperProps<TData>) => {
  const { data, isLoading, isError, error, refetch } = query;

  if (isLoading) {
    return (
      <>
        {loadingComponent || <LoadingSpinner message="Cargando..." fullScreen />}
      </>
    );
  }

  if (isError) {
    return (
      <>
        {errorComponent || (
          <ErrorDisplay
            error={error instanceof Error ? error : new Error('Ha ocurrido un error')}
            onRetry={() => refetch()}
          />
        )}
      </>
    );
  }

  if (data === undefined || data === null) {
    return (
      <>
        {emptyComponent || (
          <EmptyState
            icon={AlertCircle}
            title={emptyTitle}
            description={emptyMessage}
          />
        )}
      </>
    );
  }

  if (isEmpty && isEmpty(data)) {
    return (
      <>
        {emptyComponent || (
          <EmptyState
            icon={AlertCircle}
            title={emptyTitle}
            description={emptyMessage}
          />
        )}
      </>
    );
  }

  return <>{children(data)}</>;
};

export { QueryWrapper };

