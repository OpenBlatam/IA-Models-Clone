'use client';

import React, { memo } from 'react';
import { Loading } from './Loading';
import { EmptyState } from '@/lib/utils/emptyStates';
import { ErrorFallback } from './ErrorFallback';

interface AsyncContentProps<T> {
  data: T | undefined;
  isLoading: boolean;
  error: Error | null;
  loadingComponent?: React.ReactNode;
  emptyState?: {
    icon?: React.ReactNode;
    title: string;
    description: string;
    actionLabel?: string;
    actionHref?: string;
    onAction?: () => void;
  };
  isEmpty?: (data: T) => boolean;
  children: (data: T) => React.ReactNode;
  onRetry?: () => void;
}

export const AsyncContent = memo(<T,>({
  data,
  isLoading,
  error,
  loadingComponent,
  emptyState,
  isEmpty,
  children,
  onRetry,
}: AsyncContentProps<T>) => {
  if (isLoading) {
    return <>{loadingComponent || <Loading fullScreen text="Loading..." />}</>;
  }

  if (error) {
    return (
      <ErrorFallback
        error={error}
        onRetry={onRetry}
      />
    );
  }

  if (data === undefined || data === null) {
    return null;
  }

  const isDataEmpty = isEmpty ? isEmpty(data) : Array.isArray(data) && data.length === 0;

  if (isDataEmpty && emptyState) {
    return (
      <EmptyState
        icon={emptyState.icon}
        title={emptyState.title}
        description={emptyState.description}
        actionLabel={emptyState.actionLabel}
        actionHref={emptyState.actionHref}
        onAction={emptyState.onAction}
      />
    );
  }

  return <>{children(data)}</>;
}) as <T>(props: AsyncContentProps<T>) => React.ReactElement;

AsyncContent.displayName = 'AsyncContent';



