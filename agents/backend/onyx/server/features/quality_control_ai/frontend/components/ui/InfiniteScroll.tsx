'use client';

import { memo, type ReactNode } from 'react';
import { useInfiniteScroll } from '@/lib/hooks';
import { cn } from '@/lib/utils';
import { LoadingSpinner } from './LoadingSpinner';

interface InfiniteScrollProps {
  hasMore: boolean;
  isLoading: boolean;
  onLoadMore: () => void;
  children: ReactNode;
  loader?: ReactNode;
  endMessage?: ReactNode;
  className?: string;
  threshold?: number;
}

const InfiniteScroll = memo(
  ({
    hasMore,
    isLoading,
    onLoadMore,
    children,
    loader,
    endMessage,
    className,
    threshold = 100,
  }: InfiniteScrollProps): JSX.Element => {
    const { ref } = useInfiniteScroll({
      hasMore,
      isLoading,
      onLoadMore,
      threshold,
    });

    return (
      <div className={cn('space-y-4', className)}>
        {children}
        {hasMore && (
          <div ref={ref} className="flex justify-center py-4">
            {isLoading && (loader || <LoadingSpinner size="md" />)}
          </div>
        )}
        {!hasMore && endMessage && (
          <div className="text-center py-4 text-gray-500">{endMessage}</div>
        )}
      </div>
    );
  }
);

InfiniteScroll.displayName = 'InfiniteScroll';

export default InfiniteScroll;

