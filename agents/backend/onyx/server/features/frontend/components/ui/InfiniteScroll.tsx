'use client';

import { ReactNode, useEffect, useRef, useState } from 'react';
import { useIntersectionObserver } from '@/hooks';
import { cn } from '@/utils/classNames';
import { Spinner } from './Spinner';

interface InfiniteScrollProps {
  children: ReactNode;
  onLoadMore: () => void | Promise<void>;
  hasMore: boolean;
  isLoading?: boolean;
  loader?: ReactNode;
  endMessage?: ReactNode;
  className?: string;
}

export function InfiniteScroll({
  children,
  onLoadMore,
  hasMore,
  isLoading = false,
  loader,
  endMessage,
  className,
}: InfiniteScrollProps) {
  const [triggerRef, isIntersecting] = useIntersectionObserver({
    threshold: 0.1,
    triggerOnce: false,
  });

  const [isLoadingMore, setIsLoadingMore] = useState(false);

  useEffect(() => {
    if (isIntersecting && hasMore && !isLoading && !isLoadingMore) {
      setIsLoadingMore(true);
      Promise.resolve(onLoadMore()).finally(() => {
        setIsLoadingMore(false);
      });
    }
  }, [isIntersecting, hasMore, isLoading, isLoadingMore, onLoadMore]);

  return (
    <div className={cn('w-full', className)}>
      {children}
      {hasMore && (
        <div ref={triggerRef} className="flex justify-center py-4">
          {loader || <Spinner size="md" />}
        </div>
      )}
      {!hasMore && endMessage && (
        <div className="text-center py-4 text-gray-500 dark:text-gray-400">
          {endMessage}
        </div>
      )}
    </div>
  );
}

