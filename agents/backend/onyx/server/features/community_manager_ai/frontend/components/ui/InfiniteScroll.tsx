'use client';

import { useEffect, useRef, ReactNode } from 'react';
import { Loading } from './Loading';
import { cn } from '@/lib/utils';

interface InfiniteScrollProps {
  children: ReactNode;
  hasMore: boolean;
  isLoading: boolean;
  onLoadMore: () => void;
  loader?: ReactNode;
  endMessage?: ReactNode;
  className?: string;
}

export const InfiniteScroll = ({
  children,
  hasMore,
  isLoading,
  onLoadMore,
  loader,
  endMessage,
  className,
}: InfiniteScrollProps) => {
  const observerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!hasMore || isLoading) return;

    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting) {
          onLoadMore();
        }
      },
      { threshold: 0.1 }
    );

    if (observerRef.current) {
      observer.observe(observerRef.current);
    }

    return () => observer.disconnect();
  }, [hasMore, isLoading, onLoadMore]);

  return (
    <div className={cn('space-y-4', className)}>
      {children}
      {isLoading && (
        <div className="flex justify-center py-4">
          {loader || <Loading size="md" />}
        </div>
      )}
      {!hasMore && endMessage && (
        <div className="text-center py-4 text-sm text-gray-500 dark:text-gray-400">
          {endMessage}
        </div>
      )}
      <div ref={observerRef} className="h-1" />
    </div>
  );
};



