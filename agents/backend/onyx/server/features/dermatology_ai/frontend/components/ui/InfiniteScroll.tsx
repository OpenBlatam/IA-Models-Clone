'use client';

import React, { useEffect, useRef, useState } from 'react';
import { useIntersectionObserver } from '@/hooks';

interface InfiniteScrollProps {
  children: React.ReactNode;
  onLoadMore: () => void | Promise<void>;
  hasMore: boolean;
  loading?: boolean;
  loader?: React.ReactNode;
  endMessage?: React.ReactNode;
  threshold?: number;
}

export const InfiniteScroll: React.FC<InfiniteScrollProps> = ({
  children,
  onLoadMore,
  hasMore,
  loading = false,
  loader,
  endMessage,
  threshold = 0.1,
}) => {
  const [elementRef, isIntersecting] = useIntersectionObserver({
    threshold,
    triggerOnce: false,
  });
  const [isLoading, setIsLoading] = useState(false);
  const hasLoadedRef = useRef(false);

  useEffect(() => {
    if (isIntersecting && hasMore && !loading && !isLoading && !hasLoadedRef.current) {
      hasLoadedRef.current = true;
      setIsLoading(true);
      
      Promise.resolve(onLoadMore()).finally(() => {
        setIsLoading(false);
        setTimeout(() => {
          hasLoadedRef.current = false;
        }, 100);
      });
    }
  }, [isIntersecting, hasMore, loading, isLoading, onLoadMore]);

  return (
    <div>
      {children}
      <div ref={elementRef as React.RefObject<HTMLDivElement>}>
        {hasMore && (loading || isLoading) && (
          <div className="flex justify-center py-8">
            {loader || (
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600" />
            )}
          </div>
        )}
        {!hasMore && endMessage && (
          <div className="text-center py-8 text-gray-500 dark:text-gray-400">
            {endMessage}
          </div>
        )}
      </div>
    </div>
  );
};


