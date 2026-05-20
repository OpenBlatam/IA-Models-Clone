'use client'

import { useEffect, useRef, ReactNode } from 'react'
import { useInView } from 'react-intersection-observer'

interface InfiniteScrollProps {
  children: ReactNode
  onLoadMore: () => void
  hasMore: boolean
  loading?: boolean
  loader?: ReactNode
  className?: string
}

const InfiniteScroll = ({
  children,
  onLoadMore,
  hasMore,
  loading = false,
  loader,
  className,
}: InfiniteScrollProps) => {
  const { ref, inView } = useInView({
    threshold: 0,
    rootMargin: '100px',
  })

  useEffect(() => {
    if (inView && hasMore && !loading) {
      onLoadMore()
    }
  }, [inView, hasMore, loading, onLoadMore])

  return (
    <div className={className}>
      {children}
      {hasMore && (
        <div ref={ref} className="flex justify-center py-4">
          {loading && (loader || <div className="text-gray-500">Loading...</div>)}
        </div>
      )}
    </div>
  )
}

export default InfiniteScroll

