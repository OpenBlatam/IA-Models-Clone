'use client'

import Skeleton from 'react-loading-skeleton'
import 'react-loading-skeleton/dist/skeleton.css'

interface LoadingSkeletonProps {
  count?: number
  height?: number
  className?: string
}

export default function LoadingSkeleton({ count = 1, height = 20, className }: LoadingSkeletonProps) {
  return (
    <Skeleton
      count={count}
      height={height}
      baseColor="#1e293b"
      highlightColor="#334155"
      className={className}
    />
  )
}


