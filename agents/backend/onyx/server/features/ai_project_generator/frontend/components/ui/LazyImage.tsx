'use client'

import { useState, useCallback } from 'react'
import { useInView } from 'react-intersection-observer'
import { ImageIcon } from 'lucide-react'
import { cn } from '@/lib/utils'
import Skeleton from './Skeleton'

interface LazyImageProps {
  src: string
  alt: string
  className?: string
  width?: number
  height?: number
  fallback?: string
}

const LazyImage = ({ src, alt, className, width, height, fallback }: LazyImageProps) => {
  const [isLoading, setIsLoading] = useState(true)
  const [hasError, setHasError] = useState(false)
  const { ref, inView } = useInView({
    triggerOnce: true,
    threshold: 0.1,
  })

  const handleLoad = useCallback(() => {
    setIsLoading(false)
  }, [])

  const handleError = useCallback(() => {
    setIsLoading(false)
    setHasError(true)
  }, [])

  if (!inView) {
    return (
      <div ref={ref} className={cn('flex items-center justify-center bg-gray-100', className)}>
        <Skeleton width={width} height={height} />
      </div>
    )
  }

  if (hasError && fallback) {
    return (
      <div
        className={cn('flex items-center justify-center bg-gray-100 text-gray-400', className)}
        style={{ width, height }}
      >
        <ImageIcon className="w-8 h-8" />
      </div>
    )
  }

  return (
    <div ref={ref} className={cn('relative', className)}>
      {isLoading && (
        <div className="absolute inset-0">
          <Skeleton width={width} height={height} />
        </div>
      )}
      <img
        src={src}
        alt={alt}
        width={width}
        height={height}
        onLoad={handleLoad}
        onError={handleError}
        className={cn('transition-opacity duration-300', isLoading ? 'opacity-0' : 'opacity-100')}
        loading="lazy"
      />
    </div>
  )
}

export default LazyImage

