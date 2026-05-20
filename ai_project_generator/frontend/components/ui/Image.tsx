'use client'

import { useState, ReactNode } from 'react'
import NextImage from 'next/image'
import { cn } from '@/lib/utils'
import { Skeleton } from '@/components/ui'

interface CustomImageProps {
  src: string
  alt: string
  width?: number
  height?: number
  className?: string
  fill?: boolean
  priority?: boolean
  placeholder?: 'blur' | 'empty'
  blurDataURL?: string
  onError?: () => void
  fallback?: ReactNode
}

const CustomImage = ({
  src,
  alt,
  width,
  height,
  className,
  fill = false,
  priority = false,
  placeholder = 'empty',
  blurDataURL,
  onError,
  fallback,
}: CustomImageProps) => {
  const [error, setError] = useState(false)
  const [loading, setLoading] = useState(true)

  const handleError = () => {
    setError(true)
    setLoading(false)
    onError?.()
  }

  const handleLoad = () => {
    setLoading(false)
  }

  if (error && fallback) {
    return <>{fallback}</>
  }

  if (error) {
    return (
      <div
        className={cn('bg-gray-200 flex items-center justify-center', className)}
        style={fill ? {} : { width, height }}
      >
        <span className="text-gray-400 text-sm">Failed to load image</span>
      </div>
    )
  }

  return (
    <div className={cn('relative', className)}>
      {loading && (
        <Skeleton
          className="absolute inset-0"
          variant="rectangular"
          width={fill ? '100%' : width}
          height={fill ? '100%' : height}
        />
      )}
      <NextImage
        src={src}
        alt={alt}
        width={fill ? undefined : width}
        height={fill ? undefined : height}
        fill={fill}
        priority={priority}
        placeholder={placeholder}
        blurDataURL={blurDataURL}
        onError={handleError}
        onLoad={handleLoad}
        className={cn(loading && 'opacity-0', 'transition-opacity')}
      />
    </div>
  )
}

const Image = CustomImage

export default Image

