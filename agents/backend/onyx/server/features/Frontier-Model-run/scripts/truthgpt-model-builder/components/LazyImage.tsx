/**
 * Componente LazyImage
 * =====================
 * 
 * Componente de imagen con lazy loading
 */

'use client'

import React from 'react'
import { useLazyLoad } from '@/lib/hooks/useLazyLoad'

export interface LazyImageProps extends React.ImgHTMLAttributes<HTMLImageElement> {
  src: string
  alt: string
  placeholder?: string
  fallback?: string
  onLoad?: () => void
  onError?: () => void
}

export default function LazyImage({
  src,
  alt,
  placeholder,
  fallback,
  onLoad,
  onError,
  className = '',
  ...props
}: LazyImageProps) {
  const [imageRef, isVisible] = useLazyLoad<HTMLImageElement>({
    triggerOnce: true
  })
  const [imageSrc, setImageSrc] = React.useState(placeholder || '')
  const [hasError, setHasError] = React.useState(false)

  React.useEffect(() => {
    if (isVisible && !hasError) {
      setImageSrc(src)
    }
  }, [isVisible, src, hasError])

  const handleLoad = () => {
    onLoad?.()
  }

  const handleError = () => {
    setHasError(true)
    if (fallback) {
      setImageSrc(fallback)
    }
    onError?.()
  }

  return (
    <img
      ref={imageRef}
      src={imageSrc}
      alt={alt}
      onLoad={handleLoad}
      onError={handleError}
      className={`
        transition-opacity duration-300
        ${isVisible && !hasError ? 'opacity-100' : 'opacity-0'}
        ${className}
      `}
      {...props}
    />
  )
}






