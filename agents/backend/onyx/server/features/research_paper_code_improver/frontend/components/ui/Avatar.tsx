'use client'

import React from 'react'
import { User } from 'lucide-react'

interface AvatarProps {
  src?: string
  alt?: string
  name?: string
  size?: 'sm' | 'md' | 'lg' | 'xl'
  className?: string
  fallback?: React.ReactNode
}

const sizeClasses = {
  sm: 'w-8 h-8 text-xs',
  md: 'w-10 h-10 text-sm',
  lg: 'w-12 h-12 text-base',
  xl: 'w-16 h-16 text-lg',
}

const Avatar: React.FC<AvatarProps> = ({
  src,
  alt,
  name,
  size = 'md',
  className = '',
  fallback,
}) => {
  const [imageError, setImageError] = React.useState(false)

  const getInitials = (name: string) => {
    return name
      .split(' ')
      .map((n) => n[0])
      .join('')
      .toUpperCase()
      .slice(0, 2)
  }

  const displayContent = () => {
    if (src && !imageError) {
      return (
        <img
          src={src}
          alt={alt || name || 'Avatar'}
          className="w-full h-full object-cover rounded-full"
          onError={() => setImageError(true)}
        />
      )
    }

    if (fallback) {
      return fallback
    }

    if (name) {
      return (
        <span className="font-semibold text-gray-700">
          {getInitials(name)}
        </span>
      )
    }

    return <User className="w-1/2 h-1/2 text-gray-400" />
  }

  return (
    <div
      className={`${sizeClasses[size]} rounded-full bg-gray-200 flex items-center justify-center overflow-hidden ${className}`}
      role="img"
      aria-label={alt || name || 'Avatar'}
    >
      {displayContent()}
    </div>
  )
}

export default Avatar



