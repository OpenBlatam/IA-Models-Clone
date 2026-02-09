/**
 * Componente Avatar
 * =================
 * 
 * Componente de avatar mejorado
 */

'use client'

import React from 'react'
import { User } from 'lucide-react'

export interface AvatarProps {
  src?: string
  alt?: string
  name?: string
  size?: 'sm' | 'md' | 'lg' | 'xl'
  shape?: 'circle' | 'square'
  className?: string
  fallback?: React.ReactNode
}

function getInitials(name: string): string {
  const parts = name.trim().split(/\s+/)
  if (parts.length >= 2) {
    return (parts[0][0] + parts[parts.length - 1][0]).toUpperCase()
  }
  return name.substring(0, 2).toUpperCase()
}

function getColorFromName(name: string): string {
  const colors = [
    'bg-red-500',
    'bg-blue-500',
    'bg-green-500',
    'bg-yellow-500',
    'bg-purple-500',
    'bg-pink-500',
    'bg-indigo-500',
    'bg-teal-500'
  ]
  const hash = name.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0)
  return colors[hash % colors.length]
}

export default function Avatar({
  src,
  alt,
  name,
  size = 'md',
  shape = 'circle',
  className = '',
  fallback
}: AvatarProps) {
  const sizeStyles = {
    sm: 'w-8 h-8 text-xs',
    md: 'w-10 h-10 text-sm',
    lg: 'w-12 h-12 text-base',
    xl: 'w-16 h-16 text-lg'
  }

  const shapeStyles = {
    circle: 'rounded-full',
    square: 'rounded-lg'
  }

  const displayName = name || alt || 'User'
  const initials = getInitials(displayName)
  const bgColor = getColorFromName(displayName)

  return (
    <div
      className={`
        ${sizeStyles[size]}
        ${shapeStyles[shape]}
        flex items-center justify-center
        bg-gray-200 dark:bg-gray-700
        text-gray-700 dark:text-gray-300
        font-medium
        overflow-hidden
        ${className}
      `}
      role="img"
      aria-label={alt || displayName}
    >
      {src ? (
        <img
          src={src}
          alt={alt || displayName}
          className={`w-full h-full object-cover ${shapeStyles[shape]}`}
          onError={(e) => {
            // Fallback si la imagen falla
            const target = e.target as HTMLImageElement
            target.style.display = 'none'
            const parent = target.parentElement
            if (parent && !parent.querySelector('.avatar-fallback')) {
              const fallbackEl = document.createElement('div')
              fallbackEl.className = `avatar-fallback ${bgColor} text-white ${sizeStyles[size]} ${shapeStyles[shape]} flex items-center justify-center`
              fallbackEl.textContent = initials
              parent.appendChild(fallbackEl)
            }
          }}
        />
      ) : fallback ? (
        fallback
      ) : (
        <div className={`${bgColor} text-white w-full h-full flex items-center justify-center ${shapeStyles[shape]}`}>
          {initials || <User className="w-1/2 h-1/2" />}
        </div>
      )}
    </div>
  )
}






