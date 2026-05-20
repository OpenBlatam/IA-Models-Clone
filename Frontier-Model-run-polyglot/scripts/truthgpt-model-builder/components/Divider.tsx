/**
 * Componente Divider
 * ==================
 * 
 * Componente de divisor/separador
 */

'use client'

import React from 'react'

export interface DividerProps {
  orientation?: 'horizontal' | 'vertical'
  variant?: 'solid' | 'dashed' | 'dotted'
  spacing?: 'sm' | 'md' | 'lg'
  label?: string
  className?: string
}

export default function Divider({
  orientation = 'horizontal',
  variant = 'solid',
  spacing = 'md',
  label,
  className = ''
}: DividerProps) {
  const variantStyles = {
    solid: 'border-solid',
    dashed: 'border-dashed',
    dotted: 'border-dotted'
  }

  const spacingStyles = {
    sm: orientation === 'horizontal' ? 'my-2' : 'mx-2',
    md: orientation === 'horizontal' ? 'my-4' : 'mx-4',
    lg: orientation === 'horizontal' ? 'my-6' : 'mx-6'
  }

  if (orientation === 'vertical') {
    return (
      <div
        className={`
          inline-block h-full w-px border-l border-gray-300 dark:border-gray-700
          ${variantStyles[variant]}
          ${spacingStyles[spacing]}
          ${className}
        `}
        role="separator"
        aria-orientation="vertical"
      />
    )
  }

  if (label) {
    return (
      <div className={`flex items-center ${spacingStyles[spacing]} ${className}`}>
        <div className={`flex-1 border-t border-gray-300 dark:border-gray-700 ${variantStyles[variant]}`} />
        <span className="px-4 text-sm text-gray-500 dark:text-gray-400">
          {label}
        </span>
        <div className={`flex-1 border-t border-gray-300 dark:border-gray-700 ${variantStyles[variant]}`} />
      </div>
    )
  }

  return (
    <hr
      className={`
        border-0 border-t border-gray-300 dark:border-gray-700
        ${variantStyles[variant]}
        ${spacingStyles[spacing]}
        ${className}
      `}
      role="separator"
    />
  )
}






