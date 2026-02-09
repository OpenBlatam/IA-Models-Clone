/**
 * Componente Card
 * ===============
 * 
 * Componente de tarjeta mejorado
 */

'use client'

import React from 'react'
import { motion } from 'framer-motion'

export interface CardProps {
  children: React.ReactNode
  title?: string
  subtitle?: string
  footer?: React.ReactNode
  header?: React.ReactNode
  variant?: 'default' | 'outlined' | 'elevated'
  hover?: boolean
  className?: string
  onClick?: () => void
}

export default function Card({
  children,
  title,
  subtitle,
  footer,
  header,
  variant = 'default',
  hover = false,
  className = '',
  onClick
}: CardProps) {
  const variantStyles = {
    default: 'bg-white dark:bg-gray-800',
    outlined: 'bg-transparent border-2 border-gray-200 dark:border-gray-700',
    elevated: 'bg-white dark:bg-gray-800 shadow-lg'
  }

  const hoverStyles = hover || onClick
    ? 'cursor-pointer transition-all hover:shadow-md dark:hover:shadow-lg'
    : ''

  const content = (
    <>
      {(header || title || subtitle) && (
        <div className="px-6 py-4 border-b border-gray-200 dark:border-gray-700">
          {header || (
            <>
              {title && (
                <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                  {title}
                </h3>
              )}
              {subtitle && (
                <p className="mt-1 text-sm text-gray-600 dark:text-gray-400">
                  {subtitle}
                </p>
              )}
            </>
          )}
        </div>
      )}

      <div className="px-6 py-4">
        {children}
      </div>

      {footer && (
        <div className="px-6 py-4 border-t border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900/50">
          {footer}
        </div>
      )}
    </>
  )

  const cardClasses = `
    rounded-lg overflow-hidden
    ${variantStyles[variant]}
    ${hoverStyles}
    ${className}
  `

  if (onClick) {
    return (
      <motion.div
      whileHover={hover ? { y: -2 } : {}}
      whileTap={onClick ? { scale: 0.98 } : {}}
      className={cardClasses}
      onClick={onClick}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault()
          onClick()
        }
      }}
    >
      {content}
    </motion.div>
    )
  }

  return (
    <div className={cardClasses}>
      {content}
    </div>
  )
}






