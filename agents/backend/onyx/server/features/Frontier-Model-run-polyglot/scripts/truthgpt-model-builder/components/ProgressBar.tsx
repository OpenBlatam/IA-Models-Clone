/**
 * Componente ProgressBar
 * ======================
 * 
 * Componente de barra de progreso mejorado
 */

'use client'

import React from 'react'
import { motion } from 'framer-motion'

export interface ProgressBarProps {
  value: number // 0-100
  max?: number
  variant?: 'default' | 'primary' | 'success' | 'warning' | 'danger'
  size?: 'sm' | 'md' | 'lg'
  showLabel?: boolean
  animated?: boolean
  className?: string
}

export default function ProgressBar({
  value,
  max = 100,
  variant = 'default',
  size = 'md',
  showLabel = false,
  animated = true,
  className = ''
}: ProgressBarProps) {
  const percentage = Math.min(Math.max((value / max) * 100, 0), 100)

  const sizeStyles = {
    sm: 'h-1',
    md: 'h-2',
    lg: 'h-3'
  }

  const variantStyles = {
    default: 'bg-gray-200 dark:bg-gray-700',
    primary: 'bg-blue-200 dark:bg-blue-900/30',
    success: 'bg-green-200 dark:bg-green-900/30',
    warning: 'bg-yellow-200 dark:bg-yellow-900/30',
    danger: 'bg-red-200 dark:bg-red-900/30'
  }

  const fillStyles = {
    default: 'bg-gray-600 dark:bg-gray-400',
    primary: 'bg-blue-600 dark:bg-blue-400',
    success: 'bg-green-600 dark:bg-green-400',
    warning: 'bg-yellow-600 dark:bg-yellow-400',
    danger: 'bg-red-600 dark:bg-red-400'
  }

  return (
    <div className={`w-full ${className}`}>
      {showLabel && (
        <div className="flex justify-between items-center mb-1">
          <span className="text-sm text-gray-700 dark:text-gray-300">
            Progreso
          </span>
          <span className="text-sm font-medium text-gray-900 dark:text-gray-100">
            {Math.round(percentage)}%
          </span>
        </div>
      )}
      <div className={`w-full ${variantStyles[variant]} rounded-full overflow-hidden ${sizeStyles[size]}`}>
        <motion.div
          initial={animated ? { width: 0 } : { width: `${percentage}%` }}
          animate={{ width: `${percentage}%` }}
          transition={{ duration: animated ? 0.5 : 0, ease: 'easeOut' }}
          className={`h-full ${fillStyles[variant]} rounded-full`}
        />
      </div>
    </div>
  )
}






