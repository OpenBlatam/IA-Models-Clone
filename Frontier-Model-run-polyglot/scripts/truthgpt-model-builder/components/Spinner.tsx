/**
 * Componente Spinner
 * ==================
 * 
 * Componente de spinner/loader mejorado
 */

'use client'

import React from 'react'
import { motion } from 'framer-motion'
import { Loader2 } from 'lucide-react'

export interface SpinnerProps {
  size?: 'sm' | 'md' | 'lg' | 'xl'
  variant?: 'default' | 'primary' | 'success' | 'warning' | 'danger'
  className?: string
  text?: string
}

export default function Spinner({
  size = 'md',
  variant = 'default',
  className = '',
  text
}: SpinnerProps) {
  const sizeStyles = {
    sm: 'w-4 h-4',
    md: 'w-6 h-6',
    lg: 'w-8 h-8',
    xl: 'w-12 h-12'
  }

  const variantStyles = {
    default: 'text-gray-600 dark:text-gray-400',
    primary: 'text-blue-600 dark:text-blue-400',
    success: 'text-green-600 dark:text-green-400',
    warning: 'text-yellow-600 dark:text-yellow-400',
    danger: 'text-red-600 dark:text-red-400'
  }

  return (
    <div className={`flex flex-col items-center justify-center gap-2 ${className}`}>
      <motion.div
        animate={{ rotate: 360 }}
        transition={{
          duration: 1,
          repeat: Infinity,
          ease: 'linear'
        }}
        className={variantStyles[variant]}
      >
        <Loader2 className={sizeStyles[size]} />
      </motion.div>
      {text && (
        <p className={`text-sm ${variantStyles[variant]}`}>
          {text}
        </p>
      )}
    </div>
  )
}






