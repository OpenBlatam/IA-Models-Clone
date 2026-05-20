/**
 * LoadingSpinner Component
 * Reusable loading spinner with different sizes and styles
 */

'use client'

import React, { memo } from 'react'
import { Loader2 } from 'lucide-react'
import { motion } from 'framer-motion'

interface LoadingSpinnerProps {
  size?: 'sm' | 'md' | 'lg' | 'xl'
  color?: string
  text?: string
  fullScreen?: boolean
  overlay?: boolean
}

export const LoadingSpinner = memo(function LoadingSpinner({
  size = 'md',
  color = 'currentColor',
  text,
  fullScreen = false,
  overlay = false,
}: LoadingSpinnerProps) {
  const sizeMap = {
    sm: 16,
    md: 24,
    lg: 32,
    xl: 48,
  }

  const spinner = (
    <div className={`loading-spinner loading-spinner--${size}`}>
      <motion.div
        animate={{ rotate: 360 }}
        transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
      >
        <Loader2 size={sizeMap[size]} color={color} />
      </motion.div>
      {text && <p className="loading-spinner__text">{text}</p>}
    </div>
  )

  if (fullScreen) {
    return (
      <div className="loading-spinner--fullscreen">
        {spinner}
      </div>
    )
  }

  if (overlay) {
    return (
      <div className="loading-spinner--overlay">
        {spinner}
      </div>
    )
  }

  return spinner
})

export default LoadingSpinner




