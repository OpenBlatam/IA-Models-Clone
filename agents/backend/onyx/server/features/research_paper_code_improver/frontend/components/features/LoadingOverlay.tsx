'use client'

import React from 'react'
import { LoadingSpinner } from '../ui'

interface LoadingOverlayProps {
  isLoading: boolean
  message?: string
  fullScreen?: boolean
}

const LoadingOverlay: React.FC<LoadingOverlayProps> = ({
  isLoading,
  message,
  fullScreen = false,
}) => {
  if (!isLoading) return null

  const containerClass = fullScreen
    ? 'fixed inset-0 z-50'
    : 'absolute inset-0 z-10'

  return (
    <div
      className={`${containerClass} flex items-center justify-center bg-white bg-opacity-90 backdrop-blur-sm`}
    >
      <div className="text-center">
        <LoadingSpinner size="lg" />
        {message && (
          <p className="mt-4 text-sm text-gray-600 font-medium">{message}</p>
        )}
      </div>
    </div>
  )
}

export default LoadingOverlay




