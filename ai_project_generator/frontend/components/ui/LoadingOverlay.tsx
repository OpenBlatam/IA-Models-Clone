'use client'

import { ReactNode } from 'react'
import clsx from 'clsx'
import LoadingSpinner from './LoadingSpinner'

interface LoadingOverlayProps {
  isLoading: boolean
  children: ReactNode
  message?: string
  className?: string
}

const LoadingOverlay = ({ isLoading, children, message, className }: LoadingOverlayProps) => {
  if (!isLoading) {
    return <>{children}</>
  }

  return (
    <div className={clsx('relative', className)}>
      <div className="opacity-50 pointer-events-none">{children}</div>
      <div className="absolute inset-0 flex items-center justify-center bg-white bg-opacity-75 z-10">
        <LoadingSpinner size="lg" text={message} />
      </div>
    </div>
  )
}

export default LoadingOverlay

