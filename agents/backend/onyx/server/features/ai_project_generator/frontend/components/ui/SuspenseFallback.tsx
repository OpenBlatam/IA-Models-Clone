'use client'

import { ReactNode } from 'react'
import { LoadingSpinner } from '@/components/ui'

interface SuspenseFallbackProps {
  children?: ReactNode
  text?: string
}

const SuspenseFallback = ({ children, text = 'Loading...' }: SuspenseFallbackProps) => {
  return (
    <div className="flex items-center justify-center min-h-[200px]">
      {children || <LoadingSpinner size="lg" text={text} />}
    </div>
  )
}

export default SuspenseFallback

