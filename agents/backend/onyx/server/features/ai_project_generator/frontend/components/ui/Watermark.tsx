'use client'

import { ReactNode } from 'react'
import { cn } from '@/lib/utils'

interface WatermarkProps {
  children: ReactNode
  text?: string
  opacity?: number
  fontSize?: number
  className?: string
}

const Watermark = ({
  children,
  text = 'Watermark',
  opacity = 0.1,
  fontSize = 48,
  className,
}: WatermarkProps) => {
  return (
    <div className={cn('relative overflow-hidden', className)}>
      {children}
      <div
        className="absolute inset-0 pointer-events-none flex items-center justify-center"
        style={{
          opacity,
          fontSize: `${fontSize}px`,
          fontWeight: 'bold',
          color: '#000',
          transform: 'rotate(-45deg)',
          userSelect: 'none',
        }}
      >
        {text}
      </div>
    </div>
  )
}

export default Watermark

