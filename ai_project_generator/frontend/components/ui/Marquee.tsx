'use client'

import { ReactNode } from 'react'
import { cn } from '@/lib/utils'

interface MarqueeProps {
  children: ReactNode
  direction?: 'left' | 'right' | 'up' | 'down'
  speed?: 'slow' | 'normal' | 'fast'
  pauseOnHover?: boolean
  className?: string
}

const Marquee = ({
  children,
  direction = 'left',
  speed = 'normal',
  pauseOnHover = false,
  className,
}: MarqueeProps) => {
  const speedClasses = {
    slow: 'animate-marquee-slow',
    normal: 'animate-marquee',
    fast: 'animate-marquee-fast',
  }

  const directionClasses = {
    left: 'animate-marquee-left',
    right: 'animate-marquee-right',
    up: 'animate-marquee-up',
    down: 'animate-marquee-down',
  }

  return (
    <div
      className={cn(
        'overflow-hidden whitespace-nowrap',
        pauseOnHover && 'hover:[animation-play-state:paused]',
        className
      )}
    >
      <div className={cn('inline-block', speedClasses[speed], directionClasses[direction])}>
        {children}
      </div>
    </div>
  )
}

export default Marquee

