'use client'

import { ReactNode } from 'react'
import clsx from 'clsx'
import Card from './Card'

interface AnimatedCardProps {
  children: ReactNode
  className?: string
  delay?: number
  hover?: boolean
}

const AnimatedCard = ({ children, className, delay = 0, hover = true }: AnimatedCardProps) => {
  return (
    <Card
      className={clsx(
        'animate-fade-in',
        hover && 'transition-transform duration-300 hover:scale-[1.02]',
        className
      )}
      style={{ animationDelay: `${delay}ms` }}
    >
      {children}
    </Card>
  )
}

export default AnimatedCard

