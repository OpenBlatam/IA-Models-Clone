'use client'

import { ReactNode } from 'react'
import { motion } from 'framer-motion'
import { useIntersectionObserver } from '@/hooks/ui'
import { cn } from '@/lib/utils'

interface RevealProps {
  children: ReactNode
  direction?: 'up' | 'down' | 'left' | 'right' | 'fade'
  delay?: number
  duration?: number
  className?: string
}

const Reveal = ({
  children,
  direction = 'up',
  delay = 0,
  duration = 0.5,
  className,
}: RevealProps) => {
  const { ref, isIntersecting } = useIntersectionObserver({ triggerOnce: true })

  const variants = {
    up: { y: 50, opacity: 0 },
    down: { y: -50, opacity: 0 },
    left: { x: 50, opacity: 0 },
    right: { x: -50, opacity: 0 },
    fade: { opacity: 0 },
  }

  return (
    <motion.div
      ref={ref}
      initial={variants[direction]}
      animate={isIntersecting ? { x: 0, y: 0, opacity: 1 } : variants[direction]}
      transition={{ duration, delay }}
      className={cn(className)}
    >
      {children}
    </motion.div>
  )
}

export default Reveal

