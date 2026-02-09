'use client'

import { ReactNode } from 'react'
import { motion, type Variants } from 'framer-motion'

interface AnimatedWrapperProps {
  children: ReactNode
  delay?: number
  duration?: number
  className?: string
  variant?: 'fade' | 'slide' | 'scale'
}

const variants: Record<string, Variants> = {
  fade: {
    hidden: { opacity: 0 },
    visible: { opacity: 1 },
  },
  slide: {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0 },
  },
  scale: {
    hidden: { opacity: 0, scale: 0.95 },
    visible: { opacity: 1, scale: 1 },
  },
}

const AnimatedWrapper = ({
  children,
  delay = 0,
  duration = 0.3,
  className,
  variant = 'fade',
}: AnimatedWrapperProps) => {
  return (
    <motion.div
      initial="hidden"
      animate="visible"
      variants={variants[variant]}
      transition={{ duration, delay }}
      className={className}
    >
      {children}
    </motion.div>
  )
}

export default AnimatedWrapper

