'use client'

import { useState, ReactNode, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { cn } from '@/lib/utils'

interface DisclosureProps {
  button: ReactNode
  children: ReactNode
  defaultOpen?: boolean
  className?: string
}

const Disclosure = ({
  button,
  children,
  defaultOpen = false,
  className,
}: DisclosureProps) => {
  const [isOpen, setIsOpen] = useState(defaultOpen)

  const handleToggle = useCallback(() => {
    setIsOpen((prev) => !prev)
  }, [])

  return (
    <div className={cn('w-full', className)}>
      <div onClick={handleToggle} className="cursor-pointer">
        {button}
      </div>
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            {children}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

export default Disclosure

