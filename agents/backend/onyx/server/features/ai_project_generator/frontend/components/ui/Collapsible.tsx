'use client'

import { useState, ReactNode, useCallback } from 'react'
import { ChevronDown } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'
import { cn } from '@/lib/utils'

interface CollapsibleProps {
  title: string
  children: ReactNode
  defaultOpen?: boolean
  icon?: ReactNode
  className?: string
  triggerClassName?: string
  contentClassName?: string
}

const Collapsible = ({
  title,
  children,
  defaultOpen = false,
  icon,
  className,
  triggerClassName,
  contentClassName,
}: CollapsibleProps) => {
  const [isOpen, setIsOpen] = useState(defaultOpen)

  const handleToggle = useCallback(() => {
    setIsOpen((prev) => !prev)
  }, [])

  return (
    <div className={cn('w-full', className)}>
      <button
        onClick={handleToggle}
        className={cn(
          'w-full flex items-center justify-between p-4 text-left hover:bg-gray-50 transition-colors rounded-lg',
          triggerClassName
        )}
        aria-expanded={isOpen}
      >
        <div className="flex items-center gap-3">
          {icon}
          <span className="font-medium text-gray-900">{title}</span>
        </div>
        <ChevronDown
          className={cn(
            'w-5 h-5 text-gray-500 transition-transform',
            isOpen && 'transform rotate-180'
          )}
        />
      </button>
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className={cn('p-4 pt-0', contentClassName)}>{children}</div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

export default Collapsible
