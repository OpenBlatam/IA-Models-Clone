'use client'

import { ReactNode } from 'react'
import { X } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'
import { cn } from '@/lib/utils'
import { useClickOutside } from '@/hooks/ui'
import Portal from './Portal'
import Button from './Button'

interface SheetProps {
  isOpen: boolean
  onClose: () => void
  title?: string
  children: ReactNode
  side?: 'left' | 'right' | 'top' | 'bottom'
  size?: 'sm' | 'md' | 'lg' | 'xl' | 'full'
  showCloseButton?: boolean
  className?: string
}

const Sheet = ({
  isOpen,
  onClose,
  title,
  children,
  side = 'right',
  size = 'md',
  showCloseButton = true,
  className,
}: SheetProps) => {
  const containerRef = useClickOutside<HTMLDivElement>(onClose)

  const sizeClasses = {
    sm: side === 'left' || side === 'right' ? 'w-80' : 'h-80',
    md: side === 'left' || side === 'right' ? 'w-96' : 'h-96',
    lg: side === 'left' || side === 'right' ? 'w-[32rem]' : 'h-[32rem]',
    xl: side === 'left' || side === 'right' ? 'w-[42rem]' : 'h-[42rem]',
    full: side === 'left' || side === 'right' ? 'w-full' : 'h-full',
  }

  const sideClasses = {
    left: 'left-0 top-0 bottom-0',
    right: 'right-0 top-0 bottom-0',
    top: 'top-0 left-0 right-0',
    bottom: 'bottom-0 left-0 right-0',
  }

  const variants = {
    left: {
      initial: { x: '-100%' },
      animate: { x: 0 },
      exit: { x: '-100%' },
    },
    right: {
      initial: { x: '100%' },
      animate: { x: 0 },
      exit: { x: '100%' },
    },
    top: {
      initial: { y: '-100%' },
      animate: { y: 0 },
      exit: { y: '-100%' },
    },
    bottom: {
      initial: { y: '100%' },
      animate: { y: 0 },
      exit: { y: '100%' },
    },
  }

  if (!isOpen) {
    return null
  }

  return (
    <Portal>
      <AnimatePresence>
        {isOpen && (
          <>
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 bg-black/50 z-40"
              onClick={onClose}
            />
            <motion.div
              ref={containerRef}
              initial={variants[side].initial}
              animate={variants[side].animate}
              exit={variants[side].exit}
              transition={{ type: 'spring', damping: 25, stiffness: 200 }}
              className={cn(
                'fixed z-50 bg-white shadow-xl flex flex-col',
                sizeClasses[size],
                sideClasses[side],
                className
              )}
            >
              {title && (
                <div className="flex items-center justify-between p-4 border-b border-gray-200">
                  <h2 className="text-lg font-semibold text-gray-900">{title}</h2>
                  {showCloseButton && (
                    <Button
                      variant="secondary"
                      size="sm"
                      onClick={onClose}
                      leftIcon={<X className="w-4 h-4" />}
                    />
                  )}
                </div>
              )}
              <div className="flex-1 overflow-y-auto p-4">{children}</div>
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </Portal>
  )
}

export default Sheet

