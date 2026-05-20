'use client'

import { useState, useCallback, useRef, ReactNode, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useClickOutside } from '@/hooks/ui'
import { cn } from '@/lib/utils'
import Portal from './Portal'

interface ContextMenuItem {
  label: string
  icon?: ReactNode
  onClick: () => void
  disabled?: boolean
  separator?: boolean
}

interface ContextMenuProps {
  trigger: ReactNode
  items: ContextMenuItem[]
  className?: string
}

const ContextMenu = ({ trigger, items, className }: ContextMenuProps) => {
  const [isOpen, setIsOpen] = useState(false)
  const [position, setPosition] = useState({ x: 0, y: 0 })
  const containerRef = useClickOutside<HTMLDivElement>(() => setIsOpen(false))

  const handleContextMenu = useCallback((e: React.MouseEvent) => {
    e.preventDefault()
    setPosition({ x: e.clientX, y: e.clientY })
    setIsOpen(true)
  }, [])

  useEffect(() => {
    if (isOpen) {
      const handleClick = () => setIsOpen(false)
      document.addEventListener('click', handleClick)
      return () => document.removeEventListener('click', handleClick)
    }
  }, [isOpen])

  return (
    <div className={cn('inline-block', className)} onContextMenu={handleContextMenu}>
      {trigger}
      {isOpen && typeof window !== 'undefined' && (
        <Portal>
          <AnimatePresence>
            <motion.div
              ref={containerRef}
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              transition={{ duration: 0.15 }}
              className="fixed z-50 bg-white rounded-lg shadow-lg border border-gray-200 py-1 min-w-[200px]"
              style={{ left: position.x, top: position.y }}
              role="menu"
            >
              {items.map((item, index) => {
                if (item.separator) {
                  return <div key={index} className="h-px bg-gray-200 my-1" />
                }

                return (
                  <button
                    key={index}
                    onClick={() => {
                      if (!item.disabled) {
                        item.onClick()
                        setIsOpen(false)
                      }
                    }}
                    disabled={item.disabled}
                    className={cn(
                      'w-full flex items-center gap-2 px-4 py-2 text-sm text-left hover:bg-gray-50 transition-colors',
                      item.disabled && 'opacity-50 cursor-not-allowed'
                    )}
                    role="menuitem"
                    tabIndex={item.disabled ? -1 : 0}
                  >
                    {item.icon}
                    {item.label}
                  </button>
                )
              })}
            </motion.div>
          </AnimatePresence>
        </Portal>
      )}
    </div>
  )
}

export default ContextMenu
