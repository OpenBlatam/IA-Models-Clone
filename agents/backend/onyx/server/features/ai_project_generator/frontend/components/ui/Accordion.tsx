'use client'

import { useState, ReactNode, useCallback } from 'react'
import { ChevronDown } from 'lucide-react'
import { cn } from '@/lib/utils'
import { motion, AnimatePresence } from 'framer-motion'

interface AccordionItem {
  id: string
  title: string
  content: ReactNode
  icon?: ReactNode
  disabled?: boolean
}

interface AccordionProps {
  items: AccordionItem[]
  defaultOpen?: string[]
  allowMultiple?: boolean
  className?: string
  variant?: 'default' | 'bordered'
}

const Accordion = ({
  items,
  defaultOpen = [],
  allowMultiple = false,
  className,
  variant = 'default',
}: AccordionProps) => {
  const [openItems, setOpenItems] = useState<string[]>(defaultOpen)

  const handleToggle = useCallback(
    (itemId: string) => {
      setOpenItems((prev) => {
        if (allowMultiple) {
          return prev.includes(itemId) ? prev.filter((id) => id !== itemId) : [...prev, itemId]
        } else {
          return prev.includes(itemId) ? [] : [itemId]
        }
      })
    },
    [allowMultiple]
  )

  const variantClasses = {
    default: 'border-b border-gray-200',
    bordered: 'border border-gray-200 rounded-lg mb-2',
  }

  return (
    <div className={cn('w-full', className)}>
      {items.map((item) => {
        const isOpen = openItems.includes(item.id)

        return (
          <div key={item.id} className={cn(variantClasses[variant], variant === 'bordered' && 'p-1')}>
            <button
              onClick={() => !item.disabled && handleToggle(item.id)}
              disabled={item.disabled}
              className={cn(
                'w-full flex items-center justify-between p-4 text-left hover:bg-gray-50 transition-colors',
                item.disabled && 'opacity-50 cursor-not-allowed'
              )}
              aria-expanded={isOpen}
              aria-controls={`accordion-content-${item.id}`}
            >
              <div className="flex items-center gap-3">
                {item.icon}
                <span className="font-medium text-gray-900">{item.title}</span>
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
                  <div id={`accordion-content-${item.id}`} className="p-4 pt-0 text-gray-600">
                    {item.content}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        )
      })}
    </div>
  )
}

export default Accordion
