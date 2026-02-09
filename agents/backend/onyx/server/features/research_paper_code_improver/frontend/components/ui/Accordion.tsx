'use client'

import React, { useState } from 'react'
import { ChevronDown } from 'lucide-react'
import { clsx } from 'clsx'

interface AccordionItem {
  title: string
  content: React.ReactNode
  defaultOpen?: boolean
}

interface AccordionProps {
  items: AccordionItem[]
  allowMultiple?: boolean
  className?: string
}

const Accordion: React.FC<AccordionProps> = ({
  items,
  allowMultiple = false,
  className,
}) => {
  const [openItems, setOpenItems] = useState<Set<number>>(
    new Set(items.map((item, index) => (item.defaultOpen ? index : -1)).filter((i) => i >= 0))
  )

  const toggleItem = (index: number) => {
    setOpenItems((prev) => {
      const newSet = new Set(prev)
      if (newSet.has(index)) {
        newSet.delete(index)
      } else {
        if (!allowMultiple) {
          newSet.clear()
        }
        newSet.add(index)
      }
      return newSet
    })
  }

  return (
    <div className={clsx('space-y-2', className)}>
      {items.map((item, index) => {
        const isOpen = openItems.has(index)
        return (
          <div
            key={index}
            className="border border-gray-200 rounded-lg overflow-hidden"
          >
            <button
              onClick={() => toggleItem(index)}
              className="w-full flex items-center justify-between p-4 text-left hover:bg-gray-50 transition-colors focus:outline-none focus:ring-2 focus:ring-primary-500"
              aria-expanded={isOpen}
              aria-controls={`accordion-content-${index}`}
            >
              <span className="font-medium text-gray-900">{item.title}</span>
              <ChevronDown
                className={clsx(
                  'w-5 h-5 text-gray-500 transition-transform',
                  isOpen && 'transform rotate-180'
                )}
              />
            </button>
            {isOpen && (
              <div
                id={`accordion-content-${index}`}
                className="p-4 pt-0 text-sm text-gray-700 border-t border-gray-200"
              >
                {item.content}
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}

export default Accordion




