'use client'

import { useState, useRef, ReactNode, useCallback } from 'react'
import { ChevronDown } from 'lucide-react'
import { cn } from '@/lib/utils'
import { useClickOutside } from '@/hooks/ui'
import Button from './Button'

interface DropdownMenuItem {
  label: string
  icon?: ReactNode
  onClick: () => void
  disabled?: boolean
  separator?: boolean
}

interface DropdownMenuProps {
  trigger: ReactNode
  items: DropdownMenuItem[]
  position?: 'top' | 'bottom' | 'left' | 'right'
  className?: string
  align?: 'start' | 'center' | 'end'
}

const DropdownMenu = ({
  trigger,
  items,
  position = 'bottom',
  align = 'start',
  className,
}: DropdownMenuProps) => {
  const [isOpen, setIsOpen] = useState(false)
  const triggerRef = useRef<HTMLDivElement>(null)
  const menuRef = useRef<HTMLDivElement>(null)

  const handleToggle = useCallback(() => {
    setIsOpen((prev) => !prev)
  }, [])

  const handleClose = useCallback(() => {
    setIsOpen(false)
  }, [])

  const handleItemClick = useCallback(
    (item: DropdownMenuItem) => {
      if (!item.disabled) {
        item.onClick()
        handleClose()
      }
    },
    [handleClose]
  )

  const containerRef = useClickOutside<HTMLDivElement>(handleClose)

  return (
    <div ref={containerRef} className="relative inline-block">
      <div ref={triggerRef} onClick={handleToggle} className="inline-block">
        {trigger}
      </div>
      {isOpen && (
        <div
          ref={menuRef}
          className={cn(
            'absolute z-50 mt-2 bg-white rounded-lg shadow-lg border border-gray-200 py-1 min-w-[200px]',
            position === 'top' && 'bottom-full mb-2',
            position === 'bottom' && 'top-full',
            position === 'left' && 'right-full mr-2',
            position === 'right' && 'left-full ml-2',
            align === 'start' && 'left-0',
            align === 'center' && 'left-1/2 -translate-x-1/2',
            align === 'end' && 'right-0',
            className
          )}
          role="menu"
        >
          {items.map((item, index) => {
            if (item.separator) {
              return <div key={index} className="h-px bg-gray-200 my-1" />
            }

            return (
              <button
                key={index}
                onClick={() => handleItemClick(item)}
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
        </div>
      )}
    </div>
  )
}

export default DropdownMenu

