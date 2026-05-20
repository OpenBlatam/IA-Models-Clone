'use client'

import { useCallback, ReactNode } from 'react'
import { cn } from '@/lib/utils'
import { useTabsContext } from './index'

interface TabsTriggerProps {
  value: string
  children: ReactNode
  className?: string
  disabled?: boolean
}

const TabsTrigger = ({ value, children, className, disabled = false }: TabsTriggerProps) => {
  const { activeTab, setActiveTab } = useTabsContext()

  const handleClick = useCallback(() => {
    if (!disabled) {
      setActiveTab(value)
    }
  }, [value, disabled, setActiveTab])

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if ((e.key === 'Enter' || e.key === ' ') && !disabled) {
        e.preventDefault()
        setActiveTab(value)
      }
    },
    [value, disabled, setActiveTab]
  )

  const isActive = activeTab === value

  return (
    <button
      onClick={handleClick}
      onKeyDown={handleKeyDown}
      disabled={disabled}
      className={cn(
        'inline-flex items-center justify-center whitespace-nowrap rounded-sm px-3 py-1.5 text-sm font-medium transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-500 disabled:pointer-events-none disabled:opacity-50',
        isActive
          ? 'bg-white text-primary-600 shadow-sm'
          : 'text-gray-600 hover:bg-gray-50 hover:text-gray-900',
        className
      )}
      role="tab"
      aria-selected={isActive}
      aria-controls={`tabpanel-${value}`}
      tabIndex={disabled ? -1 : 0}
    >
      {children}
    </button>
  )
}

export default TabsTrigger
