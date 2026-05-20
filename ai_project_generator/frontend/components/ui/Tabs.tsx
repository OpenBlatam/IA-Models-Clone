'use client'

import { useState, ReactNode, useCallback } from 'react'
import { cn } from '@/lib/utils'

interface Tab {
  id: string
  label: string
  content: ReactNode
  icon?: ReactNode
  disabled?: boolean
}

interface TabsProps {
  tabs: Tab[]
  defaultTab?: string
  activeTab?: string
  onTabChange?: (tabId: string) => void
  variant?: 'default' | 'pills' | 'underline'
  orientation?: 'horizontal' | 'vertical'
  className?: string
}

const Tabs = ({
  tabs,
  defaultTab,
  activeTab: controlledActiveTab,
  onTabChange,
  variant = 'default',
  orientation = 'horizontal',
  className,
}: TabsProps) => {
  const [internalActiveTab, setInternalActiveTab] = useState(defaultTab || tabs[0]?.id)
  const isControlled = controlledActiveTab !== undefined
  const activeTab = isControlled ? controlledActiveTab : internalActiveTab

  const handleTabChange = useCallback(
    (tabId: string) => {
      if (!isControlled) {
        setInternalActiveTab(tabId)
      }
      onTabChange?.(tabId)
    },
    [isControlled, onTabChange]
  )

  const activeTabContent = tabs.find((tab) => tab.id === activeTab)?.content

  const variantClasses = {
    default: 'border-b-2 border-gray-200',
    pills: 'bg-gray-100 rounded-lg p-1',
    underline: 'border-b border-gray-200',
  }

  return (
    <div
      className={cn(
        'w-full',
        orientation === 'vertical' && 'flex gap-4',
        className
      )}
    >
      <div
        className={cn(
          'flex',
          orientation === 'horizontal' ? 'flex-row' : 'flex-col',
          variantClasses[variant],
          variant === 'pills' && 'inline-flex'
        )}
        role="tablist"
      >
        {tabs.map((tab) => {
          const isActive = tab.id === activeTab

          return (
            <button
              key={tab.id}
              onClick={() => !tab.disabled && handleTabChange(tab.id)}
              disabled={tab.disabled}
              className={cn(
                'px-4 py-2 text-sm font-medium transition-colors',
                variant === 'default' &&
                  (isActive
                    ? 'border-b-2 border-primary-600 text-primary-600'
                    : 'text-gray-500 hover:text-gray-700'),
                variant === 'pills' &&
                  (isActive
                    ? 'bg-white text-primary-600 shadow-sm'
                    : 'text-gray-600 hover:text-gray-900'),
                variant === 'underline' &&
                  (isActive
                    ? 'border-b-2 border-primary-600 text-primary-600'
                    : 'text-gray-500 hover:text-gray-700 hover:border-gray-300'),
                tab.disabled && 'opacity-50 cursor-not-allowed'
              )}
              role="tab"
              aria-selected={isActive}
              aria-controls={`tabpanel-${tab.id}`}
              tabIndex={tab.disabled ? -1 : 0}
            >
              <div className="flex items-center gap-2">
                {tab.icon}
                {tab.label}
              </div>
            </button>
          )
        })}
      </div>
      <div
        className={cn(
          'mt-4',
          orientation === 'vertical' && 'flex-1'
        )}
        role="tabpanel"
      >
        {activeTabContent}
      </div>
    </div>
  )
}

export default Tabs

