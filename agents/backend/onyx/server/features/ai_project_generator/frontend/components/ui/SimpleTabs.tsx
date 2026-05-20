'use client'

import { useCallback } from 'react'
import { cn } from '@/lib/utils'
import KeyboardShortcut from './KeyboardShortcut'

interface Tab {
  id: string
  label: string
  icon?: React.ReactNode
  shortcut?: string
}

interface SimpleTabsProps {
  tabs: Tab[]
  activeTab: string
  onTabChange: (tabId: string) => void
  className?: string
}

const SimpleTabs = ({ tabs, activeTab, onTabChange, className }: SimpleTabsProps) => {
  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent, tabId: string) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault()
        onTabChange(tabId)
      }
    },
    [onTabChange]
  )

  return (
    <div className={cn('border-b border-gray-200', className)} role="tablist">
      <div className="flex space-x-8">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => onTabChange(tab.id)}
            onKeyDown={(e) => handleKeyDown(e, tab.id)}
            className={cn(
              'py-4 px-1 border-b-2 font-medium text-sm transition-colors',
              activeTab === tab.id
                ? 'border-primary-500 text-primary-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            )}
            tabIndex={0}
            role="tab"
            aria-selected={activeTab === tab.id}
            aria-controls={`tabpanel-${tab.id}`}
            id={`tab-${tab.id}`}
          >
            <span className="flex items-center gap-2">
              {tab.icon}
              {tab.label}
              {tab.shortcut && (
                <KeyboardShortcut keys={[tab.shortcut]} className="ml-auto opacity-60" />
              )}
            </span>
          </button>
        ))}
      </div>
    </div>
  )
}

export default SimpleTabs

