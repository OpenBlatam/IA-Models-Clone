/**
 * Componente Tabs
 * ===============
 * 
 * Componente de pestañas mejorado
 */

'use client'

import React, { useState, useId } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

export interface Tab {
  id: string
  label: string
  content: React.ReactNode
  disabled?: boolean
  icon?: React.ReactNode
}

export interface TabsProps {
  tabs: Tab[]
  defaultTab?: string
  onChange?: (tabId: string) => void
  variant?: 'default' | 'pills' | 'underline'
  orientation?: 'horizontal' | 'vertical'
  className?: string
}

export default function Tabs({
  tabs,
  defaultTab,
  onChange,
  variant = 'default',
  orientation = 'horizontal',
  className = ''
}: TabsProps) {
  const [activeTab, setActiveTab] = useState(defaultTab || tabs[0]?.id || '')
  const tabsId = useId('tabs')

  const handleTabChange = (tabId: string) => {
    if (tabs.find(t => t.id === tabId)?.disabled) return
    
    setActiveTab(tabId)
    onChange?.(tabId)
  }

  const activeTabContent = tabs.find(t => t.id === activeTab)?.content

  const variantStyles = {
    default: {
      tab: 'px-4 py-2 text-sm font-medium text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-100 border-b-2 border-transparent hover:border-gray-300 dark:hover:border-gray-600',
      active: 'text-gray-900 dark:text-gray-100 border-blue-600 dark:border-blue-400',
      disabled: 'opacity-50 cursor-not-allowed'
    },
    pills: {
      tab: 'px-4 py-2 text-sm font-medium rounded-lg text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800',
      active: 'bg-blue-100 dark:bg-blue-900/30 text-blue-900 dark:text-blue-100',
      disabled: 'opacity-50 cursor-not-allowed'
    },
    underline: {
      tab: 'px-4 py-2 text-sm font-medium text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-100 border-b-2 border-transparent',
      active: 'text-gray-900 dark:text-gray-100 border-gray-900 dark:border-gray-100',
      disabled: 'opacity-50 cursor-not-allowed'
    }
  }

  const styles = variantStyles[variant]

  return (
    <div className={className}>
      <div
        role="tablist"
        aria-orientation={orientation}
        className={orientation === 'vertical' ? 'flex flex-col gap-2' : 'flex border-b border-gray-200 dark:border-gray-700'}
      >
        {tabs.map((tab) => {
          const isActive = activeTab === tab.id
          const isDisabled = tab.disabled

          return (
            <button
              key={tab.id}
              role="tab"
              aria-selected={isActive}
              aria-controls={`${tabsId}-panel-${tab.id}`}
              id={`${tabsId}-tab-${tab.id}`}
              disabled={isDisabled}
              onClick={() => handleTabChange(tab.id)}
              className={`
                ${styles.tab}
                ${isActive ? styles.active : ''}
                ${isDisabled ? styles.disabled : ''}
                flex items-center gap-2 transition-colors
              `}
            >
              {tab.icon && <span>{tab.icon}</span>}
              {tab.label}
            </button>
          )
        })}
      </div>

      <div role="tabpanel" className="mt-4">
        <AnimatePresence mode="wait">
          <motion.div
            key={activeTab}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.2 }}
          >
            {activeTabContent}
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  )
}






