'use client'

import React, { createContext, useContext, useState } from 'react'
import { clsx } from 'clsx'

interface TabsContextValue {
  activeTab: string
  setActiveTab: (tab: string) => void
}

const TabsContext = createContext<TabsContextValue | undefined>(undefined)

interface TabsProps {
  children: React.ReactNode
  defaultValue: string
  className?: string
}

const Tabs: React.FC<TabsProps & { value?: string; onValueChange?: (value: string) => void }> = ({ 
  children, 
  defaultValue, 
  value: controlledValue,
  onValueChange,
  className 
}) => {
  const [internalValue, setInternalValue] = useState(defaultValue)
  const isControlled = controlledValue !== undefined
  const activeTab = isControlled ? controlledValue : internalValue

  const handleTabChange = (tab: string) => {
    if (isControlled) {
      onValueChange?.(tab)
    } else {
      setInternalValue(tab)
    }
  }

  return (
    <TabsContext.Provider value={{ activeTab, setActiveTab: handleTabChange }}>
      <div className={clsx('w-full', className)}>{children}</div>
    </TabsContext.Provider>
  )
}

interface TabsListProps {
  children: React.ReactNode
  className?: string
}

const TabsList: React.FC<TabsListProps> = ({ children, className }) => {
  return (
    <div
      className={clsx(
        'flex border-b border-gray-200 space-x-1',
        className
      )}
      role="tablist"
    >
      {children}
    </div>
  )
}

interface TabsTriggerProps {
  value: string
  children: React.ReactNode
  className?: string
}

const TabsTrigger: React.FC<TabsTriggerProps> = ({
  value,
  children,
  className,
  onClick,
}) => {
  const context = useContext(TabsContext)
  if (!context) {
    throw new Error('TabsTrigger must be used within Tabs')
  }

  const { activeTab, setActiveTab } = context
  const isActive = activeTab === value

  const handleClick = (e: React.MouseEvent<HTMLButtonElement>) => {
    setActiveTab(value)
    onClick?.(e)
  }

  return (
    <button
      role="tab"
      aria-selected={isActive}
      aria-controls={`tabpanel-${value}`}
      id={`tab-${value}`}
      onClick={handleClick}
      className={clsx(
        'px-4 py-2 text-sm font-medium border-b-2 transition-colors focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2',
        isActive
          ? 'border-primary-600 text-primary-600'
          : 'border-transparent text-gray-600 hover:text-gray-900 hover:border-gray-300',
        className
      )}
    >
      {children}
    </button>
  )
}

interface TabsContentProps {
  value: string
  children: React.ReactNode
  className?: string
}

const TabsContent: React.FC<TabsContentProps> = ({
  value,
  children,
  className,
}) => {
  const context = useContext(TabsContext)
  if (!context) {
    throw new Error('TabsContent must be used within Tabs')
  }

  const { activeTab } = context

  if (activeTab !== value) {
    return null
  }

  return (
    <div
      role="tabpanel"
      id={`tabpanel-${value}`}
      aria-labelledby={`tab-${value}`}
      className={clsx('mt-4', className)}
    >
      {children}
    </div>
  )
}

export { Tabs, TabsList, TabsTrigger, TabsContent }

