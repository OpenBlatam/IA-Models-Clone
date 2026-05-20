'use client'

import { createContext, useContext, useState, useCallback, ReactNode } from 'react'

interface TabsContextValue {
  activeTab: string
  setActiveTab: (tab: string) => void
}

const TabsContext = createContext<TabsContextValue | undefined>(undefined)

export const useTabsContext = () => {
  const context = useContext(TabsContext)
  if (!context) {
    throw new Error('Tabs components must be used within Tabs provider')
  }
  return context
}

interface TabsProps {
  children: ReactNode
  defaultValue: string
  value?: string
  onValueChange?: (value: string) => void
  className?: string
}

export const Tabs = ({ children, defaultValue, value, onValueChange, className }: TabsProps) => {
  const [internalValue, setInternalValue] = useState(defaultValue)
  const activeTab = value ?? internalValue

  const handleChange = useCallback(
    (newValue: string) => {
      if (value === undefined) {
        setInternalValue(newValue)
      }
      onValueChange?.(newValue)
    },
    [value, onValueChange]
  )

  return (
    <TabsContext.Provider value={{ activeTab, setActiveTab: handleChange }}>
      <div className={className} role="tablist">
        {children}
      </div>
    </TabsContext.Provider>
  )
}

export { default as TabsList } from './TabsList'
export { default as TabsTrigger } from './TabsTrigger'
export { default as TabsContent } from './TabsContent'
