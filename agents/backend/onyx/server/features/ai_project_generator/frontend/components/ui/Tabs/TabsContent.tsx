'use client'

import { ReactNode } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { cn } from '@/lib/utils'
import { useTabsContext } from './index'

interface TabsContentProps {
  value: string
  children: ReactNode
  className?: string
}

const TabsContent = ({ value, children, className }: TabsContentProps) => {
  const { activeTab } = useTabsContext()
  const isActive = activeTab === value

  return (
    <AnimatePresence mode="wait">
      {isActive && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -10 }}
          transition={{ duration: 0.2 }}
          id={`tabpanel-${value}`}
          role="tabpanel"
          aria-labelledby={`tab-${value}`}
          className={cn('mt-2', className)}
        >
          {children}
        </motion.div>
      )}
    </AnimatePresence>
  )
}

export default TabsContent
