'use client'

import { ReactNode } from 'react'
import { cn } from '@/lib/utils'

interface TabsListProps {
  children: ReactNode
  className?: string
}

const TabsList = ({ children, className }: TabsListProps) => {
  return (
    <div
      className={cn('inline-flex h-10 items-center justify-center rounded-md bg-gray-100 p-1', className)}
      role="tablist"
    >
      {children}
    </div>
  )
}

export default TabsList
