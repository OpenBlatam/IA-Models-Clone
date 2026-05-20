'use client'

import { ReactNode } from 'react'
import { cn } from '@/lib/utils'
import { Breadcrumb } from '@/components/ui'

interface BreadcrumbItem {
  label: string
  href?: string
  icon?: ReactNode
}

interface PageHeaderProps {
  title: string
  description?: string
  breadcrumbs?: BreadcrumbItem[]
  actions?: ReactNode
  className?: string
  children?: ReactNode
}

const PageHeader = ({
  title,
  description,
  breadcrumbs,
  actions,
  className,
  children,
}: PageHeaderProps) => {
  return (
    <div className={cn('mb-6', className)}>
      {breadcrumbs && breadcrumbs.length > 0 && (
        <div className="mb-4">
          <Breadcrumb items={breadcrumbs} />
        </div>
      )}
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">{title}</h1>
          {description && <p className="text-gray-600">{description}</p>}
          {children}
        </div>
        {actions && <div className="flex items-center gap-2 ml-4">{actions}</div>}
      </div>
    </div>
  )
}

export default PageHeader

