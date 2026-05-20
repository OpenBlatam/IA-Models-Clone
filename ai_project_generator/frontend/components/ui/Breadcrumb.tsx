'use client'

import { ReactNode } from 'react'
import { ChevronRight, Home } from 'lucide-react'
import { cn } from '@/lib/utils'
import Link from './Link'

interface BreadcrumbItem {
  label: string
  href?: string
  icon?: ReactNode
}

interface BreadcrumbProps {
  items: BreadcrumbItem[]
  separator?: ReactNode
  showHome?: boolean
  homeHref?: string
  className?: string
}

const Breadcrumb = ({
  items,
  separator,
  showHome = true,
  homeHref = '/',
  className,
}: BreadcrumbProps) => {
  const defaultSeparator = <ChevronRight className="w-4 h-4 text-gray-400" />

  return (
    <nav className={cn('flex items-center gap-2', className)} aria-label="Breadcrumb">
      {showHome && (
        <>
          <Link href={homeHref} className="text-gray-500 hover:text-gray-700">
            <Home className="w-4 h-4" />
          </Link>
          {items.length > 0 && <span className="text-gray-400">{separator || defaultSeparator}</span>}
        </>
      )}
      {items.map((item, index) => {
        const isLast = index === items.length - 1

        return (
          <div key={index} className="flex items-center gap-2">
            {item.href && !isLast ? (
              <Link href={item.href} className="text-gray-500 hover:text-gray-700">
                {item.icon && <span className="mr-1">{item.icon}</span>}
                {item.label}
              </Link>
            ) : (
              <span className={cn('text-gray-900', isLast && 'font-medium')}>
                {item.icon && <span className="mr-1">{item.icon}</span>}
                {item.label}
              </span>
            )}
            {!isLast && <span className="text-gray-400">{separator || defaultSeparator}</span>}
          </div>
        )
      })}
    </nav>
  )
}

export default Breadcrumb

