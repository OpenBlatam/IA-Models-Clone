'use client'

import { ReactNode } from 'react'
import { ChevronRight, Home } from 'lucide-react'
import { cn } from '@/lib/utils'
import Button from './Button'

interface BreadcrumbItem {
  label: string
  href?: string
  icon?: ReactNode
}

interface BreadcrumbsProps {
  items: BreadcrumbItem[]
  className?: string
  showHome?: boolean
}

const Breadcrumbs = ({ items, className, showHome = true }: BreadcrumbsProps) => {
  const handleClick = (href?: string) => {
    if (href) {
      window.location.href = href
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent, href?: string) => {
    if ((e.key === 'Enter' || e.key === ' ') && href) {
      e.preventDefault()
      handleClick(href)
    }
  }

  return (
    <nav className={cn('flex items-center gap-2 text-sm', className)} aria-label="Breadcrumb">
      {showHome && (
        <>
          <Button
            variant="secondary"
            size="sm"
            onClick={() => handleClick('/')}
            onKeyDown={(e) => handleKeyDown(e, '/')}
            className="!p-2"
            aria-label="Home"
          >
            <Home className="w-4 h-4" />
          </Button>
          {items.length > 0 && <ChevronRight className="w-4 h-4 text-gray-400" />}
        </>
      )}
      {items.map((item, index) => {
        const isLast = index === items.length - 1
        return (
          <div key={index} className="flex items-center gap-2">
            {item.href && !isLast ? (
              <button
                onClick={() => handleClick(item.href)}
                onKeyDown={(e) => handleKeyDown(e, item.href)}
                className="text-gray-600 hover:text-gray-900 transition-colors"
                tabIndex={0}
                aria-label={item.label}
              >
                {item.icon}
                {item.label}
              </button>
            ) : (
              <span className={cn('flex items-center gap-1', isLast && 'text-gray-900 font-medium')}>
                {item.icon}
                {item.label}
              </span>
            )}
            {!isLast && <ChevronRight className="w-4 h-4 text-gray-400" />}
          </div>
        )
      })}
    </nav>
  )
}

export default Breadcrumbs

