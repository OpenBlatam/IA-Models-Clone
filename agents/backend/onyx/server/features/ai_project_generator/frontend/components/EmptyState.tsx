'use client'

import clsx from 'clsx'
import type { LucideIcon } from 'lucide-react'

interface EmptyStateProps {
  icon: LucideIcon
  title: string
  description?: string
  action?: {
    label: string
    onClick: () => void
  }
  className?: string
}

const EmptyState = ({ icon: Icon, title, description, action, className }: EmptyStateProps) => {
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if ((e.key === 'Enter' || e.key === ' ') && action) {
      e.preventDefault()
      action.onClick()
    }
  }

  return (
    <div
      className={clsx(
        'flex flex-col items-center justify-center py-12 px-4 text-center',
        className
      )}
    >
      <Icon className="w-16 h-16 text-gray-400 mb-4" aria-hidden="true" />
      <h3 className="text-lg font-semibold text-gray-900 mb-2">{title}</h3>
      {description && <p className="text-gray-500 max-w-md mb-6">{description}</p>}
      {action && (
        <button
          onClick={action.onClick}
          onKeyDown={handleKeyDown}
          className="btn btn-primary"
          tabIndex={0}
          aria-label={action.label}
        >
          {action.label}
        </button>
      )}
    </div>
  )
}

export default EmptyState

