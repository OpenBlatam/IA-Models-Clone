/**
 * EmptyState Component
 * Display when there's no content to show
 */

'use client'

import React, { memo } from 'react'
import { MessageSquare, Inbox, Search } from 'lucide-react'

interface EmptyStateProps {
  icon?: 'messages' | 'inbox' | 'search' | React.ReactNode
  title: string
  description?: string
  action?: {
    label: string
    onClick: () => void
  }
}

const iconMap = {
  messages: MessageSquare,
  inbox: Inbox,
  search: Search,
}

export const EmptyState = memo(function EmptyState({
  icon = 'messages',
  title,
  description,
  action,
}: EmptyStateProps) {
  const IconComponent = typeof icon === 'string' ? iconMap[icon] : null

  return (
    <div className="empty-state">
      <div className="empty-state__icon">
        {IconComponent ? (
          <IconComponent size={48} />
        ) : (
          icon
        )}
      </div>
      <h3 className="empty-state__title">{title}</h3>
      {description && (
        <p className="empty-state__description">{description}</p>
      )}
      {action && (
        <button
          type="button"
          onClick={action.onClick}
          className="empty-state__action"
        >
          {action.label}
        </button>
      )}
    </div>
  )
})

export default EmptyState




