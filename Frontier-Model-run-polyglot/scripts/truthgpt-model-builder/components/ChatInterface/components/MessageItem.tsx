/**
 * MessageItem Component
 * Individual message item with actions and metadata
 */

'use client'

import React, { memo } from 'react'
import { Heart, Pin, Archive, Tag, Bookmark, Reply, Copy, Trash2, MoreVertical } from 'lucide-react'
import { motion } from 'framer-motion'
import { Message } from '../types'

interface MessageItemProps {
  message: Message
  isFavorite?: boolean
  isPinned?: boolean
  isArchived?: boolean
  tags?: string[]
  reactions?: string[]
  showReactions?: boolean
  showTags?: boolean
  showTimestamp?: boolean
  highlightSearch?: string
  onToggleFavorite?: () => void
  onTogglePin?: () => void
  onToggleArchive?: () => void
  onAddTag?: () => void
  onAddReaction?: (reaction: string) => void
  onReply?: () => void
  onCopy?: () => void
  onDelete?: () => void
  onClick?: () => void
  viewMode?: 'normal' | 'compact' | 'comfortable'
}

export const MessageItem = memo(function MessageItem({
  message,
  isFavorite = false,
  isPinned = false,
  isArchived = false,
  tags = [],
  reactions = [],
  showReactions = true,
  showTags = true,
  showTimestamp = true,
  highlightSearch,
  onToggleFavorite,
  onTogglePin,
  onToggleArchive,
  onAddTag,
  onAddReaction,
  onReply,
  onCopy,
  onDelete,
  onClick,
  viewMode = 'normal',
}: MessageItemProps) {
  const [showActions, setShowActions] = React.useState(false)

  const handleCopy = () => {
    navigator.clipboard.writeText(message.content)
    onCopy?.()
  }

  const formatTimestamp = (timestamp: number) => {
    return new Date(timestamp).toLocaleTimeString('es-ES', {
      hour: '2-digit',
      minute: '2-digit',
    })
  }

  const highlightContent = (content: string, query?: string) => {
    if (!query) return content
    const regex = new RegExp(`(${query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi')
    return content.replace(regex, '<mark>$1</mark>')
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -10 }}
      className={`message-item message-item--${message.role} message-item--${viewMode}`}
      onClick={onClick}
    >
      <div className="message-item__header">
        <div className="message-item__role">
          {message.role === 'user' ? '👤 Usuario' : '🤖 Asistente'}
        </div>
        {showTimestamp && (
          <div className="message-item__timestamp">
            {formatTimestamp(message.timestamp)}
          </div>
        )}
        {isPinned && (
          <div className="message-item__badge" title="Mensaje fijado">
            <Pin size={14} />
          </div>
        )}
        <button
          type="button"
          className="message-item__more"
          onClick={(e) => {
            e.stopPropagation()
            setShowActions(!showActions)
          }}
        >
          <MoreVertical size={16} />
        </button>
      </div>

      <div
        className="message-item__content"
        dangerouslySetInnerHTML={{
          __html: highlightContent(message.content, highlightSearch),
        }}
      />

      {showTags && tags.length > 0 && (
        <div className="message-item__tags">
          {tags.map(tag => (
            <span key={tag} className="message-item__tag">
              {tag}
            </span>
          ))}
        </div>
      )}

      {showReactions && reactions.length > 0 && (
        <div className="message-item__reactions">
          {reactions.map(reaction => (
            <span key={reaction} className="message-item__reaction">
              {reaction}
            </span>
          ))}
        </div>
      )}

      {showActions && (
        <div className="message-item__actions" onClick={(e) => e.stopPropagation()}>
          {onToggleFavorite && (
            <button
              type="button"
              onClick={onToggleFavorite}
              className={`message-item__action ${isFavorite ? 'message-item__action--active' : ''}`}
              title={isFavorite ? 'Quitar de favoritos' : 'Agregar a favoritos'}
            >
              <Heart size={16} />
            </button>
          )}
          {onTogglePin && (
            <button
              type="button"
              onClick={onTogglePin}
              className={`message-item__action ${isPinned ? 'message-item__action--active' : ''}`}
              title={isPinned ? 'Desfijar' : 'Fijar'}
            >
              <Pin size={16} />
            </button>
          )}
          {onToggleArchive && (
            <button
              type="button"
              onClick={onToggleArchive}
              className={`message-item__action ${isArchived ? 'message-item__action--active' : ''}`}
              title={isArchived ? 'Desarchivar' : 'Archivar'}
            >
              <Archive size={16} />
            </button>
          )}
          {onAddTag && (
            <button
              type="button"
              onClick={onAddTag}
              className="message-item__action"
              title="Agregar etiqueta"
            >
              <Tag size={16} />
            </button>
          )}
          {onReply && (
            <button
              type="button"
              onClick={onReply}
              className="message-item__action"
              title="Responder"
            >
              <Reply size={16} />
            </button>
          )}
          {onCopy && (
            <button
              type="button"
              onClick={handleCopy}
              className="message-item__action"
              title="Copiar"
            >
              <Copy size={16} />
            </button>
          )}
          {onDelete && (
            <button
              type="button"
              onClick={onDelete}
              className="message-item__action message-item__action--danger"
              title="Eliminar"
            >
              <Trash2 size={16} />
            </button>
          )}
        </div>
      )}

      {message.metadata && (
        <div className="message-item__metadata">
          {message.metadata.wordCount && (
            <span>{message.metadata.wordCount} palabras</span>
          )}
          {message.metadata.hasCode && <span>Código</span>}
          {message.metadata.hasLinks && <span>Links</span>}
        </div>
      )}
    </motion.div>
  )
})

export default MessageItem




