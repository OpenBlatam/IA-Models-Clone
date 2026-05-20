/**
 * MessageList Component
 * Optimized list of messages with virtualization support
 */

'use client'

import React, { memo, useMemo } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import Message from '../../Message'
import { useMessageManagement } from '../hooks/useMessageManagement'
import { useSearchAndFilters } from '../hooks/useSearchAndFilters'
import { filterByRole, sortByTimestamp } from '../utils/messageUtils'

interface MessageListProps {
  messages: Array<{
    id: string
    role: 'user' | 'assistant' | 'system'
    content: string
    timestamp?: number
    [key: string]: any
  }>
  viewMode?: 'normal' | 'compact' | 'comfortable'
  filterRole?: 'all' | 'user' | 'assistant'
  searchQuery?: string
  highlightSearch?: boolean
  showReactions?: boolean
  showTags?: boolean
  showTimestamps?: boolean
  onMessageClick?: (messageId: string) => void
}

export const MessageList = memo(function MessageList({
  messages,
  viewMode = 'normal',
  filterRole = 'all',
  searchQuery = '',
  highlightSearch = true,
  showReactions = true,
  showTags = true,
  showTimestamps = true,
  onMessageClick,
}: MessageListProps) {
  const messageIds = useMemo(() => messages.map(m => m.id), [messages])
  const messageManagement = useMessageManagement(messageIds)
  const search = useSearchAndFilters(messages)

  // Apply filters
  const filteredMessages = useMemo(() => {
    let result = messages

    // Apply role filter
    if (filterRole !== 'all') {
      result = filterByRole(result, filterRole)
    }

    // Apply search
    if (searchQuery) {
      search.setSearchQuery(searchQuery)
      result = search.filteredMessages
    }

    // Sort by timestamp
    result = sortByTimestamp(result, true)

    return result
  }, [messages, filterRole, searchQuery, search])

  return (
    <div className={`message-list message-list--${viewMode}`}>
      <AnimatePresence mode="popLayout">
        {filteredMessages.map((message, index) => (
          <motion.div
            key={message.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.2, delay: index * 0.05 }}
          >
            <Message
              id={message.id}
              role={message.role}
              content={message.content}
              timestamp={message.timestamp}
              isFavorite={messageManagement.favoriteMessages.has(message.id)}
              isPinned={messageManagement.pinnedMessages.has(message.id)}
              tags={messageManagement.messageTags.get(message.id) || []}
              reactions={messageManagement.messageReactions.get(message.id) || []}
              showReactions={showReactions}
              showTags={showTags}
              showTimestamp={showTimestamps}
              onToggleFavorite={() => messageManagement.toggleFavorite(message.id)}
              onTogglePin={() => messageManagement.togglePin(message.id)}
              onClick={() => onMessageClick?.(message.id)}
              highlightSearch={highlightSearch && searchQuery ? searchQuery : undefined}
            />
          </motion.div>
        ))}
      </AnimatePresence>
    </div>
  )
})

export default MessageList




