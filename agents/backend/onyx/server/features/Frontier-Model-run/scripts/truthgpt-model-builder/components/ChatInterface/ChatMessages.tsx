'use client'

import { memo } from 'react'
import Message from '../Message'
import { Message as MessageType } from './types'

interface ChatMessagesProps {
  messages: MessageType[]
  isLoading: boolean
  highlightSearch?: string
  viewMode?: 'normal' | 'compact' | 'comfortable'
}

function ChatMessagesComponent({
  messages,
  isLoading,
  highlightSearch,
  viewMode = 'normal',
}: ChatMessagesProps) {
  const viewModeClasses = {
    normal: 'space-y-4',
    compact: 'space-y-2',
    comfortable: 'space-y-6',
  }

  return (
    <div className={`flex flex-col ${viewModeClasses[viewMode]}`}>
      {messages.map((message) => (
        <Message
          key={message.id}
          role={message.role}
          content={message.content}
          timestamp={message.timestamp}
          highlight={highlightSearch}
        />
      ))}
      {isLoading && (
        <div className="flex items-center gap-2 text-slate-400">
          <div className="w-2 h-2 bg-purple-500 rounded-full animate-bounce" />
          <div className="w-2 h-2 bg-purple-500 rounded-full animate-bounce delay-75" />
          <div className="w-2 h-2 bg-purple-500 rounded-full animate-bounce delay-150" />
          <span className="ml-2">Generando modelo...</span>
        </div>
      )}
    </div>
  )
}

export default memo(ChatMessagesComponent)

