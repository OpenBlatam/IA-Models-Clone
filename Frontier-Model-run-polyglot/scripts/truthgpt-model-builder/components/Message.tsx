'use client'

import { motion } from 'framer-motion'
import { User, Bot } from 'lucide-react'
import { format } from 'date-fns'
import ReactMarkdown from 'react-markdown'

interface MessageProps {
  message: {
    id: string
    role: 'user' | 'assistant'
    content: string
    timestamp: Date
  }
}

export default function Message({ message }: MessageProps) {
  const isUser = message.role === 'user'

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className={`flex gap-3 ${isUser ? 'justify-end' : 'justify-start'}`}
    >
      {!isUser && (
        <div className="w-8 h-8 rounded-full bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center flex-shrink-0">
          <Bot className="w-5 h-5 text-white" />
        </div>
      )}
      <div
        className={`max-w-[80%] rounded-lg px-4 py-3 ${
          isUser
            ? 'bg-gradient-to-r from-purple-600 to-pink-600 text-white'
            : 'bg-slate-700/50 text-slate-200'
        }`}
      >
        <div className="prose prose-invert max-w-none">
          <ReactMarkdown className="text-sm">{message.content}</ReactMarkdown>
        </div>
        <p className={`text-xs mt-2 ${isUser ? 'text-purple-100' : 'text-slate-400'}`}>
          {format(message.timestamp, 'HH:mm')}
        </p>
      </div>
      {isUser && (
        <div className="w-8 h-8 rounded-full bg-slate-600 flex items-center justify-center flex-shrink-0">
          <User className="w-5 h-5 text-white" />
        </div>
      )}
    </motion.div>
  )
}

