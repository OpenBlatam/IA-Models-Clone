/**
 * Utility functions for message operations
 */

export interface Message {
  id: string
  role: 'user' | 'assistant' | 'system'
  content: string
  timestamp?: number
  [key: string]: any
}

/**
 * Calculate word count for a message
 */
export function getWordCount(message: Message | string): number {
  const content = typeof message === 'string' ? message : message.content || ''
  return content.trim().split(/\s+/).filter(word => word.length > 0).length
}

/**
 * Calculate character count for a message
 */
export function getCharCount(message: Message | string): number {
  const content = typeof message === 'string' ? message : message.content || ''
  return content.length
}

/**
 * Check if message contains code
 */
export function hasCode(message: Message | string): boolean {
  const content = typeof message === 'string' ? message : message.content || ''
  return content.includes('```') || content.includes('<code>') || /```[\s\S]*```/.test(content)
}

/**
 * Check if message contains links
 */
export function hasLinks(message: Message | string): boolean {
  const content = typeof message === 'string' ? message : message.content || ''
  return /https?:\/\//.test(content)
}

/**
 * Extract links from message
 */
export function extractLinks(message: Message | string): string[] {
  const content = typeof message === 'string' ? message : message.content || ''
  const linkRegex = /https?:\/\/[^\s]+/g
  return content.match(linkRegex) || []
}

/**
 * Format message timestamp
 */
export function formatTimestamp(timestamp: number | Date | undefined): string {
  if (!timestamp) return ''
  
  const date = timestamp instanceof Date ? timestamp : new Date(timestamp)
  return date.toLocaleString()
}

/**
 * Check if message is recent (within last hour)
 */
export function isRecent(message: Message): boolean {
  if (!message.timestamp) return false
  const oneHourAgo = Date.now() - 3600000
  return message.timestamp > oneHourAgo
}

/**
 * Group messages by time
 */
export function groupMessagesByTime(messages: Message[], intervalMinutes: number = 30): Map<string, Message[]> {
  const groups = new Map<string, Message[]>()
  
  messages.forEach(message => {
    if (!message.timestamp) return
    
    const date = new Date(message.timestamp)
    const groupKey = `${date.getFullYear()}-${date.getMonth()}-${date.getDate()}-${Math.floor(date.getHours() * 60 + date.getMinutes() / intervalMinutes)}`
    
    if (!groups.has(groupKey)) {
      groups.set(groupKey, [])
    }
    groups.get(groupKey)!.push(message)
  })
  
  return groups
}

/**
 * Filter messages by role
 */
export function filterByRole(messages: Message[], role: 'all' | 'user' | 'assistant' | 'system'): Message[] {
  if (role === 'all') return messages
  return messages.filter(msg => msg.role === role)
}

/**
 * Sort messages by timestamp
 */
export function sortByTimestamp(messages: Message[], ascending: boolean = true): Message[] {
  return [...messages].sort((a, b) => {
    const aTime = a.timestamp || 0
    const bTime = b.timestamp || 0
    return ascending ? aTime - bTime : bTime - aTime
  })
}

/**
 * Find duplicate messages
 */
export function findDuplicates(messages: Message[], threshold: number = 0.8): Map<string, string[]> {
  const duplicates = new Map<string, string[]>()
  
  for (let i = 0; i < messages.length; i++) {
    for (let j = i + 1; j < messages.length; j++) {
      const similarity = calculateSimilarity(messages[i].content, messages[j].content)
      if (similarity >= threshold) {
        const key = messages[i].id
        if (!duplicates.has(key)) {
          duplicates.set(key, [])
        }
        duplicates.get(key)!.push(messages[j].id)
      }
    }
  }
  
  return duplicates
}

/**
 * Calculate similarity between two strings (simple Jaccard similarity)
 */
function calculateSimilarity(str1: string, str2: string): number {
  const words1 = new Set(str1.toLowerCase().split(/\s+/))
  const words2 = new Set(str2.toLowerCase().split(/\s+/))
  
  const intersection = new Set([...words1].filter(x => words2.has(x)))
  const union = new Set([...words1, ...words2])
  
  return intersection.size / union.size
}

/**
 * Truncate message content
 */
export function truncateMessage(message: Message | string, maxLength: number = 100): string {
  const content = typeof message === 'string' ? message : message.content || ''
  if (content.length <= maxLength) return content
  return content.substring(0, maxLength - 3) + '...'
}

/**
 * Sanitize message content for display
 */
export function sanitizeMessage(message: Message | string): string {
  const content = typeof message === 'string' ? message : message.content || ''
  // Remove potentially dangerous HTML but keep safe formatting
  return content
    .replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '')
    .replace(/<iframe\b[^<]*(?:(?!<\/iframe>)<[^<]*)*<\/iframe>/gi, '')
}




