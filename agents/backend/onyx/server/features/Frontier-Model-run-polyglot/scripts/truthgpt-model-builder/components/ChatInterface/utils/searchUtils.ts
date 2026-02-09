/**
 * Utility functions for search operations
 */

export interface SearchResult {
  messageId: string
  matches: number[]
  score: number
}

/**
 * Simple text search in content
 */
export function searchInText(text: string, query: string): number[] {
  const positions: number[] = []
  const lowerText = text.toLowerCase()
  const lowerQuery = query.toLowerCase()
  let startIndex = 0

  while ((startIndex = lowerText.indexOf(lowerQuery, startIndex)) !== -1) {
    positions.push(startIndex)
    startIndex += lowerQuery.length
  }

  return positions
}

/**
 * Fuzzy search with scoring
 */
export function fuzzySearch(text: string, query: string): { score: number, matches: number[] } {
  const lowerText = text.toLowerCase()
  const lowerQuery = query.toLowerCase()

  // Exact match
  if (lowerText.includes(lowerQuery)) {
    return {
      score: 1.0,
      matches: searchInText(text, query),
    }
  }

  // Word-based matching
  const textWords = lowerText.split(/\s+/)
  const queryWords = lowerQuery.split(/\s+/)
  
  let matchCount = 0
  for (const queryWord of queryWords) {
    if (textWords.some(word => word.includes(queryWord))) {
      matchCount++
    }
  }

  const score = matchCount / queryWords.length
  return {
    score,
    matches: [],
  }
}

/**
 * Search messages with ranking
 */
export function searchMessages(
  messages: Array<{ id: string, content: string }>,
  query: string
): SearchResult[] {
  if (!query.trim()) {
    return []
  }

  const results: SearchResult[] = []

  for (const message of messages) {
    const content = typeof message.content === 'string' ? message.content : ''
    const search = fuzzySearch(content, query)
    
    if (search.score > 0) {
      results.push({
        messageId: message.id,
        matches: search.matches,
        score: search.score,
      })
    }
  }

  // Sort by score (highest first)
  return results.sort((a, b) => b.score - a.score)
}

/**
 * Build search index for fast searching
 */
export function buildSearchIndex(
  messages: Array<{ id: string, content: string }>
): Map<string, Set<string>> {
  const index = new Map<string, Set<string>>()

  for (const message of messages) {
    const content = typeof message.content === 'string' ? message.content.toLowerCase() : ''
    const words = content.split(/\s+/).filter(word => word.length > 2)

    for (const word of words) {
      if (!index.has(word)) {
        index.set(word, new Set())
      }
      index.get(word)!.add(message.id)
    }
  }

  return index
}

/**
 * Search using index
 */
export function searchWithIndex(
  index: Map<string, Set<string>>,
  query: string
): Set<string> {
  const queryWords = query.toLowerCase().split(/\s+/).filter(word => word.length > 2)
  const results = new Set<string>()

  for (const word of queryWords) {
    const messageIds = index.get(word)
    if (messageIds) {
      messageIds.forEach(id => results.add(id))
    }
  }

  return results
}

/**
 * Highlight search terms in text
 */
export function highlightSearchTerms(text: string, query: string): string {
  if (!query.trim()) return text

  const regex = new RegExp(`(${query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi')
  return text.replace(regex, '<mark>$1</mark>')
}

/**
 * Get search suggestions based on query
 */
export function getSearchSuggestions(
  messages: Array<{ id: string, content: string }>,
  query: string,
  maxSuggestions: number = 5
): string[] {
  if (!query.trim() || query.length < 2) {
    return []
  }

  const suggestions = new Set<string>()
  const lowerQuery = query.toLowerCase()

  for (const message of messages) {
    const content = typeof message.content === 'string' ? message.content : ''
    const words = content.split(/\s+/)

    for (const word of words) {
      const lowerWord = word.toLowerCase()
      if (lowerWord.startsWith(lowerQuery) && lowerWord.length > lowerQuery.length) {
        suggestions.add(word)
        if (suggestions.size >= maxSuggestions) {
          return Array.from(suggestions)
        }
      }
    }
  }

  return Array.from(suggestions)
}




