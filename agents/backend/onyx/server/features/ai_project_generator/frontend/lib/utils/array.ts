export const arrayUtils = {
  unique: <T,>(array: T[]): T[] => {
    return Array.from(new Set(array))
  },

  groupBy: <T, K extends string | number>(
    array: T[],
    key: (item: T) => K
  ): Record<K, T[]> => {
    return array.reduce((result, item) => {
      const groupKey = key(item)
      if (!result[groupKey]) {
        result[groupKey] = []
      }
      result[groupKey].push(item)
      return result
    }, {} as Record<K, T[]>)
  },

  chunk: <T,>(array: T[], size: number): T[][] => {
    const chunks: T[][] = []
    for (let i = 0; i < array.length; i += size) {
      chunks.push(array.slice(i, i + size))
    }
    return chunks
  },

  shuffle: <T,>(array: T[]): T[] => {
    const shuffled = [...array]
    for (let i = shuffled.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1))
      ;[shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]]
    }
    return shuffled
  },

  sortBy: <T,>(array: T[], key: (item: T) => number | string, order: 'asc' | 'desc' = 'asc'): T[] => {
    const sorted = [...array].sort((a, b) => {
      const aVal = key(a)
      const bVal = key(b)
      if (aVal < bVal) return order === 'asc' ? -1 : 1
      if (aVal > bVal) return order === 'asc' ? 1 : -1
      return 0
    })
    return sorted
  },
}

