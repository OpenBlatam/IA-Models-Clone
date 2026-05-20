import { format, formatDistance, formatRelative, isToday, isYesterday, isTomorrow, parseISO } from 'date-fns'

export const dateUtils = {
  format: (date: Date | string, formatStr = 'PPp') => {
    const dateObj = typeof date === 'string' ? parseISO(date) : date
    return format(dateObj, formatStr)
  },

  formatDistance: (date: Date | string, baseDate = new Date()) => {
    const dateObj = typeof date === 'string' ? parseISO(date) : date
    const baseDateObj = typeof baseDate === 'string' ? parseISO(baseDate) : baseDate
    return formatDistance(dateObj, baseDateObj, { addSuffix: true })
  },

  formatRelative: (date: Date | string, baseDate = new Date()) => {
    const dateObj = typeof date === 'string' ? parseISO(date) : date
    const baseDateObj = typeof baseDate === 'string' ? parseISO(baseDate) : baseDate
    return formatRelative(dateObj, baseDateObj)
  },

  isToday: (date: Date | string) => {
    const dateObj = typeof date === 'string' ? parseISO(date) : date
    return isToday(dateObj)
  },

  isYesterday: (date: Date | string) => {
    const dateObj = typeof date === 'string' ? parseISO(date) : date
    return isYesterday(dateObj)
  },

  isTomorrow: (date: Date | string) => {
    const dateObj = typeof date === 'string' ? parseISO(date) : date
    return isTomorrow(dateObj)
  },

  getRelativeTime: (date: Date | string) => {
    const dateObj = typeof date === 'string' ? parseISO(date) : date
    if (isToday(dateObj)) {
      return 'Today'
    }
    if (isYesterday(dateObj)) {
      return 'Yesterday'
    }
    if (isTomorrow(dateObj)) {
      return 'Tomorrow'
    }
    return dateUtils.formatDistance(dateObj)
  },
}

