export const timeUtils = {
  format: (date: Date, format: '12h' | '24h' = '24h'): string => {
    const hours = date.getHours()
    const minutes = date.getMinutes()
    const seconds = date.getSeconds()

    if (format === '12h') {
      const period = hours >= 12 ? 'PM' : 'AM'
      const displayHours = hours % 12 || 12
      return `${displayHours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')} ${period}`
    }

    return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`
  },

  parse: (timeString: string): { hours: number; minutes: number; seconds?: number } | null => {
    const patterns = [
      /^(\d{1,2}):(\d{2})(?::(\d{2}))?\s*(AM|PM)?$/i,
      /^(\d{1,2}):(\d{2})(?::(\d{2}))?$/,
    ]

    for (const pattern of patterns) {
      const match = timeString.match(pattern)
      if (match) {
        let hours = parseInt(match[1], 10)
        const minutes = parseInt(match[2], 10)
        const seconds = match[3] ? parseInt(match[3], 10) : undefined
        const period = match[4]?.toUpperCase()

        if (period === 'PM' && hours !== 12) {
          hours += 12
        } else if (period === 'AM' && hours === 12) {
          hours = 0
        }

        return { hours, minutes, seconds }
      }
    }

    return null
  },

  addHours: (date: Date, hours: number): Date => {
    const newDate = new Date(date)
    newDate.setHours(newDate.getHours() + hours)
    return newDate
  },

  addMinutes: (date: Date, minutes: number): Date => {
    const newDate = new Date(date)
    newDate.setMinutes(newDate.getMinutes() + minutes)
    return newDate
  },

  addSeconds: (date: Date, seconds: number): Date => {
    const newDate = new Date(date)
    newDate.setSeconds(newDate.getSeconds() + seconds)
    return newDate
  },

  difference: (date1: Date, date2: Date): {
    hours: number
    minutes: number
    seconds: number
    totalSeconds: number
  } => {
    const diff = Math.abs(date1.getTime() - date2.getTime())
    const totalSeconds = Math.floor(diff / 1000)
    const hours = Math.floor(totalSeconds / 3600)
    const minutes = Math.floor((totalSeconds % 3600) / 60)
    const seconds = totalSeconds % 60

    return { hours, minutes, seconds, totalSeconds }
  },
}

