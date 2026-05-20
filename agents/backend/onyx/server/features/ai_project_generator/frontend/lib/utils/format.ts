export const formatUtils = {
  currency: (amount: number, currency = 'USD', locale = 'en-US'): string => {
    return new Intl.NumberFormat(locale, {
      style: 'currency',
      currency,
    }).format(amount)
  },

  number: (value: number, locale = 'en-US', options?: Intl.NumberFormatOptions): string => {
    return new Intl.NumberFormat(locale, options).format(value)
  },

  percentage: (value: number, decimals = 0): string => {
    return `${value.toFixed(decimals)}%`
  },

  bytes: (bytes: number): string => {
    if (bytes === 0) {
      return '0 Bytes'
    }

    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB', 'PB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))

    return `${parseFloat((bytes / Math.pow(k, i)).toFixed(2))} ${sizes[i]}`
  },

  duration: (seconds: number): string => {
    const hours = Math.floor(seconds / 3600)
    const minutes = Math.floor((seconds % 3600) / 60)
    const secs = Math.floor(seconds % 60)

    if (hours > 0) {
      return `${hours}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`
    }
    return `${minutes}:${secs.toString().padStart(2, '0')}`
  },

  truncate: (text: string, maxLength: number, suffix = '...'): string => {
    if (text.length <= maxLength) {
      return text
    }
    return text.slice(0, maxLength - suffix.length) + suffix
  },

  initials: (name: string): string => {
    return name
      .split(' ')
      .map((word) => word[0])
      .join('')
      .toUpperCase()
      .slice(0, 2)
  },
}
