export const numberUtils = {
  format: (num: number, decimals = 0): string => {
    return new Intl.NumberFormat('en-US', {
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals,
    }).format(num)
  },

  formatCurrency: (amount: number, currency = 'USD'): string => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency,
    }).format(amount)
  },

  formatPercent: (value: number, decimals = 1): string => {
    return `${value.toFixed(decimals)}%`
  },

  clamp: (value: number, min: number, max: number): number => {
    return Math.min(Math.max(value, min), max)
  },

  random: (min: number, max: number): number => {
    return Math.floor(Math.random() * (max - min + 1)) + min
  },

  round: (value: number, decimals = 0): number => {
    const factor = Math.pow(10, decimals)
    return Math.round(value * factor) / factor
  },
}

