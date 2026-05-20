export const colorUtils = {
  hexToRgb: (hex: string): { r: number; g: number; b: number } | null => {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex)
    return result
      ? {
          r: parseInt(result[1], 16),
          g: parseInt(result[2], 16),
          b: parseInt(result[3], 16),
        }
      : null
  },

  rgbToHex: (r: number, g: number, b: number): string => {
    return `#${[r, g, b].map((x) => x.toString(16).padStart(2, '0')).join('')}`
  },

  getContrastColor: (hex: string): string => {
    const rgb = colorUtils.hexToRgb(hex)
    if (!rgb) {
      return '#000000'
    }

    const brightness = (rgb.r * 299 + rgb.g * 587 + rgb.b * 114) / 1000
    return brightness > 128 ? '#000000' : '#FFFFFF'
  },

  lighten: (hex: string, percent: number): string => {
    const rgb = colorUtils.hexToRgb(hex)
    if (!rgb) {
      return hex
    }

    const factor = 1 + percent / 100
    return colorUtils.rgbToHex(
      Math.min(255, Math.round(rgb.r * factor)),
      Math.min(255, Math.round(rgb.g * factor)),
      Math.min(255, Math.round(rgb.b * factor))
    )
  },

  darken: (hex: string, percent: number): string => {
    const rgb = colorUtils.hexToRgb(hex)
    if (!rgb) {
      return hex
    }

    const factor = 1 - percent / 100
    return colorUtils.rgbToHex(
      Math.max(0, Math.round(rgb.r * factor)),
      Math.max(0, Math.round(rgb.g * factor)),
      Math.max(0, Math.round(rgb.b * factor))
    )
  },
}

