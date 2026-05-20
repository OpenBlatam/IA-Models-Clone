export const urlUtils = {
  parse: (url: string) => {
    try {
      return new URL(url)
    } catch {
      return null
    }
  },

  getQueryParams: (url?: string) => {
    const searchParams = url ? new URL(url).searchParams : new URLSearchParams(window.location.search)
    const params: Record<string, string> = {}
    searchParams.forEach((value, key) => {
      params[key] = value
    })
    return params
  },

  buildQueryString: (params: Record<string, string | number | boolean>) => {
    const searchParams = new URLSearchParams()
    Object.entries(params).forEach(([key, value]) => {
      if (value !== null && value !== undefined) {
        searchParams.append(key, String(value))
      }
    })
    return searchParams.toString()
  },

  isValidUrl: (url: string): boolean => {
    try {
      new URL(url)
      return true
    } catch {
      return false
    }
  },

  getDomain: (url: string): string | null => {
    try {
      return new URL(url).hostname
    } catch {
      return null
    }
  },
}

