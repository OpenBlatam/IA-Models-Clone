export const queryUtils = {
  parse: (search?: string): Record<string, string> => {
    const params: Record<string, string> = {}
    const queryString = search || (typeof window !== 'undefined' ? window.location.search : '')
    const urlParams = new URLSearchParams(queryString)

    urlParams.forEach((value, key) => {
      params[key] = value
    })

    return params
  },

  stringify: (params: Record<string, string | number | boolean>): string => {
    const searchParams = new URLSearchParams()

    Object.entries(params).forEach(([key, value]) => {
      if (value !== null && value !== undefined) {
        searchParams.append(key, String(value))
      }
    })

    return searchParams.toString()
  },

  get: (key: string, search?: string): string | null => {
    const queryString = search || (typeof window !== 'undefined' ? window.location.search : '')
    const urlParams = new URLSearchParams(queryString)
    return urlParams.get(key)
  },

  set: (key: string, value: string | number | boolean, currentSearch?: string): string => {
    const urlParams = new URLSearchParams(currentSearch || (typeof window !== 'undefined' ? window.location.search : ''))
    urlParams.set(key, String(value))
    return urlParams.toString()
  },

  remove: (key: string, currentSearch?: string): string => {
    const urlParams = new URLSearchParams(currentSearch || (typeof window !== 'undefined' ? window.location.search : ''))
    urlParams.delete(key)
    return urlParams.toString()
  },
}

