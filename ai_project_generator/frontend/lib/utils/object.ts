export const objectUtils = {
  pick: <T extends Record<string, unknown>, K extends keyof T>(obj: T, keys: K[]): Pick<T, K> => {
    const result = {} as Pick<T, K>
    keys.forEach((key) => {
      if (key in obj) {
        result[key] = obj[key]
      }
    })
    return result
  },

  omit: <T extends Record<string, unknown>, K extends keyof T>(obj: T, keys: K[]): Omit<T, K> => {
    const result = { ...obj }
    keys.forEach((key) => {
      delete result[key]
    })
    return result
  },

  isEmpty: (obj: Record<string, unknown>): boolean => {
    return Object.keys(obj).length === 0
  },

  deepMerge: <T extends Record<string, unknown>>(target: T, source: Partial<T>): T => {
    const output = { ...target }
    Object.keys(source).forEach((key) => {
      const sourceValue = source[key]
      const targetValue = target[key]

      if (
        sourceValue &&
        typeof sourceValue === 'object' &&
        !Array.isArray(sourceValue) &&
        targetValue &&
        typeof targetValue === 'object' &&
        !Array.isArray(targetValue)
      ) {
        output[key] = objectUtils.deepMerge(targetValue as T, sourceValue as Partial<T>)
      } else {
        output[key] = sourceValue as T[Extract<keyof T, string>]
      }
    })
    return output
  },

  get: <T,>(obj: Record<string, unknown>, path: string, defaultValue?: T): T | undefined => {
    const keys = path.split('.')
    let result: unknown = obj

    for (const key of keys) {
      if (result && typeof result === 'object' && key in result) {
        result = (result as Record<string, unknown>)[key]
      } else {
        return defaultValue
      }
    }

    return result as T
  },
}

