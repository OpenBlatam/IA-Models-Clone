/**
 * Validation utilities
 */

export const validation = {
  email: (email: string): boolean => {
    const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/
    return re.test(email)
  },

  url: (url: string): boolean => {
    try {
      new URL(url)
      return true
    } catch {
      return false
    }
  },

  githubRepo: (repo: string): boolean => {
    const re = /^[a-zA-Z0-9._-]+\/[a-zA-Z0-9._-]+$/
    return re.test(repo)
  },

  required: (value: string | null | undefined): boolean => {
    return value !== null && value !== undefined && value.trim() !== ''
  },

  minLength: (value: string, min: number): boolean => {
    return value.length >= min
  },

  maxLength: (value: string, max: number): boolean => {
    return value.length <= max
  },

  fileSize: (file: File, maxSizeMB: number): boolean => {
    const maxSizeBytes = maxSizeMB * 1024 * 1024
    return file.size <= maxSizeBytes
  },

  fileType: (file: File, allowedTypes: string[]): boolean => {
    return allowedTypes.some((type) => file.type.includes(type))
  },
}




