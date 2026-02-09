/**
 * Unit Tests - String Utilities
 */

describe('String Utilities', () => {
  describe('Truncation', () => {
    it('should truncate long strings', () => {
      const truncate = (str: string, maxLength: number) => {
        if (str.length <= maxLength) return str
        return str.substring(0, maxLength - 3) + '...'
      }

      expect(truncate('short', 10)).toBe('short')
      expect(truncate('this is a very long string', 10)).toBe('this is...')
      expect(truncate('exact', 5)).toBe('exact')
    })

    it('should truncate at word boundaries', () => {
      const truncateAtWord = (str: string, maxLength: number) => {
        if (str.length <= maxLength) return str
        const truncated = str.substring(0, maxLength)
        const lastSpace = truncated.lastIndexOf(' ')
        if (lastSpace > 0) {
          return truncated.substring(0, lastSpace) + '...'
        }
        return truncated + '...'
      }

      expect(truncateAtWord('this is a test', 10)).toBe('this is...')
    })
  })

  describe('Normalization', () => {
    it('should normalize whitespace', () => {
      const normalize = (str: string) => {
        return str.replace(/\s+/g, ' ').trim()
      }

      expect(normalize('  multiple   spaces  ')).toBe('multiple spaces')
      expect(normalize('tab\tseparated')).toBe('tab separated')
    })

    it('should remove special characters', () => {
      const sanitize = (str: string) => {
        return str.replace(/[^a-zA-Z0-9\s]/g, '')
      }

      expect(sanitize('test@#$%model')).toBe('testmodel')
      expect(sanitize('model-123')).toBe('model123')
    })
  })

  describe('Case Conversion', () => {
    it('should convert to camelCase', () => {
      const toCamelCase = (str: string) => {
        return str
          .replace(/(?:^\w|[A-Z]|\b\w)/g, (word, index) => {
            return index === 0 ? word.toLowerCase() : word.toUpperCase()
          })
          .replace(/\s+/g, '')
      }

      expect(toCamelCase('hello world')).toBe('helloWorld')
      expect(toCamelCase('Hello World')).toBe('helloWorld')
    })

    it('should convert to kebab-case', () => {
      const toKebabCase = (str: string) => {
        return str
          .replace(/([a-z])([A-Z])/g, '$1-$2')
          .toLowerCase()
          .replace(/\s+/g, '-')
      }

      expect(toKebabCase('hello world')).toBe('hello-world')
      expect(toKebabCase('HelloWorld')).toBe('hello-world')
    })

    it('should convert to snake_case', () => {
      const toSnakeCase = (str: string) => {
        return str
          .replace(/([a-z])([A-Z])/g, '$1_$2')
          .toLowerCase()
          .replace(/\s+/g, '_')
      }

      expect(toSnakeCase('hello world')).toBe('hello_world')
      expect(toSnakeCase('HelloWorld')).toBe('hello_world')
    })
  })

  describe('Search and Replace', () => {
    it('should replace all occurrences', () => {
      const replaceAll = (str: string, search: string, replace: string) => {
        return str.split(search).join(replace)
      }

      expect(replaceAll('test test test', 'test', 'model')).toBe('model model model')
    })

    it('should replace case-insensitive', () => {
      const replaceCaseInsensitive = (str: string, search: string, replace: string) => {
        const regex = new RegExp(search, 'gi')
        return str.replace(regex, replace)
      }

      expect(replaceCaseInsensitive('Test TEST test', 'test', 'model')).toBe('model model model')
    })
  })

  describe('Validation', () => {
    it('should check if string is empty', () => {
      const isEmpty = (str: string) => {
        return !str || str.trim().length === 0
      }

      expect(isEmpty('')).toBe(true)
      expect(isEmpty('   ')).toBe(true)
      expect(isEmpty('test')).toBe(false)
    })

    it('should validate email format', () => {
      const isValidEmail = (email: string) => {
        return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)
      }

      expect(isValidEmail('test@example.com')).toBe(true)
      expect(isValidEmail('invalid-email')).toBe(false)
      expect(isValidEmail('test@')).toBe(false)
    })

    it('should validate URL format', () => {
      const isValidURL = (url: string) => {
        try {
          new URL(url)
          return true
        } catch {
          return false
        }
      }

      expect(isValidURL('https://example.com')).toBe(true)
      expect(isValidURL('http://example.com/path')).toBe(true)
      expect(isValidURL('not-a-url')).toBe(false)
    })
  })

  describe('Formatting', () => {
    it('should format numbers with commas', () => {
      const formatNumber = (num: number) => {
        return num.toLocaleString()
      }

      expect(formatNumber(1000)).toBe('1,000')
      expect(formatNumber(1000000)).toBe('1,000,000')
    })

    it('should pad strings', () => {
      const pad = (str: string, length: number, char: string = ' ') => {
        return str.padStart(length, char)
      }

      expect(pad('5', 3, '0')).toBe('005')
      expect(pad('test', 10, ' ')).toBe('      test')
    })
  })
})










