/**
 * Utilidades de Búsqueda y Filtrado
 * ==================================
 * 
 * Funciones para búsqueda avanzada y filtrado
 */

/**
 * Opciones de búsqueda
 */
export interface SearchOptions {
  caseSensitive?: boolean
  wholeWord?: boolean
  fuzzy?: boolean
  threshold?: number // Para búsqueda fuzzy (0-1)
}

/**
 * Busca texto en una cadena
 */
export function searchText(
  text: string,
  query: string,
  options: SearchOptions = {}
): boolean {
  const {
    caseSensitive = false,
    wholeWord = false,
    fuzzy = false,
    threshold = 0.6
  } = options

  if (!query) return true

  let searchText = text
  let searchQuery = query

  if (!caseSensitive) {
    searchText = searchText.toLowerCase()
    searchQuery = searchQuery.toLowerCase()
  }

  if (wholeWord) {
    const regex = new RegExp(`\\b${escapeRegex(searchQuery)}\\b`, caseSensitive ? 'g' : 'gi')
    return regex.test(searchText)
  }

  if (fuzzy) {
    return fuzzyMatch(searchText, searchQuery, threshold)
  }

  return searchText.includes(searchQuery)
}

/**
 * Búsqueda fuzzy (aproximada)
 */
export function fuzzyMatch(
  text: string,
  query: string,
  threshold: number = 0.6
): boolean {
  if (!query) return true

  const textLower = text.toLowerCase()
  const queryLower = query.toLowerCase()

  let textIndex = 0
  let queryIndex = 0
  let matches = 0

  while (textIndex < textLower.length && queryIndex < queryLower.length) {
    if (textLower[textIndex] === queryLower[queryIndex]) {
      matches++
      queryIndex++
    }
    textIndex++
  }

  const similarity = matches / queryLower.length
  return similarity >= threshold
}

/**
 * Escapa caracteres especiales de regex
 */
function escapeRegex(str: string): string {
  return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

/**
 * Busca en un array de objetos
 */
export function searchInArray<T>(
  array: T[],
  query: string,
  getSearchableText: (item: T) => string,
  options: SearchOptions = {}
): T[] {
  if (!query) return array

  return array.filter(item => {
    const searchableText = getSearchableText(item)
    return searchText(searchableText, query, options)
  })
}

/**
 * Busca en múltiples campos de un objeto
 */
export function searchInObject<T>(
  obj: T,
  query: string,
  fields: (keyof T)[],
  options: SearchOptions = {}
): boolean {
  if (!query) return true

  return fields.some(field => {
    const value = obj[field]
    if (value === null || value === undefined) return false
    return searchText(String(value), query, options)
  })
}

/**
 * Filtra un array con múltiples criterios
 */
export interface FilterCriteria<T> {
  field: keyof T
  operator: 'equals' | 'contains' | 'startsWith' | 'endsWith' | 'gt' | 'gte' | 'lt' | 'lte' | 'in' | 'notIn'
  value: any
  caseSensitive?: boolean
}

export function filterArray<T>(
  array: T[],
  criteria: FilterCriteria<T> | FilterCriteria<T>[]
): T[] {
  const criteriaArray = Array.isArray(criteria) ? criteria : [criteria]

  return array.filter(item => {
    return criteriaArray.every(criterion => {
      const fieldValue = item[criterion.field]
      const { operator, value, caseSensitive = false } = criterion

      let compareValue = fieldValue
      let compareTarget = value

      if (typeof compareValue === 'string' && typeof compareTarget === 'string' && !caseSensitive) {
        compareValue = compareValue.toLowerCase()
        compareTarget = compareTarget.toLowerCase()
      }

      switch (operator) {
        case 'equals':
          return compareValue === compareTarget
        case 'contains':
          return String(compareValue).includes(String(compareTarget))
        case 'startsWith':
          return String(compareValue).startsWith(String(compareTarget))
        case 'endsWith':
          return String(compareValue).endsWith(String(compareTarget))
        case 'gt':
          return compareValue > compareTarget
        case 'gte':
          return compareValue >= compareTarget
        case 'lt':
          return compareValue < compareTarget
        case 'lte':
          return compareValue <= compareTarget
        case 'in':
          return Array.isArray(compareTarget) && compareTarget.includes(compareValue)
        case 'notIn':
          return Array.isArray(compareTarget) && !compareTarget.includes(compareValue)
        default:
          return true
      }
    })
  })
}

/**
 * Ordena un array por múltiples campos
 */
export interface SortCriteria<T> {
  field: keyof T
  direction: 'asc' | 'desc'
}

export function sortArray<T>(
  array: T[],
  criteria: SortCriteria<T> | SortCriteria<T>[]
): T[] {
  const criteriaArray = Array.isArray(criteria) ? criteria : [criteria]
  const sorted = [...array]

  sorted.sort((a, b) => {
    for (const criterion of criteriaArray) {
      const { field, direction } = criterion
      const aValue = a[field]
      const bValue = b[field]

      let comparison = 0

      if (aValue === bValue) {
        comparison = 0
      } else if (aValue === null || aValue === undefined) {
        comparison = 1
      } else if (bValue === null || bValue === undefined) {
        comparison = -1
      } else if (typeof aValue === 'string' && typeof bValue === 'string') {
        comparison = aValue.localeCompare(bValue)
      } else {
        comparison = aValue < bValue ? -1 : 1
      }

      if (comparison !== 0) {
        return direction === 'asc' ? comparison : -comparison
      }
    }
    return 0
  })

  return sorted
}

/**
 * Pagina un array
 */
export function paginateArray<T>(
  array: T[],
  page: number,
  pageSize: number
): { data: T[]; total: number; page: number; pageSize: number; totalPages: number } {
  const total = array.length
  const totalPages = Math.ceil(total / pageSize)
  const start = (page - 1) * pageSize
  const end = start + pageSize
  const data = array.slice(start, end)

  return {
    data,
    total,
    page,
    pageSize,
    totalPages
  }
}

/**
 * Combina búsqueda, filtrado, ordenamiento y paginación
 */
export function searchFilterSortPaginate<T>(
  array: T[],
  options: {
    search?: {
      query: string
      getSearchableText: (item: T) => string
      searchOptions?: SearchOptions
    }
    filter?: FilterCriteria<T> | FilterCriteria<T>[]
    sort?: SortCriteria<T> | SortCriteria<T>[]
    pagination?: {
      page: number
      pageSize: number
    }
  }
): {
  data: T[]
  total: number
  page?: number
  pageSize?: number
  totalPages?: number
} {
  let result = [...array]

  // Búsqueda
  if (options.search) {
    result = searchInArray(
      result,
      options.search.query,
      options.search.getSearchableText,
      options.search.searchOptions
    )
  }

  // Filtrado
  if (options.filter) {
    result = filterArray(result, options.filter)
  }

  const total = result.length

  // Ordenamiento
  if (options.sort) {
    result = sortArray(result, options.sort)
  }

  // Paginación
  if (options.pagination) {
    const { data, page, pageSize, totalPages } = paginateArray(
      result,
      options.pagination.page,
      options.pagination.pageSize
    )
    return {
      data,
      total,
      page,
      pageSize,
      totalPages
    }
  }

  return {
    data: result,
    total
  }
}






