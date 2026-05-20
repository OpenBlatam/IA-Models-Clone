import { useMemo, useState, useCallback } from 'react'

interface UseSearchProps<T> {
  data: T[]
  searchKeys?: (keyof T)[]
  searchFunction?: (item: T, query: string) => boolean
}

export const useSearch = <T extends Record<string, unknown>>({
  data,
  searchKeys,
  searchFunction,
}: UseSearchProps<T>) => {
  const [searchQuery, setSearchQuery] = useState('')

  const filteredData = useMemo(() => {
    if (!searchQuery.trim()) {
      return data
    }

    const query = searchQuery.toLowerCase().trim()

    if (searchFunction) {
      return data.filter((item) => searchFunction(item, query))
    }

    if (searchKeys) {
      return data.filter((item) =>
        searchKeys.some((key) => {
          const value = item[key]
          return value && String(value).toLowerCase().includes(query)
        })
      )
    }

    return data.filter((item) =>
      Object.values(item).some((value) => value && String(value).toLowerCase().includes(query))
    )
  }, [data, searchQuery, searchKeys, searchFunction])

  const clearSearch = useCallback(() => {
    setSearchQuery('')
  }, [])

  return {
    searchQuery,
    setSearchQuery,
    filteredData,
    clearSearch,
    hasResults: filteredData.length > 0,
    resultCount: filteredData.length,
  }
}

