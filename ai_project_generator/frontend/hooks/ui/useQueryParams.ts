import { useState, useCallback, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { queryUtils } from '@/lib/utils'

export const useQueryParams = () => {
  const router = useRouter()
  const [params, setParams] = useState<Record<string, string>>({})

  useEffect(() => {
    if (typeof window !== 'undefined') {
      const currentParams = queryUtils.parse(window.location.search)
      setParams(currentParams)
    }
  }, [])

  const setParam = useCallback(
    (key: string, value: string | number | boolean) => {
      if (typeof window === 'undefined') {
        return
      }
      const newParams = { ...params, [key]: String(value) }
      const queryString = queryUtils.stringify(newParams)
      const newUrl = `${window.location.pathname}?${queryString}`
      router.push(newUrl)
      setParams(newParams)
    },
    [params, router]
  )

  const removeParam = useCallback(
    (key: string) => {
      if (typeof window === 'undefined') {
        return
      }
      const newParams = { ...params }
      delete newParams[key]
      const queryString = queryUtils.stringify(newParams)
      const newUrl = queryString ? `${window.location.pathname}?${queryString}` : window.location.pathname
      router.push(newUrl)
      setParams(newParams)
    },
    [params, router]
  )

  const getParam = useCallback(
    (key: string): string | null => {
      return params[key] || null
    },
    [params]
  )

  const clearParams = useCallback(() => {
    if (typeof window === 'undefined') {
      return
    }
    router.push(window.location.pathname)
    setParams({})
  }, [router])

  return {
    params,
    setParam,
    removeParam,
    getParam,
    clearParams,
  }
}

