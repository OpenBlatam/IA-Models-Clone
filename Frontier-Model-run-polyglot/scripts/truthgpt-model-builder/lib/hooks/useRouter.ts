/**
 * Hook useRouter
 * ===============
 * 
 * Hook para manejo de routing
 */

import { useState, useEffect, useCallback } from 'react'
import {
  getCurrentPath,
  navigateTo,
  navigateBack,
  navigateForward,
  isCurrentPath,
  extractRouteParams,
  buildRoute
} from '../utils/routingUtils'

export interface UseRouterOptions {
  onRouteChange?: (path: string) => void
}

/**
 * Hook para manejo de routing
 */
export function useRouter(options: UseRouterOptions = {}) {
  const { onRouteChange } = options
  const [currentPath, setCurrentPath] = useState(getCurrentPath())

  useEffect(() => {
    const handlePopState = () => {
      const newPath = getCurrentPath()
      setCurrentPath(newPath)
      onRouteChange?.(newPath)
    }

    window.addEventListener('popstate', handlePopState)
    return () => window.removeEventListener('popstate', handlePopState)
  }, [onRouteChange])

  const navigate = useCallback((path: string, options?: { replace?: boolean; state?: any }) => {
    navigateTo(path, options)
    setCurrentPath(getCurrentPath())
    onRouteChange?.(getCurrentPath())
  }, [onRouteChange])

  const back = useCallback(() => {
    navigateBack()
    setCurrentPath(getCurrentPath())
  }, [])

  const forward = useCallback(() => {
    navigateForward()
    setCurrentPath(getCurrentPath())
  }, [])

  const isActive = useCallback((path: string) => {
    return isCurrentPath(path)
  }, [])

  const getParams = useCallback((pattern: string) => {
    return extractRouteParams(currentPath, pattern)
  }, [currentPath])

  const build = useCallback((pattern: string, params: Record<string, string | number>) => {
    return buildRoute(pattern, params)
  }, [])

  return {
    currentPath,
    navigate,
    back,
    forward,
    isActive,
    getParams,
    build
  }
}






