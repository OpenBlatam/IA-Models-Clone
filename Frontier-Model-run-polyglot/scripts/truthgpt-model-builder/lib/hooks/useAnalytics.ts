/**
 * Hook useAnalytics
 * =================
 * 
 * Hook para analytics en componentes
 */

import { useCallback } from 'react'
import { getAnalytics } from '../utils/analyticsUtils'

/**
 * Hook para usar analytics
 */
export function useAnalytics() {
  const analytics = getAnalytics()

  const track = useCallback((eventName: string, properties?: Record<string, any>) => {
    analytics.track(eventName, properties)
  }, [analytics])

  const page = useCallback((pageName: string, properties?: Record<string, any>) => {
    analytics.page(pageName, properties)
  }, [analytics])

  const click = useCallback((elementName: string, properties?: Record<string, any>) => {
    analytics.click(elementName, properties)
  }, [analytics])

  const conversion = useCallback((
    conversionName: string,
    value?: number,
    properties?: Record<string, any>
  ) => {
    analytics.conversion(conversionName, value, properties)
  }, [analytics])

  const error = useCallback((error: Error | string, properties?: Record<string, any>) => {
    analytics.error(error, properties)
  }, [analytics])

  return {
    track,
    page,
    click,
    conversion,
    error
  }
}






