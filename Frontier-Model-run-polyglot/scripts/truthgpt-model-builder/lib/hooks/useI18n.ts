/**
 * Hook useI18n
 * ============
 * 
 * Hook para usar internacionalización
 */

import { useState, useCallback, useEffect } from 'react'
import { Locale, TranslationKey, setLocale, getLocale, t as translate, formatNumber, formatDate, formatRelativeTime } from '../utils/i18n'
import { useLocalStorage } from './useLocalStorage'

export interface UseI18nResult {
  locale: Locale
  setLocale: (locale: Locale) => void
  t: (key: TranslationKey, params?: Record<string, string | number>) => string
  formatNumber: (value: number, options?: Intl.NumberFormatOptions) => string
  formatDate: (date: Date | string, options?: Intl.DateTimeFormatOptions) => string
  formatRelativeTime: (date: Date | string) => string
}

/**
 * Hook para usar internacionalización
 */
export function useI18n(defaultLocale: Locale = 'es'): UseI18nResult {
  const { value: storedLocale, setValue: setStoredLocale } = useLocalStorage<Locale>(
    'locale',
    { defaultValue: defaultLocale }
  )

  const [locale, setLocaleState] = useState<Locale>(storedLocale || defaultLocale)

  useEffect(() => {
    setLocale(locale)
  }, [locale])

  const handleSetLocale = useCallback((newLocale: Locale) => {
    setLocaleState(newLocale)
    setStoredLocale(newLocale)
    setLocale(newLocale)
  }, [setStoredLocale])

  const t = useCallback((key: TranslationKey, params?: Record<string, string | number>) => {
    return translate(key, params)
  }, [])

  return {
    locale,
    setLocale: handleSetLocale,
    t,
    formatNumber: (value: number, options?: Intl.NumberFormatOptions) => formatNumber(value, options),
    formatDate: (date: Date | string, options?: Intl.DateTimeFormatOptions) => formatDate(date, options),
    formatRelativeTime: (date: Date | string) => formatRelativeTime(date)
  }
}







