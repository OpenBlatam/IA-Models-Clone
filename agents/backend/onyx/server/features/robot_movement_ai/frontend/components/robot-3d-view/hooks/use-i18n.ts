/**
 * Hook for internationalization
 * @module robot-3d-view/hooks/use-i18n
 */

import { useState, useEffect, useCallback } from 'react';
import { i18nManager, type TranslationKey } from '../utils/i18n';

/**
 * Hook for internationalization
 * 
 * @returns Translation function and language management
 * 
 * @example
 * ```tsx
 * const { t, setLanguage, currentLanguage } = useI18n();
 * <button>{t('common.save')}</button>
 * ```
 */
export function useI18n() {
  const [currentLanguage, setCurrentLanguageState] = useState(() =>
    i18nManager.getCurrentLanguage()
  );

  useEffect(() => {
    // Listen for language changes
    const interval = setInterval(() => {
      const current = i18nManager.getCurrentLanguage();
      if (current !== currentLanguage) {
        setCurrentLanguageState(current);
      }
    }, 100);

    return () => clearInterval(interval);
  }, [currentLanguage]);

  const t = useCallback(
    (key: TranslationKey, params?: Record<string, string | number>) => {
      return i18nManager.t(key, params);
    },
    [currentLanguage]
  );

  const setLanguage = useCallback((code: string) => {
    i18nManager.setLanguage(code);
    setCurrentLanguageState(code);
  }, []);

  const getAvailableLanguages = useCallback(() => {
    return i18nManager.getAvailableLanguages();
  }, []);

  return {
    t,
    setLanguage,
    currentLanguage,
    getAvailableLanguages,
  };
}



