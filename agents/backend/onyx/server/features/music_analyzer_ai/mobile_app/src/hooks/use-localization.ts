import { useState, useEffect, useCallback } from 'react';
import * as Localization from 'expo-localization';
import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';

// Initialize i18n if not already initialized
if (!i18n.isInitialized) {
  i18n
    .use(initReactI18next)
    .init({
      compatibilityJSON: 'v3',
      resources: {
        en: {
          translation: {},
        },
        es: {
          translation: {},
        },
      },
      lng: Localization.locale.split('-')[0],
      fallbackLng: 'en',
      interpolation: {
        escapeValue: false,
      },
    });
}

/**
 * Hook for localization
 * Provides translation function and language management
 */
export function useLocalization() {
  const [language, setLanguage] = useState(i18n.language);

  const changeLanguage = useCallback((lng: string) => {
    i18n.changeLanguage(lng);
    setLanguage(lng);
  }, []);

  const t = useCallback(
    (key: string, options?: Record<string, unknown>) => {
      return i18n.t(key, options);
    },
    [language]
  );

  useEffect(() => {
    const handleLanguageChange = (lng: string) => {
      setLanguage(lng);
    };

    i18n.on('languageChanged', handleLanguageChange);

    return () => {
      i18n.off('languageChanged', handleLanguageChange);
    };
  }, []);

  return {
    t,
    language,
    changeLanguage,
    isRTL: i18n.dir() === 'rtl',
  };
}

