/**
 * useTranslation Hook
 * ===================
 * Custom hook for translations with better TypeScript support
 */

import { useTranslation as useI18nTranslation } from 'react-i18next';

export function useTranslation() {
  const { t, i18n } = useI18nTranslation();

  const changeLanguage = (lang: 'en' | 'es') => {
    i18n.changeLanguage(lang);
  };

  const currentLanguage = i18n.language as 'en' | 'es';

  return {
    t,
    changeLanguage,
    currentLanguage,
    isEnglish: currentLanguage === 'en',
    isSpanish: currentLanguage === 'es',
  };
}



