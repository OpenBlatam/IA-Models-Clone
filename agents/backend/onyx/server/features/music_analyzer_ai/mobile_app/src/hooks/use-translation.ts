import { useTranslation as useI18nTranslation } from 'react-i18next';

export function useTranslation() {
  const { t, i18n } = useI18nTranslation();

  const changeLanguage = (lang: string) => {
    i18n.changeLanguage(lang);
  };

  return {
    t,
    currentLanguage: i18n.language,
    changeLanguage,
    isRTL: i18n.dir() === 'rtl',
  };
}

