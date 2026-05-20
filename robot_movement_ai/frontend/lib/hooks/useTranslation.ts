import { useI18nStore } from '../store/i18nStore';
import { getTranslation } from '../i18n/translations';

export function useTranslation() {
  const language = useI18nStore((state) => state.language);

  const t = (key: string): string => {
    return getTranslation(key, language);
  };

  return { t, language };
}

