import { useMemo } from 'react';
import i18n from '@/i18n/config';

export function useTranslation() {
  const t = useMemo(() => {
    return (key: string, params?: Record<string, string>) => {
      let translation = i18n.t(key);
      
      if (params) {
        Object.keys(params).forEach((param) => {
          translation = translation.replace(`{${param}}`, params[param]);
        });
      }
      
      return translation;
    };
  }, []);

  return { t, locale: i18n.locale };
}


