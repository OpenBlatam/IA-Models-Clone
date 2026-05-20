import { useState, useEffect, useCallback } from 'react';
import { i18n, Locale, Translations } from '@/lib/utils/i18n';

export function useI18n() {
  const [locale, setLocaleState] = useState<Locale>(i18n.getLocale());

  useEffect(() => {
    setLocaleState(i18n.getLocale());
  }, []);

  const setLocale = useCallback((newLocale: Locale) => {
    i18n.setLocale(newLocale);
    setLocaleState(newLocale);
  }, []);

  const t = useCallback(
    (key: string, params?: Record<string, string | number>) => {
      return i18n.t(key, params);
    },
    [locale]
  );

  const formatNumber = useCallback(
    (value: number, options?: Intl.NumberFormatOptions) => {
      return i18n.formatNumber(value, options);
    },
    [locale]
  );

  const formatDate = useCallback(
    (date: Date, options?: Intl.DateTimeFormatOptions) => {
      return i18n.formatDate(date, options);
    },
    [locale]
  );

  const formatCurrency = useCallback(
    (value: number, currency: string = 'EUR') => {
      return i18n.formatCurrency(value, currency);
    },
    [locale]
  );

  return {
    locale,
    setLocale,
    t,
    formatNumber,
    formatDate,
    formatCurrency,
  };
}



