'use client';

import { useMemo } from 'react';
import { i18n } from '@/lib/i18n';

export function useI18n() {
  const locale = useMemo(() => i18n.getLocale(), []);

  const t = (key: string, params?: Record<string, string | number>) => {
    return i18n.t(key, params);
  };

  const setLocale = (locale: string) => {
    i18n.setLocale(locale);
  };

  return {
    t,
    locale,
    setLocale,
    hasTranslation: (key: string) => i18n.hasTranslation(key),
    getAvailableLocales: () => i18n.getAvailableLocales(),
  };
}

