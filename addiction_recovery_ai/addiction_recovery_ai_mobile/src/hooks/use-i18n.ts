import { useTranslation } from 'react-i18next';
import { useMemo } from 'react';
import {
  formatCurrency,
  formatNumber,
  formatDate,
  formatRelativeTime,
  isRTL,
} from '@/utils/i18n-helpers';

export function useI18n() {
  const { t, i18n } = useTranslation();

  const formatters = useMemo(
    () => ({
      currency: (amount: number, currency?: string) =>
        formatCurrency(amount, currency),
      number: (value: number, options?: Intl.NumberFormatOptions) =>
        formatNumber(value, options),
      date: (date: Date | string, options?: Intl.DateTimeFormatOptions) =>
        formatDate(date, options),
      relativeTime: (date: Date | string) => formatRelativeTime(date),
    }),
    []
  );

  return {
    t,
    language: i18n.language,
    changeLanguage: i18n.changeLanguage,
    isRTL: isRTL(),
    ...formatters,
  };
}

