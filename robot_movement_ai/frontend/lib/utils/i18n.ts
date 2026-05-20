/**
 * Internationalization utilities
 */

export type Locale = 'es' | 'en' | 'fr' | 'de' | 'it' | 'pt';

export interface Translations {
  [key: string]: string | Translations;
}

class I18n {
  private locale: Locale = 'es';
  private translations: Record<Locale, Translations> = {} as Record<Locale, Translations>;
  private fallbackLocale: Locale = 'es';

  setLocale(locale: Locale) {
    this.locale = locale;
    if (typeof window !== 'undefined') {
      localStorage.setItem('locale', locale);
    }
  }

  getLocale(): Locale {
    if (typeof window !== 'undefined') {
      const stored = localStorage.getItem('locale') as Locale;
      if (stored) {
        this.locale = stored;
      }
    }
    return this.locale;
  }

  setTranslations(locale: Locale, translations: Translations) {
    this.translations[locale] = translations;
  }

  setFallbackLocale(locale: Locale) {
    this.fallbackLocale = locale;
  }

  t(key: string, params?: Record<string, string | number>): string {
    const translation = this.getTranslation(key);

    if (!translation) {
      return key;
    }

    if (params) {
      return this.interpolate(translation, params);
    }

    return translation;
  }

  private getTranslation(key: string): string | null {
    const keys = key.split('.');
    let value: any = this.translations[this.locale] || this.translations[this.fallbackLocale];

    for (const k of keys) {
      if (value && typeof value === 'object' && k in value) {
        value = value[k];
      } else {
        // Try fallback
        value = this.translations[this.fallbackLocale];
        for (const fk of keys) {
          if (value && typeof value === 'object' && fk in value) {
            value = value[fk];
          } else {
            return null;
          }
        }
        break;
      }
    }

    return typeof value === 'string' ? value : null;
  }

  private interpolate(template: string, params: Record<string, string | number>): string {
    return template.replace(/\{\{(\w+)\}\}/g, (match, key) => {
      return params[key] !== undefined ? String(params[key]) : match;
    });
  }

  formatNumber(value: number, options?: Intl.NumberFormatOptions): string {
    return new Intl.NumberFormat(this.locale, options).format(value);
  }

  formatDate(date: Date, options?: Intl.DateTimeFormatOptions): string {
    return new Intl.DateTimeFormat(this.locale, options).format(date);
  }

  formatCurrency(value: number, currency: string = 'EUR'): string {
    return new Intl.NumberFormat(this.locale, {
      style: 'currency',
      currency,
    }).format(value);
  }
}

export const i18n = new I18n();



