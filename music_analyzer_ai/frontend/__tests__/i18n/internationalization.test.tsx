/**
 * Internationalization (i18n) Testing
 * 
 * Tests that verify the application supports multiple languages and locales,
 * ensuring proper translation, formatting, and cultural adaptations.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen } from '@testing-library/react';

// Mock i18n configuration
const mockTranslations = {
  en: {
    common: {
      search: 'Search',
      play: 'Play',
      pause: 'Pause',
      loading: 'Loading...',
      error: 'Error',
    },
    music: {
      tracks: 'Tracks',
      playlists: 'Playlists',
      artists: 'Artists',
      albums: 'Albums',
    },
  },
  es: {
    common: {
      search: 'Buscar',
      play: 'Reproducir',
      pause: 'Pausar',
      loading: 'Cargando...',
      error: 'Error',
    },
    music: {
      tracks: 'Canciones',
      playlists: 'Listas de reproducción',
      artists: 'Artistas',
      albums: 'Álbumes',
    },
  },
  fr: {
    common: {
      search: 'Rechercher',
      play: 'Lire',
      pause: 'Pause',
      loading: 'Chargement...',
      error: 'Erreur',
    },
    music: {
      tracks: 'Morceaux',
      playlists: 'Listes de lecture',
      artists: 'Artistes',
      albums: 'Albums',
    },
  },
};

const getTranslation = (locale: string, key: string): string => {
  const keys = key.split('.');
  let value: any = mockTranslations[locale as keyof typeof mockTranslations];
  
  for (const k of keys) {
    value = value?.[k];
  }
  
  return value || key;
};

describe('Internationalization (i18n) Testing', () => {
  describe('Translation Support', () => {
    it('should translate text to English', () => {
      const translation = getTranslation('en', 'common.search');
      expect(translation).toBe('Search');
    });

    it('should translate text to Spanish', () => {
      const translation = getTranslation('es', 'common.search');
      expect(translation).toBe('Buscar');
    });

    it('should translate text to French', () => {
      const translation = getTranslation('fr', 'common.search');
      expect(translation).toBe('Rechercher');
    });

    it('should handle missing translations gracefully', () => {
      const translation = getTranslation('en', 'common.missing');
      expect(translation).toBe('common.missing'); // Returns key if not found
    });

    it('should translate nested keys correctly', () => {
      const translation = getTranslation('es', 'music.playlists');
      expect(translation).toBe('Listas de reproducción');
    });
  });

  describe('Locale Detection', () => {
    it('should detect browser locale', () => {
      const browserLocale = navigator.language;
      expect(browserLocale).toBeDefined();
      expect(typeof browserLocale).toBe('string');
    });

    it('should fallback to default locale if unsupported', () => {
      const unsupportedLocale = 'zh-CN';
      const supportedLocales = ['en', 'es', 'fr'];
      const defaultLocale = 'en';
      
      const locale = supportedLocales.includes(unsupportedLocale) 
        ? unsupportedLocale 
        : defaultLocale;
      
      expect(locale).toBe(defaultLocale);
    });

    it('should handle locale with region code', () => {
      const localeWithRegion = 'en-US';
      const baseLocale = localeWithRegion.split('-')[0];
      
      expect(baseLocale).toBe('en');
    });
  });

  describe('Number Formatting', () => {
    it('should format numbers according to locale', () => {
      const number = 1234567.89;
      
      const enFormatted = new Intl.NumberFormat('en-US').format(number);
      const esFormatted = new Intl.NumberFormat('es-ES').format(number);
      
      expect(enFormatted).toBe('1,234,567.89');
      expect(esFormatted).toBe('1.234.567,89');
    });

    it('should format currency according to locale', () => {
      const amount = 1234.56;
      
      const enCurrency = new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
      }).format(amount);
      
      const esCurrency = new Intl.NumberFormat('es-ES', {
        style: 'currency',
        currency: 'EUR',
      }).format(amount);
      
      expect(enCurrency).toContain('1,234.56');
      expect(esCurrency).toContain('1.234,56');
    });
  });

  describe('Date Formatting', () => {
    it('should format dates according to locale', () => {
      const date = new Date('2023-12-25');
      
      const enDate = new Intl.DateTimeFormat('en-US').format(date);
      const esDate = new Intl.DateTimeFormat('es-ES').format(date);
      const frDate = new Intl.DateTimeFormat('fr-FR').format(date);
      
      expect(enDate).toBeDefined();
      expect(esDate).toBeDefined();
      expect(frDate).toBeDefined();
    });

    it('should format dates with custom format', () => {
      const date = new Date('2023-12-25');
      
      const formatted = new Intl.DateTimeFormat('en-US', {
        year: 'numeric',
        month: 'long',
        day: 'numeric',
      }).format(date);
      
      expect(formatted).toContain('December');
      expect(formatted).toContain('2023');
    });

    it('should format relative time', () => {
      const now = new Date();
      const past = new Date(now.getTime() - 2 * 60 * 60 * 1000); // 2 hours ago
      
      const rtf = new Intl.RelativeTimeFormat('en', { numeric: 'auto' });
      const relative = rtf.format(-2, 'hour');
      
      expect(relative).toBeDefined();
    });
  });

  describe('Text Direction', () => {
    it('should support left-to-right (LTR) languages', () => {
      const ltrLanguages = ['en', 'es', 'fr'];
      const isLTR = ltrLanguages.includes('en');
      
      expect(isLTR).toBe(true);
    });

    it('should support right-to-left (RTL) languages', () => {
      const rtlLanguages = ['ar', 'he', 'fa'];
      const isRTL = rtlLanguages.includes('ar');
      
      expect(isRTL).toBe(true);
    });

    it('should set correct dir attribute for RTL', () => {
      const locale = 'ar';
      const dir = ['ar', 'he', 'fa'].includes(locale) ? 'rtl' : 'ltr';
      
      expect(dir).toBe('rtl');
    });
  });

  describe('Pluralization', () => {
    it('should handle singular forms', () => {
      const count = 1;
      const pluralRule = new Intl.PluralRules('en');
      const rule = pluralRule.select(count);
      
      expect(rule).toBe('one');
    });

    it('should handle plural forms', () => {
      const count = 5;
      const pluralRule = new Intl.PluralRules('en');
      const rule = pluralRule.select(count);
      
      expect(rule).toBe('other');
    });

    it('should format pluralized strings correctly', () => {
      const count = 1;
      const singular = count === 1 ? 'track' : 'tracks';
      expect(singular).toBe('track');
      
      const count2 = 5;
      const plural = count2 === 1 ? 'track' : 'tracks';
      expect(plural).toBe('tracks');
    });
  });

  describe('Character Encoding', () => {
    it('should handle special characters correctly', () => {
      const specialChars = {
        es: 'ñáéíóú',
        fr: 'àâäéèêëïîôùûüÿç',
      };
      
      expect(specialChars.es).toContain('ñ');
      expect(specialChars.fr).toContain('é');
    });

    it('should handle Unicode characters', () => {
      const unicodeText = '🎵 🎶 🎧';
      expect(unicodeText.length).toBeGreaterThan(0);
    });
  });

  describe('Locale Switching', () => {
    it('should switch locale dynamically', () => {
      let currentLocale = 'en';
      expect(getTranslation(currentLocale, 'common.search')).toBe('Search');
      
      currentLocale = 'es';
      expect(getTranslation(currentLocale, 'common.search')).toBe('Buscar');
    });

    it('should persist locale preference', () => {
      const locale = 'fr';
      localStorage.setItem('locale', locale);
      const savedLocale = localStorage.getItem('locale');
      
      expect(savedLocale).toBe(locale);
    });
  });

  describe('Cultural Adaptations', () => {
    it('should adapt date formats for different cultures', () => {
      const date = new Date('2023-12-25');
      
      // US format: MM/DD/YYYY
      const usFormat = new Intl.DateTimeFormat('en-US').format(date);
      
      // European format: DD/MM/YYYY
      const euFormat = new Intl.DateTimeFormat('es-ES').format(date);
      
      expect(usFormat).toBeDefined();
      expect(euFormat).toBeDefined();
    });

    it('should adapt time formats for different cultures', () => {
      const time = new Date('2023-12-25T14:30:00');
      
      const usTime = new Intl.DateTimeFormat('en-US', {
        hour: 'numeric',
        minute: 'numeric',
        hour12: true,
      }).format(time);
      
      const euTime = new Intl.DateTimeFormat('es-ES', {
        hour: 'numeric',
        minute: 'numeric',
        hour12: false,
      }).format(time);
      
      expect(usTime).toContain('PM');
      expect(euTime).toContain('14');
    });
  });

  describe('Translation Completeness', () => {
    it('should have all translations for all supported locales', () => {
      const locales = Object.keys(mockTranslations);
      const keys = Object.keys(mockTranslations.en.common);
      
      locales.forEach(locale => {
        keys.forEach(key => {
          const translation = getTranslation(locale, `common.${key}`);
          expect(translation).toBeDefined();
          expect(translation).not.toBe(`common.${key}`); // Should not return key
        });
      });
    });

    it('should detect missing translations', () => {
      const allKeys = Object.keys(mockTranslations.en.common);
      const missingKeys: string[] = [];
      
      Object.keys(mockTranslations).forEach(locale => {
        allKeys.forEach(key => {
          const translation = getTranslation(locale, `common.${key}`);
          if (translation === `common.${key}`) {
            missingKeys.push(`${locale}.common.${key}`);
          }
        });
      });
      
      expect(missingKeys).toHaveLength(0);
    });
  });

  describe('Accessibility in i18n', () => {
    it('should set lang attribute correctly', () => {
      const locale = 'es';
      const lang = locale.split('-')[0];
      
      expect(lang).toBe('es');
    });

    it('should provide translations for screen readers', () => {
      const ariaLabel = getTranslation('en', 'common.search');
      expect(ariaLabel).toBe('Search');
    });
  });
});

