type Translations = Record<string, string | Translations>;

class I18n {
  private translations: Record<string, Translations> = {};
  private currentLocale: string = 'es';
  private fallbackLocale: string = 'es';

  setLocale(locale: string) {
    this.currentLocale = locale;
    if (typeof window !== 'undefined') {
      localStorage.setItem('bul_locale', locale);
    }
  }

  getLocale(): string {
    if (typeof window !== 'undefined') {
      const stored = localStorage.getItem('bul_locale');
      if (stored) {
        this.currentLocale = stored;
      }
    }
    return this.currentLocale;
  }

  setTranslations(locale: string, translations: Translations) {
    this.translations[locale] = translations;
  }

  t(key: string, params?: Record<string, string | number>): string {
    const locale = this.getLocale();
    const translation = this.getTranslation(key, locale) || this.getTranslation(key, this.fallbackLocale) || key;

    if (params) {
      return this.interpolate(translation, params);
    }

    return translation;
  }

  private getTranslation(key: string, locale: string): string | null {
    const keys = key.split('.');
    let value: any = this.translations[locale];

    for (const k of keys) {
      if (value && typeof value === 'object' && k in value) {
        value = value[k];
      } else {
        return null;
      }
    }

    return typeof value === 'string' ? value : null;
  }

  private interpolate(template: string, params: Record<string, string | number>): string {
    return template.replace(/\{\{(\w+)\}\}/g, (match, key) => {
      return params[key]?.toString() || match;
    });
  }

  hasTranslation(key: string, locale?: string): boolean {
    const loc = locale || this.getLocale();
    return this.getTranslation(key, loc) !== null;
  }

  getAvailableLocales(): string[] {
    return Object.keys(this.translations);
  }
}

export const i18n = new I18n();

// Default Spanish translations
i18n.setTranslations('es', {
  common: {
    save: 'Guardar',
    cancel: 'Cancelar',
    delete: 'Eliminar',
    edit: 'Editar',
    create: 'Crear',
    search: 'Buscar',
    filter: 'Filtrar',
    loading: 'Cargando...',
    error: 'Error',
    success: 'Éxito',
    confirm: 'Confirmar',
    close: 'Cerrar',
  },
});
