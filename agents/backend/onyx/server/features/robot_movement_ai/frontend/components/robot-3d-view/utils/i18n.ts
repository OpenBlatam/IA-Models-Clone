/**
 * Internationalization (i18n) system
 * @module robot-3d-view/utils/i18n
 */

/**
 * Translation key
 */
export type TranslationKey = string;

/**
 * Translation dictionary
 */
export interface TranslationDictionary {
  [key: string]: string | TranslationDictionary;
}

/**
 * Language configuration
 */
export interface LanguageConfig {
  code: string;
  name: string;
  nativeName: string;
  dictionary: TranslationDictionary;
}

/**
 * i18n Manager class
 */
export class I18nManager {
  private languages: Map<string, LanguageConfig> = new Map();
  private currentLanguage: string = 'en';
  private fallbackLanguage: string = 'en';

  /**
   * Registers a language
   */
  registerLanguage(config: LanguageConfig): void {
    this.languages.set(config.code, config);
  }

  /**
   * Sets current language
   */
  setLanguage(code: string): void {
    if (this.languages.has(code)) {
      this.currentLanguage = code;
    }
  }

  /**
   * Gets current language
   */
  getCurrentLanguage(): string {
    return this.currentLanguage;
  }

  /**
   * Translates a key
   */
  t(key: TranslationKey, params?: Record<string, string | number>): string {
    const lang = this.languages.get(this.currentLanguage);
    const fallback = this.languages.get(this.fallbackLanguage);

    let value = this.getNestedValue(lang?.dictionary, key) ||
                this.getNestedValue(fallback?.dictionary, key) ||
                key;

    // Replace parameters
    if (params) {
      Object.entries(params).forEach(([paramKey, paramValue]) => {
        value = value.replace(`{{${paramKey}}}`, String(paramValue));
      });
    }

    return value;
  }

  /**
   * Gets nested value from dictionary
   */
  private getNestedValue(
    dict: TranslationDictionary | undefined,
    key: string
  ): string | undefined {
    if (!dict) return undefined;

    const keys = key.split('.');
    let value: string | TranslationDictionary | undefined = dict;

    for (const k of keys) {
      if (typeof value === 'object' && value !== null) {
        value = value[k];
      } else {
        return undefined;
      }
    }

    return typeof value === 'string' ? value : undefined;
  }

  /**
   * Gets available languages
   */
  getAvailableLanguages(): LanguageConfig[] {
    return Array.from(this.languages.values());
  }

  /**
   * Checks if language is available
   */
  isLanguageAvailable(code: string): boolean {
    return this.languages.has(code);
  }
}

/**
 * Global i18n manager instance
 */
export const i18nManager = new I18nManager();

// Register default English translations
i18nManager.registerLanguage({
  code: 'en',
  name: 'English',
  nativeName: 'English',
  dictionary: {
    common: {
      save: 'Save',
      cancel: 'Cancel',
      delete: 'Delete',
      edit: 'Edit',
      close: 'Close',
      open: 'Open',
      export: 'Export',
      import: 'Import',
      copy: 'Copy',
      paste: 'Paste',
      undo: 'Undo',
      redo: 'Redo',
      reset: 'Reset',
      clear: 'Clear',
      search: 'Search',
      filter: 'Filter',
      help: 'Help',
      settings: 'Settings',
    },
    controls: {
      toggleStats: 'Toggle Statistics',
      toggleGizmo: 'Toggle Gizmo',
      toggleGrid: 'Toggle Grid',
      toggleObjects: 'Toggle Objects',
      toggleAutoRotate: 'Toggle Auto Rotate',
      toggleStars: 'Toggle Stars',
      toggleWaypoints: 'Toggle Waypoints',
      screenshot: 'Take Screenshot',
      cameraFront: 'Front Camera',
      cameraTop: 'Top Camera',
      cameraSide: 'Side Camera',
      cameraIso: 'Isometric Camera',
      resetCamera: 'Reset Camera',
      toggleFullscreen: 'Toggle Fullscreen',
    },
    errors: {
      invalidPosition: 'Invalid position',
      invalidConfig: 'Invalid configuration',
      loadError: 'Failed to load',
      saveError: 'Failed to save',
      networkError: 'Network error',
    },
  },
});

// Register Spanish translations
i18nManager.registerLanguage({
  code: 'es',
  name: 'Spanish',
  nativeName: 'Español',
  dictionary: {
    common: {
      save: 'Guardar',
      cancel: 'Cancelar',
      delete: 'Eliminar',
      edit: 'Editar',
      close: 'Cerrar',
      open: 'Abrir',
      export: 'Exportar',
      import: 'Importar',
      copy: 'Copiar',
      paste: 'Pegar',
      undo: 'Deshacer',
      redo: 'Rehacer',
      reset: 'Restablecer',
      clear: 'Limpiar',
      search: 'Buscar',
      filter: 'Filtrar',
      help: 'Ayuda',
      settings: 'Configuración',
    },
    controls: {
      toggleStats: 'Alternar Estadísticas',
      toggleGizmo: 'Alternar Gizmo',
      toggleGrid: 'Alternar Cuadrícula',
      toggleObjects: 'Alternar Objetos',
      toggleAutoRotate: 'Alternar Rotación Automática',
      toggleStars: 'Alternar Estrellas',
      toggleWaypoints: 'Alternar Puntos de Referencia',
      screenshot: 'Tomar Captura',
      cameraFront: 'Cámara Frontal',
      cameraTop: 'Cámara Superior',
      cameraSide: 'Cámara Lateral',
      cameraIso: 'Cámara Isométrica',
      resetCamera: 'Restablecer Cámara',
      toggleFullscreen: 'Alternar Pantalla Completa',
    },
    errors: {
      invalidPosition: 'Posición inválida',
      invalidConfig: 'Configuración inválida',
      loadError: 'Error al cargar',
      saveError: 'Error al guardar',
      networkError: 'Error de red',
    },
  },
});



