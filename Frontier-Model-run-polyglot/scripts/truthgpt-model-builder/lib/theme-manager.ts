/**
 * Theme Manager
 * Gestor de temas (oscuro/claro)
 */

export type Theme = 'dark' | 'light' | 'auto'

export class ThemeManager {
  private currentTheme: Theme = 'dark'
  private storageKey: string = 'proactive-builder-theme'

  constructor() {
    this.loadTheme()
  }

  /**
   * Cargar tema desde almacenamiento
   */
  private loadTheme(): void {
    try {
      if (typeof window !== 'undefined' && window.localStorage) {
        const saved = window.localStorage.getItem(this.storageKey)
        if (saved && (saved === 'dark' || saved === 'light' || saved === 'auto')) {
          this.currentTheme = saved as Theme
        }
      }
    } catch (error) {
      console.error('Error loading theme:', error)
    }
    this.applyTheme()
  }

  /**
   * Aplicar tema
   */
  private applyTheme(): void {
    if (typeof window === 'undefined') return

    const root = document.documentElement
    const effectiveTheme = this.getEffectiveTheme()

    if (effectiveTheme === 'dark') {
      root.classList.add('dark')
      root.classList.remove('light')
    } else {
      root.classList.add('light')
      root.classList.remove('dark')
    }
  }

  /**
   * Obtener tema efectivo
   */
  getEffectiveTheme(): 'dark' | 'light' {
    if (this.currentTheme === 'auto') {
      if (typeof window !== 'undefined') {
        return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
      }
      return 'dark'
    }
    return this.currentTheme
  }

  /**
   * Establecer tema
   */
  setTheme(theme: Theme): void {
    this.currentTheme = theme
    this.applyTheme()
    
    try {
      if (typeof window !== 'undefined' && window.localStorage) {
        window.localStorage.setItem(this.storageKey, theme)
      }
    } catch (error) {
      console.error('Error saving theme:', error)
    }
  }

  /**
   * Obtener tema actual
   */
  getTheme(): Theme {
    return this.currentTheme
  }

  /**
   * Toggle entre oscuro y claro
   */
  toggleTheme(): void {
    const effective = this.getEffectiveTheme()
    this.setTheme(effective === 'dark' ? 'light' : 'dark')
  }

  /**
   * Obtener preferencia del sistema
   */
  getSystemPreference(): 'dark' | 'light' {
    if (typeof window !== 'undefined') {
      return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
    }
    return 'dark'
  }

  /**
   * Obtener tema aplicado (efectivo)
   */
  getAppliedTheme(): 'dark' | 'light' {
    return this.getEffectiveTheme()
  }

  /**
   * Suscribirse a cambios de tema
   */
  onThemeChange(callback: (theme: Theme) => void): () => void {
    // Simple implementation - in a real app, you'd use an event emitter
    const originalSetTheme = this.setTheme.bind(this)
    this.setTheme = (theme: Theme) => {
      originalSetTheme(theme)
      callback(theme)
    }
    return () => {
      this.setTheme = originalSetTheme
    }
  }

  /**
   * Limpiar (resetear a tema por defecto)
   */
  clear(): void {
    this.currentTheme = 'dark'
    if (typeof window !== 'undefined' && window.localStorage) {
      window.localStorage.removeItem(this.storageKey)
    }
    this.applyTheme()
  }
}

// Singleton instance
let themeManagerInstance: ThemeManager | null = null

export function getThemeManager(): ThemeManager {
  if (!themeManagerInstance) {
    themeManagerInstance = new ThemeManager()
  }
  return themeManagerInstance
}

export default ThemeManager


