/**
 * Enhanced Keyboard Shortcuts
 * Sistema mejorado de atajos de teclado
 */

export interface KeyboardShortcut {
  id: string
  keys: string[]
  description: string
  category?: string
  action: () => void | Promise<void>
  enabled?: boolean
  global?: boolean
}

export class EnhancedKeyboardShortcuts {
  private shortcuts: Map<string, KeyboardShortcut> = new Map()
  private listeners: Map<string, (event: KeyboardEvent) => void> = new Map()
  private isEnabled: boolean = true

  constructor() {
    if (typeof window !== 'undefined') {
      window.addEventListener('keydown', this.handleKeyDown.bind(this))
    }
  }

  /**
   * Registrar shortcut
   */
  registerShortcut(shortcut: KeyboardShortcut): void {
    this.shortcuts.set(shortcut.id, {
      ...shortcut,
      enabled: shortcut.enabled !== false,
    })

    const keyString = this.keysToString(shortcut.keys)
    if (!this.listeners.has(keyString)) {
      this.listeners.set(keyString, (event: KeyboardEvent) => {
        if (this.isEnabled && shortcut.enabled !== false) {
          // Verificar si está en input/textarea
          const target = event.target as HTMLElement
          if (
            shortcut.global ||
            (target.tagName !== 'INPUT' &&
              target.tagName !== 'TEXTAREA' &&
              !target.isContentEditable)
          ) {
            event.preventDefault()
            shortcut.action()
          }
        }
      })
    }
  }

  /**
   * Eliminar shortcut
   */
  unregisterShortcut(id: string): void {
    const shortcut = this.shortcuts.get(id)
    if (shortcut) {
      const keyString = this.keysToString(shortcut.keys)
      this.listeners.delete(keyString)
      this.shortcuts.delete(id)
    }
  }

  /**
   * Habilitar/deshabilitar shortcut
   */
  setShortcutEnabled(id: string, enabled: boolean): void {
    const shortcut = this.shortcuts.get(id)
    if (shortcut) {
      shortcut.enabled = enabled
    }
  }

  /**
   * Habilitar/deshabilitar todos los shortcuts
   */
  setEnabled(enabled: boolean): void {
    this.isEnabled = enabled
  }

  /**
   * Manejar keydown
   */
  private handleKeyDown(event: KeyboardEvent): void {
    if (!this.isEnabled) return

    const pressedKeys: string[] = []
    if (event.ctrlKey || event.metaKey) pressedKeys.push('Ctrl')
    if (event.shiftKey) pressedKeys.push('Shift')
    if (event.altKey) pressedKeys.push('Alt')
    if (event.key && !['Control', 'Shift', 'Alt', 'Meta'].includes(event.key)) {
      pressedKeys.push(event.key)
    }

    const keyString = this.keysToString(pressedKeys)
    const listener = this.listeners.get(keyString)
    if (listener) {
      listener(event)
    }
  }

  /**
   * Convertir keys a string
   */
  private keysToString(keys: string[]): string {
    return keys
      .map(key => key.toLowerCase())
      .sort()
      .join('+')
  }

  /**
   * Obtener shortcut
   */
  getShortcut(id: string): KeyboardShortcut | undefined {
    return this.shortcuts.get(id)
  }

  /**
   * Obtener todos los shortcuts
   */
  getAllShortcuts(): KeyboardShortcut[] {
    return Array.from(this.shortcuts.values())
  }

  /**
   * Obtener shortcuts por categoría
   */
  getShortcutsByCategory(category: string): KeyboardShortcut[] {
    return Array.from(this.shortcuts.values()).filter(
      s => s.category === category
    )
  }

  /**
   * Buscar shortcuts
   */
  searchShortcuts(query: string): KeyboardShortcut[] {
    const queryLower = query.toLowerCase()
    return Array.from(this.shortcuts.values()).filter(
      s =>
        s.description.toLowerCase().includes(queryLower) ||
        s.keys.some(key => key.toLowerCase().includes(queryLower))
    )
  }

  /**
   * Obtener categorías
   */
  getCategories(): string[] {
    const categories = new Set<string>()
    this.shortcuts.forEach(s => {
      if (s.category) categories.add(s.category)
    })
    return Array.from(categories).sort()
  }

  /**
   * Formatear keys para display
   */
  formatKeys(keys: string[]): string {
    return keys
      .map(key => {
        if (key === 'Ctrl') return 'Ctrl'
        if (key === 'Shift') return 'Shift'
        if (key === 'Alt') return 'Alt'
        if (key === ' ') return 'Space'
        return key.toUpperCase()
      })
      .join(' + ')
  }

  /**
   * Limpiar shortcuts
   */
  clear(): void {
    this.shortcuts.clear()
    this.listeners.clear()
  }

  /**
   * Cleanup
   */
  destroy(): void {
    if (typeof window !== 'undefined') {
      window.removeEventListener('keydown', this.handleKeyDown.bind(this))
    }
    this.clear()
  }
}

// Singleton instance
let enhancedKeyboardShortcutsInstance: EnhancedKeyboardShortcuts | null = null

export function getEnhancedKeyboardShortcuts(): EnhancedKeyboardShortcuts {
  if (!enhancedKeyboardShortcutsInstance) {
    enhancedKeyboardShortcutsInstance = new EnhancedKeyboardShortcuts()
  }
  return enhancedKeyboardShortcutsInstance
}

export default EnhancedKeyboardShortcuts










