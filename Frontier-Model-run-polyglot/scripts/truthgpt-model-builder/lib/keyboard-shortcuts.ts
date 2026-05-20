/**
 * Keyboard Shortcuts Manager
 * Gestiona atajos de teclado para el Constructor Proactivo
 */

export interface KeyboardShortcut {
  keys: string[]
  description: string
  action: () => void
  ctrlKey?: boolean
  shiftKey?: boolean
  altKey?: boolean
  metaKey?: boolean
}

export class KeyboardShortcutsManager {
  private shortcuts: Map<string, KeyboardShortcut> = new Map()
  private isEnabled: boolean = true

  /**
   * Registrar un atajo de teclado
   */
  register(shortcut: KeyboardShortcut): void {
    const key = this.getShortcutKey(shortcut)
    this.shortcuts.set(key, shortcut)
  }

  /**
   * Desregistrar un atajo
   */
  unregister(keys: string[]): void {
    const key = this.getKeyString(keys)
    this.shortcuts.delete(key)
  }

  /**
   * Manejar evento de teclado
   */
  handleKeyDown(event: KeyboardEvent): void {
    if (!this.isEnabled) return

    const key = this.getKeyString([
      event.ctrlKey ? 'Ctrl' : '',
      event.shiftKey ? 'Shift' : '',
      event.altKey ? 'Alt' : '',
      event.metaKey ? 'Meta' : '',
      event.key,
    ].filter(Boolean))

    const shortcut = this.shortcuts.get(key)
    if (shortcut) {
      event.preventDefault()
      event.stopPropagation()
      shortcut.action()
    }
  }

  /**
   * Habilitar/deshabilitar atajos
   */
  setEnabled(enabled: boolean): void {
    this.isEnabled = enabled
  }

  /**
   * Obtener todos los atajos registrados
   */
  getAllShortcuts(): KeyboardShortcut[] {
    return Array.from(this.shortcuts.values())
  }

  /**
   * Obtener atajo por teclas
   */
  getShortcut(keys: string[]): KeyboardShortcut | undefined {
    const key = this.getKeyString(keys)
    return this.shortcuts.get(key)
  }

  /**
   * Limpiar todos los atajos
   */
  clear(): void {
    this.shortcuts.clear()
  }

  /**
   * Obtener clave de shortcut
   */
  private getShortcutKey(shortcut: KeyboardShortcut): string {
    return this.getKeyString(shortcut.keys)
  }

  /**
   * Convertir array de teclas a string
   */
  private getKeyString(keys: string[]): string {
    return keys
      .filter(k => k && k.trim())
      .map(k => k.toLowerCase())
      .sort()
      .join('+')
  }
}

// Singleton instance
let shortcutsManagerInstance: KeyboardShortcutsManager | null = null

export function getKeyboardShortcutsManager(): KeyboardShortcutsManager {
  if (!shortcutsManagerInstance) {
    shortcutsManagerInstance = new KeyboardShortcutsManager()
  }
  return shortcutsManagerInstance
}

export default KeyboardShortcutsManager










