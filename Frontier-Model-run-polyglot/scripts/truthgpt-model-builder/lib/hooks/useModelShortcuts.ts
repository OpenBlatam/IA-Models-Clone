/**
 * Hook para atajos de teclado y comandos rápidos
 * ===============================================
 */

import { useEffect, useCallback, useRef } from 'react'

export interface ModelShortcut {
  key: string
  ctrl?: boolean
  alt?: boolean
  shift?: boolean
  action: () => void
  description: string
}

export interface UseModelShortcutsOptions {
  enableShortcuts?: boolean
  onShortcut?: (shortcut: string) => void
}

export interface UseModelShortcutsResult {
  registerShortcut: (shortcut: ModelShortcut) => void
  unregisterShortcut: (key: string) => void
  getShortcuts: () => ModelShortcut[]
}

/**
 * Hook para atajos de teclado relacionados con modelos
 */
export function useModelShortcuts(
  callbacks: {
    onCreateModel?: () => void
    onShowHistory?: () => void
    onShowTemplates?: () => void
    onShowAnalytics?: () => void
    onClearHistory?: () => void
    onExportData?: () => void
  },
  options: UseModelShortcutsOptions = {}
): UseModelShortcutsResult {
  const { enableShortcuts = true, onShortcut } = options
  const shortcutsRef = useRef<ModelShortcut[]>([])

  // Atajos predefinidos
  const defaultShortcuts: ModelShortcut[] = [
    {
      key: 'n',
      ctrl: true,
      action: () => callbacks.onCreateModel?.(),
      description: 'Crear nuevo modelo'
    },
    {
      key: 'h',
      ctrl: true,
      action: () => callbacks.onShowHistory?.(),
      description: 'Mostrar historial'
    },
    {
      key: 't',
      ctrl: true,
      action: () => callbacks.onShowTemplates?.(),
      description: 'Mostrar plantillas'
    },
    {
      key: 'a',
      ctrl: true,
      shift: true,
      action: () => callbacks.onShowAnalytics?.(),
      description: 'Mostrar analytics'
    },
    {
      key: 'Delete',
      ctrl: true,
      shift: true,
      action: () => callbacks.onClearHistory?.(),
      description: 'Limpiar historial'
    },
    {
      key: 'e',
      ctrl: true,
      shift: true,
      action: () => callbacks.onExportData?.(),
      description: 'Exportar datos'
    }
  ]

  useEffect(() => {
    shortcutsRef.current = [...defaultShortcuts]
  }, [])

  const handleKeyDown = useCallback((event: KeyboardEvent) => {
    if (!enableShortcuts) return

    const pressedKey = event.key.toLowerCase()
    const ctrl = event.ctrlKey || event.metaKey
    const alt = event.altKey
    const shift = event.shiftKey

    const shortcut = shortcutsRef.current.find(s => {
      return s.key.toLowerCase() === pressedKey &&
             (s.ctrl === undefined || s.ctrl === ctrl) &&
             (s.alt === undefined || s.alt === alt) &&
             (s.shift === undefined || s.shift === shift)
    })

    if (shortcut) {
      event.preventDefault()
      shortcut.action()
      onShortcut?.(shortcut.key)
    }
  }, [enableShortcuts, onShortcut])

  useEffect(() => {
    if (enableShortcuts) {
      window.addEventListener('keydown', handleKeyDown)
      return () => window.removeEventListener('keydown', handleKeyDown)
    }
  }, [enableShortcuts, handleKeyDown])

  const registerShortcut = useCallback((shortcut: ModelShortcut) => {
    shortcutsRef.current = [...shortcutsRef.current, shortcut]
  }, [])

  const unregisterShortcut = useCallback((key: string) => {
    shortcutsRef.current = shortcutsRef.current.filter(s => s.key !== key)
  }, [])

  const getShortcuts = useCallback(() => {
    return [...shortcutsRef.current]
  }, [])

  return {
    registerShortcut,
    unregisterShortcut,
    getShortcuts
  }
}

