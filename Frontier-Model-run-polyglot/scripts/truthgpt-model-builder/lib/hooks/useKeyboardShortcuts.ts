import { useEffect } from 'react'

interface KeyboardShortcut {
  keys: string[]
  callback: () => void
  description?: string
}

export function useKeyboardShortcuts(shortcuts: KeyboardShortcut[]) {
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      shortcuts.forEach(({ keys, callback }) => {
        const ctrl = keys.includes('Ctrl') && (e.ctrlKey || e.metaKey)
        const shift = keys.includes('Shift') && e.shiftKey
        const alt = keys.includes('Alt') && e.altKey
        const key = keys.find(k => !['Ctrl', 'Shift', 'Alt'].includes(k))

        if (
          (keys.includes('Ctrl') ? ctrl : !e.ctrlKey && !e.metaKey) &&
          (keys.includes('Shift') ? shift : !e.shiftKey) &&
          (keys.includes('Alt') ? alt : !e.altKey) &&
          key &&
          e.key.toLowerCase() === key.toLowerCase()
        ) {
          e.preventDefault()
          callback()
        }
      })
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [shortcuts])
}

