import { useEffect } from 'react'

interface KeyboardShortcut {
  key: string
  ctrl?: boolean
  shift?: boolean
  alt?: boolean
  callback: () => void
}

export const useKeyboardShortcut = (shortcut: KeyboardShortcut) => {
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      const { key, ctrlKey, shiftKey, altKey } = event

      if (
        key.toLowerCase() === shortcut.key.toLowerCase() &&
        (shortcut.ctrl === undefined || shortcut.ctrl === ctrlKey) &&
        (shortcut.shift === undefined || shortcut.shift === shiftKey) &&
        (shortcut.alt === undefined || shortcut.alt === altKey)
      ) {
        // Don't trigger if user is typing in an input
        const target = event.target as HTMLElement
        if (
          target.tagName === 'INPUT' ||
          target.tagName === 'TEXTAREA' ||
          target.isContentEditable
        ) {
          return
        }

        event.preventDefault()
        shortcut.callback()
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [shortcut])
}




