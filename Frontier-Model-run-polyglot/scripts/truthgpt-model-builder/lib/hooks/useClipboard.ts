/**
 * Hook useClipboard
 * =================
 * 
 * Hook para trabajar con el portapapeles
 */

import { useState, useCallback } from 'react'
import { copyToClipboard, readFromClipboard, isClipboardAvailable } from '../utils/clipboardUtils'

export interface UseClipboardResult {
  copy: (text: string) => Promise<boolean>
  paste: () => Promise<string | null>
  isAvailable: boolean
  lastCopied: string | null
  error: Error | null
}

/**
 * Hook para usar el portapapeles
 */
export function useClipboard(): UseClipboardResult {
  const [lastCopied, setLastCopied] = useState<string | null>(null)
  const [error, setError] = useState<Error | null>(null)
  const isAvailable = isClipboardAvailable()

  const copy = useCallback(async (text: string): Promise<boolean> => {
    try {
      setError(null)
      const success = await copyToClipboard(text)
      if (success) {
        setLastCopied(text)
      }
      return success
    } catch (err) {
      const error = err instanceof Error ? err : new Error(String(err))
      setError(error)
      return false
    }
  }, [])

  const paste = useCallback(async (): Promise<string | null> => {
    try {
      setError(null)
      return await readFromClipboard()
    } catch (err) {
      const error = err instanceof Error ? err : new Error(String(err))
      setError(error)
      return null
    }
  }, [])

  return {
    copy,
    paste,
    isAvailable,
    lastCopied,
    error
  }
}







